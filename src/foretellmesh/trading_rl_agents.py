"""Paired PPO ablation: causal price features versus frozen multi-agent content."""
import argparse
from collections import Counter
from pathlib import Path
import math
import shutil
import statistics
import time

from .data import sha256_file, strict_json
from .evaluation import json_text
from .market_development import rows
from .schema import ValidationError, iso
from .sft_data import jsonl
from .trading_agent_signals import verify as verify_signals
from .trading_rl import artifacts, freeze_data, runtime
from .trading_rl_data import prepare, settlements
from .trading_rl_profit import SCENARIOS
from .trading_rl_retrain import fingerprint, parameter_delta

FAMILIES = ('price_control', 'market_anchor', 'multi_agent')


def read_spec(path):
    spec = strict_json(path.read_text())
    fixed = {'partition': 'train', 'families': list(FAMILIES), 'seeds': [7, 17, 27], 'primary_seed': 7,
        'periodic_activity_required': False, 'activity_penalty_usd': '0',
        'objective': 'maximize_after_cost_terminal_equity', 'reward_version': 'net_liquidation_equity_delta_usd_v1',
        'observation_version': 'profit_fee_with_agent_signals_v1', 'order_ttl_seconds': 3600,
        'minimum_entry_buffer_seconds': 86400, 'training_cost_schedule': ['cost_assumption', 'fee_zero', 'fee_high'],
        'scenarios': SCENARIOS, 'controls': ['cash_only', 'mean_reversion'],
        'checkpoint_selection': 'last_fixed_step_all_seeds_no_selection',
        'content_ablation': 'zero_content_keep_identical_availability_age_and_network_shape',
        'anchor_control': 'historical_market_probability_at_identical_cutoff_no_expert_fields',
        'validation_replayed': False, 'final_test_opened': False, 'foundation_model_updated': False,
        'real_orders_sent': 0, 'default_promotion': False}
    if (set(spec) != set(fixed) | {'run_name', 'ppo'}
            or any(type(spec[k]) is not type(v) or spec[k] != v for k, v in fixed.items())):
        raise ValidationError('unsupported agent allocation experiment')
    return spec


def make_env(data, labels, config, spec, index, costs, family, **kwargs):
    from .trading_rl_agents_env import AgentAllocationEnv
    return AgentAllocationEnv(data, labels, config['environment'], costs, signals=index,
        content_enabled=family == 'multi_agent', anchor_only=family == 'market_anchor', order_ttl_seconds=spec['order_ttl_seconds'],
        minimum_entry_buffer_seconds=spec['minimum_entry_buffer_seconds'], **kwargs)


def train_models(root, data, labels, config, spec, index):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    h = spec['ppo']; paired = {}; trained = {}
    schedule = [spec['scenarios'][s] for s in spec['training_cost_schedule']]
    for family in spec['families']:
        for seed in spec['seeds']:
            name = f'{family}_{seed}'; dest = root / 'models' / name; dest.mkdir(parents=True)
            env = make_env(data, labels, config, spec, index, schedule[0], family, cost_schedule=schedule)
            model = MaskablePPO('MlpPolicy', env, seed=seed, device='cpu', verbose=0,
                policy_kwargs={'net_arch': h['net_arch']}, **{k: h[k] for k in (
                    'n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef')})
            model.foretellmesh_observation_version = spec['observation_version']
            model.set_logger(configure(str(dest), ['csv']))
            initial_hash = fingerprint(model)
            if initial_hash != paired.setdefault(seed, initial_hash):
                raise ValidationError('unpaired initial parameters')
            initial = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
            model.save(dest / 'initial.zip')
            episodes, costs_seen = [], Counter()
            class Capture(BaseCallback):
                def _on_step(self):
                    for info in self.locals['infos']:
                        costs_seen[info['execution_costs']['fee_fraction']] += 1
                        if 'episode_summary' in info:
                            episodes.append({'timesteps': self.num_timesteps, **info['episode_summary']})
                    if self.num_timesteps % 8192 == 0:
                        print(f'{name}: {self.num_timesteps}/{h["total_timesteps"]}', flush=True)
                    return True
            started = time.perf_counter()
            model.learn(total_timesteps=h['total_timesteps'], callback=Capture())
            delta = parameter_delta(model, initial)
            if model.num_timesteps != h['total_timesteps'] or not math.isfinite(delta) or delta <= 0:
                raise ValidationError('optimizer update/budget failed')
            model.save(dest / 'trained.zip'); (dest / 'episodes.jsonl').write_text(jsonl(episodes))
            trained[name] = {'family': family, 'seed': seed, 'actual_timesteps': model.num_timesteps,
                'episodes': len(episodes), 'seconds': time.perf_counter() - started,
                'parameters': sum(p.numel() for p in model.policy.parameters()),
                'initial_parameters_sha256': initial_hash, 'trained_parameters_sha256': fingerprint(model),
                'initial_sha256': sha256_file(dest / 'initial.zip'), 'trained_sha256': sha256_file(dest / 'trained.zip'),
                'parameter_update_l2': delta, 'optimizer': type(model.policy.optimizer).__name__,
                'optimizer_defaults': model.policy.optimizer.defaults, 'timesteps_by_fee_fraction': dict(costs_seen),
                'last_optimizer_metrics': {k: float(v) if math.isfinite(float(v)) else None
                    for k, v in model.logger.name_to_value.items() if k.startswith('train/')}}
            model.logger.close(); env.close()
            print(f'{name}: frozen, {trained[name]["seconds"]:.1f}s', flush=True)
    return trained


def load_models(root, spec):
    from sb3_contrib import MaskablePPO
    trained = strict_json((root / 'training_report.json').read_text())['models']; result = {}; paired = {}
    expected = {f'{family}_{seed}' for family in spec['families'] for seed in spec['seeds']}
    if set(trained) != expected:
        raise ValidationError('incomplete frozen policy population')
    for name, row in trained.items():
        dest = root / 'models' / name
        for phase in ('initial', 'trained'):
            if sha256_file(dest / f'{phase}.zip') != row[f'{phase}_sha256']:
                raise ValidationError('checkpoint changed')
        initial = MaskablePPO.load(dest / 'initial.zip', device='cpu')
        model = MaskablePPO.load(dest / 'trained.zip', device='cpu')
        ih = fingerprint(initial)
        if (ih != paired.setdefault(row['seed'], ih) or ih != row['initial_parameters_sha256']
                or fingerprint(model) != row['trained_parameters_sha256']
                or parameter_delta(model, initial.policy.state_dict()) != row['parameter_update_l2']
                or row['parameter_update_l2'] <= 0 or model.num_timesteps != spec['ppo']['total_timesteps']
                or row['actual_timesteps'] != model.num_timesteps or model.gamma != 1.
                or sum(row['timesteps_by_fee_fraction'].values()) != model.num_timesteps
                or getattr(model, 'foretellmesh_observation_version', None) != spec['observation_version']):
            raise ValidationError('training evidence changed')
        result[name] = model
    return result


def evaluate_one(data, labels, config, spec, index, costs, arm, model=None):
    family = ('multi_agent' if arm.startswith('multi_agent_') else
              'market_anchor' if arm.startswith('market_anchor_') else 'price_control')
    env = make_env(data, labels, config, spec, index, costs, family, record=True)
    obs, _ = env.reset(); pending = []
    while not env.done:
        pending.append({'time': iso(env.time), 'orders': env.pending_snapshot()})
        action = (int(model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0]) if model is not None
                  else 0 if arm == 'cash_only' else env.fixed_action())
        obs, _, _, _, _ = env.step(action)
    result = {'metrics': env.summary(), 'lifecycle': env.lifecycle_summary(), 'ledger': env.ledger,
        'decisions': env.decisions, 'equity_curve': env.curve, 'pending': pending}
    env.close(); return result


def replay(root, data, labels, config, spec, index, *, reproduce=False):
    models = load_models(root, spec); result = {}
    # Masking content at evaluation is an intervention on each frozen Agent policy.
    # It tests use of the channel; it is not an independently trained comparator.
    arms = spec['controls'] + list(models) + [f'masked_agent_{s}' for s in spec['seeds']]
    for scenario, costs in spec['scenarios'].items():
        dest = root / scenario / str(spec['order_ttl_seconds'])
        if not reproduce:
            dest.mkdir(parents=True)
        values = {}
        for arm in arms:
            model = models.get(arm)
            if arm.startswith('masked_agent_'):
                model = models['multi_agent_' + arm.rsplit('_', 1)[1]]
            r = evaluate_one(data, labels, config, spec, index, costs, arm, model)
            values[arm] = {k: r[k] for k in ('metrics', 'lifecycle')}
            for key in ('ledger', 'decisions', 'equity_curve', 'pending'):
                path = dest / f'{arm}.{key}.jsonl'
                if reproduce:
                    if rows(path) != r[key]:
                        raise ValidationError('agent allocation replay differs: ' + str(path))
                else:
                    path.write_text(jsonl(r[key]))
        result[scenario] = {str(spec['order_ttl_seconds']): values}
        print(f'{"Audited" if reproduce else "Evaluated"} {scenario}: {len(arms)} policies', flush=True)
    return result


def aggregates(result, spec):
    out = {}
    for scenario, ttls in result.items():
        values = ttls[str(spec['order_ttl_seconds'])]; out[scenario] = {}
        for family in (*spec['families'], 'masked_agent'):
            mm = [values[f'{family}_{s}']['metrics'] for s in spec['seeds']]
            cash = [float(m['final_cash']) for m in mm]
            out[scenario][family] = {'mean_final_cash': statistics.fmean(cash),
                'min_final_cash': min(cash), 'max_final_cash': max(cash),
                'profitable_seeds': sum(x > 100 for x in cash),
                'mean_entries': statistics.fmean(m['entry_count'] for m in mm),
                'mean_fees': statistics.fmean(float(m['fees_paid']) for m in mm),
                'worst_drawdown_fraction': max(float(m['max_drawdown_fraction']) for m in mm)}
        out[scenario]['paired_cash_differences'] = [
            float(values[f'multi_agent_{s}']['metrics']['final_cash']) - float(values[f'price_control_{s}']['metrics']['final_cash'])
            for s in spec['seeds']]
        if 'market_anchor' in spec['families']:
            out[scenario]['paired_cash_differences_vs_anchor'] = [
                float(values[f'multi_agent_{s}']['metrics']['final_cash']) - float(values[f'market_anchor_{s}']['metrics']['final_cash'])
                for s in spec['seeds']]
    return out


def run(signals, config_path, output):
    from .trading_rl_agents_env import AGENT_FEATURES, SignalIndex, signal_rows
    if output.exists():
        raise ValidationError('allocation output already exists')
    spec = read_spec(config_path)
    c, signal_plan, results = verify_signals(signals)
    config = signal_plan['original_config']
    if spec['ppo'] != {**config['ppo'], 'gamma': 1.0}:
        raise ValidationError('paired PPO settings must match previous experiment')
    resources = runtime(config); started = time.perf_counter()
    paths = {k: Path(v) for k, v in signal_plan['paths'].items()}
    data = prepare(paths['dataset'], paths['store'], 'train', config)
    if data.bindings != signal_plan['data_bindings']:
        raise ValidationError('signal/replay data mismatch')
    references = {r['sample_id']: r['input']['market']['probability'] for r in rows(signals / 'inputs.jsonl')}
    records = signal_rows(results, c['signal_ttl_seconds'], reference_probabilities=references)
    index = SignalIndex(records, {m.market_id for m in data.windows})
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / 'config.json')
    shutil.copytree(Path(__file__).parent, output / 'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    (output / 'agent_signals.jsonl').write_text(jsonl(records))
    frozen = freeze_data(output, data)
    plan = {'spec': spec, 'original_config': config, 'execution_scenarios': spec['scenarios'],
        'resources': resources, 'training_data': frozen, 'agent_feature_schema': list(AGENT_FEATURES),
        'signal_generation_report_sha256': sha256_file(signals / 'generation_report.json'),
        'paths': {**signal_plan['paths'], 'signals': str(signals.resolve())}, 'artifact_hashes': artifacts(output)}
    (output / 'plan.json').write_text(json_text(plan))
    labels = settlements(paths['dataset'], data)
    trained = train_models(output, data, labels, config, spec, index)
    training = {'status': 'completed', 'plan_sha256': sha256_file(output / 'plan.json'),
                'ppo': spec['ppo'], 'models': trained, 'artifact_hashes': artifacts(output)}
    (output / 'training_report.json').write_text(json_text(training))
    evaluation = {'training_report_sha256': sha256_file(output / 'training_report.json'), 'partition': 'train',
                  'scenarios': spec['scenarios'], 'controls': spec['controls'], 'masked_content_intervention': True}
    (output / 'evaluation_plan.json').write_text(json_text(evaluation))
    result = replay(output, data, labels, config, spec, index)
    report = {'status': 'completed', 'partition': 'train', 'training_data': frozen,
        **{f'{name}_sha256': sha256_file(output / f'{name}.json') for name in ('plan', 'training_report', 'evaluation_plan')},
        'scenarios': result, 'aggregates': aggregates(result, spec), 'elapsed_seconds': time.perf_counter() - started,
        'signal_jobs': len(results), 'successful_signals': sum(r['content'] is not None for r in records),
        'signal_markets': len({r['market_id'] for r in records}), 'llm_calls': sum(len(r['calls']) for r in results),
        'foundation_model_updated': False, 'periodic_activity_required': False,
        'validation_replayed': False, 'final_test_opened': False, 'real_orders_sent': 0, 'default_promotion': False,
        'limitations': ['Training-only adaptive study on six event groups; no unseen-event profitability claim.',
            'Frozen Base model may already know historical outcomes from pretraining; timestamp filtering cannot remove parameter-memory contamination.',
            'Weekly research refresh of up to two active markets per event group is sparse, not a weekly trading requirement.',
            'An event-resolution forecast is not a forecast of the next 48-hour price; it is an optional RL observation, not an execution price or guaranteed edge.',
            'Agent confidence is self-reported and uncalibrated. Failed or expired signals are missing, never replaced by outcomes.',
            'The reference candidate ranking, 25-action encoding, entropy coefficient and training budget remain unchanged to isolate Agent information.',
            'A separately trained market-anchor control supplies the original as-of market probability in the same feature slots, without Agent views/confidence.',
            'Content-masked evaluation measures channel dependence and has distribution shift; it is not a separately trained strategy.',
            'Retrospectively admitted contracts/evidence and artificial scope boundaries remain research limitations.',
            'Hypothetical flat notional fees and first-print full fills do not reconstruct historical order books.'],
        'artifact_hashes': artifacts(output)}
    (output / 'report.json').write_text(json_text(report)); return report


def audit(root):
    from .trading_rl_agents_env import AGENT_FEATURES, SignalIndex, signal_rows
    report = strict_json((root / 'report.json').read_text())
    for name in ('plan', 'training_report', 'evaluation_plan'):
        if report[f'{name}_sha256'] != sha256_file(root / f'{name}.json'):
            raise ValidationError('experiment freeze changed')
    for name, digest in report['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root / name) != digest:
            raise ValidationError('allocation artifact changed: ' + name)
    plan = strict_json((root / 'plan.json').read_text()); spec = read_spec(root / 'config.json')
    paths = {k: Path(v) for k, v in plan['paths'].items()}
    c, signal_plan, results = verify_signals(paths['signals'])
    config = signal_plan['original_config']
    if (spec != plan['spec'] or config != plan['original_config']
            or list(AGENT_FEATURES) != plan['agent_feature_schema']
            or sha256_file(paths['signals'] / 'generation_report.json') != plan['signal_generation_report_sha256']):
        raise ValidationError('signal protocol changed')
    runtime(config); data = prepare(paths['dataset'], paths['store'], 'train', config)
    references = {r['sample_id']: r['input']['market']['probability'] for r in rows(paths['signals'] / 'inputs.jsonl')}
    records = signal_rows(results, c['signal_ttl_seconds'], reference_probabilities=references)
    if (records != rows(root / 'agent_signals.jsonl') or data.bindings != plan['training_data']['bindings']
            or data.catalog != rows(root / 'train.catalog.jsonl') or data.feature_rows() != rows(root / 'train.features.jsonl')):
        raise ValidationError('signal or price data changed')
    index = SignalIndex(records, {m.market_id for m in data.windows})
    actual = replay(root, data, settlements(paths['dataset'], data), config, spec, index, reproduce=True)
    if actual != report['scenarios'] or aggregates(actual, spec) != report['aggregates']:
        raise ValidationError('allocation metrics do not reproduce')
    return {'status': 'passed', 'report_sha256': sha256_file(root / 'report.json'),
        'signal_jobs_reproduced': len(results), 'raw_calls_replayed': sum(len(r['calls']) for r in results),
        'arms_reproduced': sum(len(v) for s in actual.values() for v in s.values()),
        'validation_replayed': False, 'final_test_opened': False, 'real_orders_sent': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    start = sub.add_parser('run')
    for name in ('signals', 'config', 'output'):
        start.add_argument('--' + name, type=Path, required=True)
    sub.add_parser('audit').add_argument('--run', type=Path, required=True)
    a = parser.parse_args()
    if a.command == 'run':
        report = run(a.signals, a.config, a.output)
        print(json_text({'status': report['status'], 'aggregates': report['aggregates']}))
    else:
        result = audit(a.run); (a.run / 'audit.json').write_text(json_text(result)); print(json_text(result))


if __name__ == '__main__':
    main()
