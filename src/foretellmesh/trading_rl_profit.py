"""Train fee-conditioned allocation policies to maximize net equity; no cadence rule."""
import argparse
from collections import Counter
import math
from pathlib import Path
import shutil
import statistics
import time

from .data import sha256_file, strict_json
from .evaluation import json_text
from .market_development import rows
from .schema import ValidationError, iso
from .sft_data import jsonl
from .trading_rl import artifacts, freeze_data, runtime
from .trading_rl_data import prepare, settlements
from .trading_rl_retrain import fingerprint, parameter_delta, load_spec as load_prior_spec

FAMILIES = ('fixed_fee', 'variable_fee')
SOURCE_FILES = ('trading_rl_profit.py', 'trading_rl_profit_env.py', 'trading_rl_retrain.py',
                'trading_rl_reward.py', 'trading_rl_execution.py', 'trading_rl_execution_env.py')
SCENARIOS = {
    'frictionless_reference': {'entry_price_premium': '0', 'fee_fraction': '0'},
    'fee_zero': {'entry_price_premium': '0.01', 'fee_fraction': '0'},
    'cost_assumption': {'entry_price_premium': '0.01', 'fee_fraction': '0.01'},
    'fee_interpolation': {'entry_price_premium': '0.01', 'fee_fraction': '0.015'},
    'fee_high': {'entry_price_premium': '0.01', 'fee_fraction': '0.02'},
    'cost_stress': {'entry_price_premium': '0.03', 'fee_fraction': '0.02'},
}


def load_spec(reference, path):
    spec = strict_json(path.read_text())
    fixed = {'schema_version': '1', 'partition': 'train', 'periodic_activity_required': False,
        'activity_penalty_usd': '0', 'objective': 'maximize_after_cost_terminal_equity',
        'reward_version': 'net_liquidation_equity_delta_usd_v1', 'order_ttl_seconds': 3600,
        'minimum_entry_buffer_seconds': 86400, 'observation_version': 'profit_fee_and_premium_no_cadence_v1',
        'seeds': [7, 17, 27], 'primary_seed': 7,
        'training_cost_schedules': {'fixed_fee': ['cost_assumption'], 'variable_fee': ['cost_assumption', 'fee_zero', 'fee_high']},
        'scenarios': SCENARIOS, 'initialization': 'fresh_paired_between_fee_families',
        'checkpoint_selection': 'last_fixed_step_all_seeds_no_selection',
        'fixed_controls': ['cash_only', 'mean_reversion'], 'validation_replayed': False,
        'final_test_opened': False, 'foundation_model_updated': False, 'llm_calls': 0,
        'real_orders_sent': 0, 'default_promotion': False}
    if set(spec) != set(fixed) | {'run_name', 'ppo', 'reference_hashes'} or any(
            type(spec[k]) is not type(v) or spec[k] != v for k, v in fixed.items()):
        raise ValidationError('unsupported profit experiment specification')
    if set(spec['reference_hashes']) != {'config.json', 'report.json', 'training_report.json', 'audit.json'}:
        raise ValidationError('missing prior experiment bindings')
    for name, digest in spec['reference_hashes'].items():
        if sha256_file(reference / name) != digest:
            raise ValidationError('profit reference changed: ' + name)
    prior = strict_json((reference / 'report.json').read_text())
    audited = strict_json((reference / 'audit.json').read_text())
    if (prior['status'] != 'completed' or prior['partition'] != 'train' or audited['status'] != 'passed'
            or audited['report_sha256'] != sha256_file(reference / 'report.json')
            or prior['plan_sha256'] != sha256_file(reference / 'plan.json')):
        raise ValidationError('audited retraining reference required')
    for name in SOURCE_FILES[2:]:
        key = 'source_snapshot/foretellmesh/' + name
        if (sha256_file(reference / key) != prior['artifact_hashes'][key]
                or sha256_file(Path(__file__).with_name(name)) != prior['artifact_hashes'][key]):
            raise ValidationError('reference source changed: ' + name)
    plan = strict_json((reference / 'plan.json').read_text())
    _, config = load_prior_spec(Path(plan['paths']['source']), Path(plan['paths']['execution']), reference / 'config.json')
    if spec['ppo'] != {**config['ppo'], 'gamma': 1.0}:
        raise ValidationError('profit budget must match prior budget with undiscounted returns')
    trained = strict_json((reference / 'training_report.json').read_text())
    for seed in spec['seeds']:
        name = f'net_equity_{seed}'
        if sha256_file(reference / 'models' / name / 'trained.zip') != trained['models'][name]['trained_sha256']:
            raise ValidationError('legacy policy changed')
    return spec, config


def make_env(data, labels, config, spec, costs, **kwargs):
    from .trading_rl_profit_env import ProfitAllocationEnv
    return ProfitAllocationEnv(data, labels, config['environment'], costs, order_ttl_seconds=spec['order_ttl_seconds'],
        minimum_entry_buffer_seconds=spec['minimum_entry_buffer_seconds'], **kwargs)


def train_models(root, data, labels, config, spec):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    trained, paired = {}, {}
    h = spec['ppo']
    for family in FAMILIES:
        schedule = [spec['scenarios'][name] for name in spec['training_cost_schedules'][family]]
        for seed in spec['seeds']:
            name = f'{family}_{seed}'
            dest = root / 'models' / name
            dest.mkdir(parents=True)
            started = time.perf_counter()
            env = make_env(data, labels, config, spec, schedule[0], cost_schedule=schedule)
            model = MaskablePPO('MlpPolicy', env, seed=seed, device='cpu', verbose=0,
                policy_kwargs={'net_arch': h['net_arch']}, **{k: h[k] for k in (
                    'n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef')})
            model.set_logger(configure(str(dest), ['csv']))
            model.foretellmesh_observation_version = spec['observation_version']
            initial_fingerprint = fingerprint(model)
            if initial_fingerprint != paired.setdefault(seed, initial_fingerprint):
                raise ValidationError('fee curriculum initializations are not paired')
            initial = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
            model.save(dest / 'initial.zip')
            episodes, fee_steps = [], Counter()

            class Capture(BaseCallback):
                def _on_step(self):
                    for info in self.locals['infos']:
                        fee_steps[info['execution_costs']['fee_fraction']] += 1
                        if 'episode_summary' in info:
                            episodes.append({'timesteps': self.num_timesteps, **info['episode_summary']})
                    if self.num_timesteps % 8192 == 0:
                        print(f'{name}: {self.num_timesteps}/{h["total_timesteps"]} steps', flush=True)
                    return True

            model.learn(total_timesteps=h['total_timesteps'], callback=Capture())
            delta = parameter_delta(model, initial)
            if model.num_timesteps != h['total_timesteps'] or not math.isfinite(delta) or delta <= 0:
                raise ValidationError('profit policy parameter update or budget failed')
            model.save(dest / 'trained.zip')
            (dest / 'episodes.jsonl').write_text(jsonl(episodes))
            trained[name] = {'family': family, 'seed': seed, 'actual_timesteps': model.num_timesteps,
                'episodes': len(episodes), 'seconds': time.perf_counter() - started,
                'parameters': sum(p.numel() for p in model.policy.parameters()), 'parameter_update_l2': delta,
                'initial_parameters_sha256': initial_fingerprint, 'trained_parameters_sha256': fingerprint(model),
                'initial_sha256': sha256_file(dest / 'initial.zip'), 'trained_sha256': sha256_file(dest / 'trained.zip'),
                'optimizer': type(model.policy.optimizer).__name__, 'optimizer_defaults': model.policy.optimizer.defaults,
                'timesteps_by_fee_fraction': dict(sorted(fee_steps.items())),
                'last_optimizer_metrics': {k: float(v) if math.isfinite(float(v)) else None
                    for k, v in model.logger.name_to_value.items() if k.startswith('train/')},
                'last_training_episode': episodes[-1] if episodes else None}
            model.logger.close()
            env.close()
            print(f'{name} frozen in {trained[name]["seconds"]:.1f}s; parameter delta {delta:.6f}', flush=True)
    return trained


def arms(spec):
    return (spec['fixed_controls'] + [f'legacy_{s}' for s in spec['seeds']]
            + [f'{family}_{s}' for family in FAMILIES for s in spec['seeds']])


def checkpoint_models(root, reference, spec):
    from sb3_contrib import MaskablePPO
    tr = strict_json((root / 'training_report.json').read_text())
    expected = {f'{family}_{s}' for family in FAMILIES for s in spec['seeds']}
    if tr['status'] != 'completed' or set(tr['models']) != expected:
        raise ValidationError('all profit policies must be frozen before replay')
    models = {f'legacy_{s}': MaskablePPO.load(reference / f'models/net_equity_{s}/trained.zip', device='cpu') for s in spec['seeds']}
    paired = {}
    for name in sorted(expected):
        row = tr['models'][name]
        dest = root / 'models' / name
        for stage in ('initial', 'trained'):
            if sha256_file(dest / f'{stage}.zip') != row[f'{stage}_sha256']:
                raise ValidationError('profit checkpoint changed')
        model = MaskablePPO.load(dest / 'trained.zip', device='cpu')
        initial = MaskablePPO.load(dest / 'initial.zip', device='cpu')
        initial_hash = fingerprint(initial)
        if (fingerprint(model) != row['trained_parameters_sha256'] or initial_hash != row['initial_parameters_sha256']
                or initial_hash != paired.setdefault(row['seed'], initial_hash)
                or row['actual_timesteps'] != spec['ppo']['total_timesteps'] or model.num_timesteps != row['actual_timesteps']
                or getattr(model, 'foretellmesh_observation_version', None) != spec['observation_version']
                or getattr(initial, 'foretellmesh_observation_version', None) != spec['observation_version']
                or model.gamma != 1.0 or parameter_delta(model, initial.policy.state_dict()) != row['parameter_update_l2']
                or row['parameter_update_l2'] <= 0
                or sum(row['timesteps_by_fee_fraction'].values()) != row['actual_timesteps']):
            raise ValidationError('profit training evidence changed')
        models[name] = model
    return models


def evaluate_one(data, labels, config, spec, costs, arm, model=None):
    env = make_env(data, labels, config, spec, costs, legacy_observation=arm.startswith('legacy_'), record=True)
    obs, _ = env.reset()
    pending = []
    while not env.done:
        pending.append({'time': iso(env.time), 'orders': env.pending_snapshot()})
        action = (int(model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0]) if model is not None
                  else 0 if arm == 'cash_only' else env.fixed_action())
        obs, _, _, _, _ = env.step(action)
    value = {'metrics': env.summary(), 'lifecycle': env.lifecycle_summary(), 'ledger': env.ledger,
             'decisions': env.decisions, 'equity_curve': env.curve, 'pending': pending}
    env.close()
    return value


def replay(root, reference, data, labels, config, spec, *, reproduce=False):
    models = checkpoint_models(root, reference, spec)
    result = {}
    key = str(spec['order_ttl_seconds'])
    for scenario, costs in spec['scenarios'].items():
        result[scenario] = {key: {}}
        dest = root / scenario / key
        if not reproduce:
            dest.mkdir(parents=True)
        for arm in arms(spec):
            value = evaluate_one(data, labels, config, spec, costs, arm, models.get(arm))
            result[scenario][key][arm] = {k: value[k] for k in ('metrics', 'lifecycle')}
            for field in ('ledger', 'decisions', 'equity_curve', 'pending'):
                path = dest / f'{arm}.{field}.jsonl'
                if reproduce:
                    if rows(path) != value[field]:
                        raise ValidationError('profit replay differs: ' + str(path))
                else:
                    path.write_text(jsonl(value[field]))
        print(f'{"Reproduced" if reproduce else "Evaluated"} {scenario}: {len(arms(spec))} policies', flush=True)
    return result


def aggregates(result, spec):
    summary = {}
    for scenario, ttls in result.items():
        values = ttls[str(spec['order_ttl_seconds'])]
        summary[scenario] = {}
        for family in ('legacy', *FAMILIES):
            metrics = [values[f'{family}_{s}']['metrics'] for s in spec['seeds']]
            cash = [float(m['final_cash']) for m in metrics]
            summary[scenario][family] = {'mean_final_cash': statistics.fmean(cash), 'min_final_cash': min(cash),
                'max_final_cash': max(cash), 'std_final_cash': statistics.pstdev(cash),
                'profitable_seeds': sum(m['outperformed_cash'] for m in metrics),
                'risk_passes': sum(m['policy_requirements_met'] for m in metrics),
                'mean_entry_count': statistics.fmean(m['entry_count'] for m in metrics),
                'mean_fees_paid': statistics.fmean(float(m['fees_paid']) for m in metrics),
                'mean_net_pnl_vs_cash': statistics.fmean(cash) - 100,
                'worst_drawdown_fraction': max(float(m['max_drawdown_fraction']) for m in metrics)}
    return summary


def run(dataset, store, reference, config_path, output):
    from .trading_rl_profit_env import observation_schema
    if output.exists():
        raise ValidationError('profit experiment output already exists')
    spec, config = load_spec(reference, config_path)
    resources = runtime(config)
    started = time.perf_counter()
    data = prepare(dataset, store, 'train', config)
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / 'config.json')
    shutil.copytree(Path(__file__).parent, output / 'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    frozen = freeze_data(output, data)
    for name in ('train.catalog.jsonl', 'train.features.jsonl'):
        if sha256_file(output / name) != sha256_file(reference / name):
            raise ValidationError('profit experiment training data changed')
    plan = {'spec': spec, 'original_config': config, 'execution_scenarios': spec['scenarios'], 'resources': resources,
        'training_data': frozen, 'observation_schema': observation_schema(),
        'paths': {k: str(v.resolve()) for k, v in dict(dataset=dataset, store=store, reference=reference).items()},
        'artifact_hashes': artifacts(output)}
    (output / 'plan.json').write_text(json_text(plan))
    labels = settlements(dataset, data)
    trained = train_models(output, data, labels, config, spec)
    tr = {'status': 'completed', 'plan_sha256': sha256_file(output / 'plan.json'), 'ppo': spec['ppo'], 'models': trained,
          'validation_used_for_gradients': False, 'final_test_opened': False, 'artifact_hashes': artifacts(output)}
    (output / 'training_report.json').write_text(json_text(tr))
    (output / 'evaluation_plan.json').write_text(json_text({'training_report_sha256': sha256_file(output / 'training_report.json'),
        'partition': 'train', 'arms': arms(spec), 'scenarios': spec['scenarios']}))
    result = replay(output, reference, data, labels, config, spec)
    report = {'status': 'completed', 'plan_sha256': sha256_file(output / 'plan.json'), 'partition': 'train',
        'training_report_sha256': sha256_file(output / 'training_report.json'),
        'evaluation_plan_sha256': sha256_file(output / 'evaluation_plan.json'), 'training_data': frozen,
        'scenarios': result, 'aggregates': aggregates(result, spec), 'elapsed_seconds': time.perf_counter() - started,
        'training_performed': 'six_fee_conditioned_masked_ppo_networks', 'periodic_activity_required': False,
        'validation_replayed': False, 'final_test_opened': False, 'llm_calls': 0, 'foundation_model_updated': False,
        'real_orders_sent': 0, 'default_promotion': False,
        'limitations': ['Training-only adaptive study on six event groups, not unseen-event profitability evidence.',
            'A new fee value is an execution sensitivity check, not an independent event test.',
            'Flat notional fees are hypothetical, known and fixed within each episode; no historical venue fee schedule is reconstructed.',
            'Price premium is distinct from fees. First-print full fills do not prove executable order-book liquidity.',
            'Artificial scope/calendar countdowns remain research features, not verified public expiries.',
            'No activity requirement, but cash/exposure limits, cooldown, daily entry cap and automatic risk exits remain.',
            'Legacy policies use their old observation schema with cadence disabled. New policies replace its two cadence fields with explicit costs.',
            'Qwen/LoRAs and agent research are not trained or called. Cash holding is allowed and earns zero interest.'],
        'artifact_hashes': artifacts(output)}
    (output / 'report.json').write_text(json_text(report))
    return report


def audit(root):
    from .trading_rl_profit_env import observation_schema
    report = strict_json((root / 'report.json').read_text())
    for name in ('plan', 'training_report', 'evaluation_plan'):
        if sha256_file(root / f'{name}.json') != report[f'{name}_sha256']:
            raise ValidationError('profit experiment freeze changed')
    for name, digest in report['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root / name) != digest:
            raise ValidationError('profit artifact changed: ' + name)
    for name in SOURCE_FILES:
        if sha256_file(Path(__file__).with_name(name)) != sha256_file(root / 'source_snapshot/foretellmesh' / name):
            raise ValidationError('use frozen profit experiment source for audit')
    plan = strict_json((root / 'plan.json').read_text())
    tr = strict_json((root / 'training_report.json').read_text())
    ep = strict_json((root / 'evaluation_plan.json').read_text())
    paths = {k: Path(v) for k, v in plan['paths'].items()}
    spec, config = load_spec(paths['reference'], root / 'config.json')
    if (spec != plan['spec'] or config != plan['original_config'] or spec['scenarios'] != plan['execution_scenarios']
            or observation_schema() != plan['observation_schema'] or tr['plan_sha256'] != report['plan_sha256']
            or tr['ppo'] != spec['ppo'] or ep != {'training_report_sha256': report['training_report_sha256'],
                'partition': 'train', 'arms': arms(spec), 'scenarios': spec['scenarios']}):
        raise ValidationError('profit protocol changed')
    runtime(config)
    data = prepare(paths['dataset'], paths['store'], 'train', config)
    if (data.bindings != plan['training_data']['bindings'] or data.catalog != rows(root / 'train.catalog.jsonl')
            or data.feature_rows() != rows(root / 'train.features.jsonl')):
        raise ValidationError('profit data changed')
    result = replay(root, paths['reference'], data, settlements(paths['dataset'], data), config, spec, reproduce=True)
    if result != report['scenarios'] or aggregates(result, spec) != report['aggregates']:
        raise ValidationError('profit results differ')
    return {'status': 'passed', 'report_sha256': sha256_file(root / 'report.json'), 'partition': 'train',
            'arms_reproduced': sum(len(a) for s in result.values() for a in s.values()),
            'paired_initializations_verified': True, 'parameter_updates_verified': True,
            'periodic_activity_required': False, 'validation_replayed': False, 'final_test_opened': False, 'real_orders_sent': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    start = sub.add_parser('run')
    for name in ('dataset', 'store', 'reference', 'config', 'output'):
        start.add_argument('--' + name, type=Path, required=True)
    sub.add_parser('audit').add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        result = run(args.dataset, args.store, args.reference, args.config, args.output)
        print(json_text({'status': result['status'], 'seconds': result['elapsed_seconds']}))
    else:
        result = audit(args.run)
        (args.run / 'audit.json').write_text(json_text(result))
        print(json_text(result))


if __name__ == '__main__':
    main()
