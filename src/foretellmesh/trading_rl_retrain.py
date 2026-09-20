"""Paired, fixed-budget PPO retraining in the pending-order simulator (train only)."""
import argparse
import hashlib
import math
from pathlib import Path
import shutil
import statistics
import time

from .data import sha256_file, strict_json
from .evaluation import json_text
from .market_development import rows
from .schema import ValidationError
from .sft_data import jsonl
from .trading_rl import artifacts, freeze_data, runtime
from .trading_rl_data import prepare, settlements
from .trading_rl_execution import evaluate_lifetime, load_spec as load_execution_spec

FAMILIES = ('net_equity', 'net_equity_cadence')
SOURCE_FILES = ('trading_rl_retrain.py', 'trading_rl_reward.py', 'trading_rl_execution.py',
                'trading_rl_execution_env.py')


def load_spec(source: Path, execution: Path, path: Path):
    spec = strict_json(path.read_text())
    fixed = {'schema_version': '1', 'partition': 'train', 'training_scenario': 'cost_assumption',
             'training_order_ttl_seconds': 3600, 'evaluation_order_ttl_seconds': [3600, 14400],
             'scenarios': ['frictionless_reference', 'cost_assumption', 'cost_stress'],
             'seeds': [7, 17, 27], 'primary_seed': 7,
             'rewards': {'net_equity': '0', 'net_equity_cadence': '0.10'},
             'reward_version': 'net_equity_minus_integrated_overdue_hours_v1',
             'initialization': 'fresh_paired_original_seed_parameters',
             'checkpoint_selection': 'last_fixed_step_all_seeds_no_selection',
             'validation_replayed': False, 'final_test_opened': False,
             'foundation_model_updated': False, 'llm_calls': 0, 'real_orders_sent': 0,
             'default_promotion': False}
    if set(spec) != set(fixed) | {'run_name', 'ppo', 'execution_reference_hashes'} or any(
            type(spec[k]) is not type(v) or spec[k] != v for k, v in fixed.items()):
        raise ValidationError('unsupported retraining specification')
    if set(spec['execution_reference_hashes']) != {'config.json', 'report.json', 'audit.json'}:
        raise ValidationError('missing execution reference bindings')
    for name, digest in spec['execution_reference_hashes'].items():
        if sha256_file(execution / name) != digest:
            raise ValidationError('execution reference changed: ' + name)
    previous = strict_json((execution / 'report.json').read_text())
    audited = strict_json((execution / 'audit.json').read_text())
    if (previous['status'] != 'completed' or previous['partition'] != 'train' or audited['status'] != 'passed'
            or audited['report_sha256'] != sha256_file(execution / 'report.json')):
        raise ValidationError('audited training execution study required')
    for name in SOURCE_FILES[2:]:
        key = 'source_snapshot/foretellmesh/' + name
        digest = previous['artifact_hashes'][key]
        if sha256_file(execution / key) != digest or sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValidationError('executor changed since lifetime study')
    _, config = load_execution_spec(source, execution / 'config.json')
    if (spec['ppo'] != config['ppo'] or spec['seeds'] != config['seeds']
            or spec['training_order_ttl_seconds'] != config['environment']['step_seconds']):
        raise ValidationError('paired budget or decision interval differs')
    original = strict_json((source / 'training_report.json').read_text())
    for seed in spec['seeds']:
        if sha256_file(source / f'seed_{seed}/initial.zip') != original['seeds'][str(seed)]['initial_sha256']:
            raise ValidationError('original initialization changed')
    return spec, config


def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        digest.update(json_text([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def parameter_delta(model, initial):
    import torch
    return sum(float(torch.sum((v.detach() - initial[k]) ** 2)) for k, v in model.policy.state_dict().items()) ** .5


def train_models(root, source, data, labels, config, spec):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from .trading_rl_reward import CadenceRewardEnv
    # Loading checkpoints may reseed libraries. Do this before constructing any training model.
    initial_hashes = {str(s): fingerprint(MaskablePPO.load(source / f'seed_{s}/initial.zip', device='cpu'))
                      for s in spec['seeds']}
    trained = {}
    h = spec['ppo']
    for family in FAMILIES:
        for seed in spec['seeds']:
            name = f'{family}_{seed}'
            dest = root / 'models' / name
            dest.mkdir(parents=True)
            episodes = []
            started = time.perf_counter()
            env = CadenceRewardEnv(data, labels, config['environment'], config['scenarios'][spec['training_scenario']],
                                   order_ttl_seconds=spec['training_order_ttl_seconds'],
                                   overdue_usd_per_hour=spec['rewards'][family])
            model = MaskablePPO('MlpPolicy', env, seed=seed, device='cpu', verbose=0,
                policy_kwargs={'net_arch': h['net_arch']}, **{k: h[k] for k in (
                    'n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef')})
            model.set_logger(configure(str(dest), ['csv']))
            initial_fingerprint = fingerprint(model)
            if initial_fingerprint != initial_hashes[str(seed)]:
                raise ValidationError('fresh initialization differs from paired original seed')
            model.save(dest / 'initial.zip')
            initial = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}

            class Capture(BaseCallback):
                def _on_step(self):
                    for info in self.locals['infos']:
                        if 'episode_summary' in info:
                            episodes.append({'timesteps': self.num_timesteps, **info['episode_summary']})
                    if self.num_timesteps % 8192 == 0:
                        print(f'{name}: {self.num_timesteps}/{h["total_timesteps"]} steps', flush=True)
                    return True

            model.learn(total_timesteps=h['total_timesteps'], callback=Capture())
            delta = parameter_delta(model, initial)
            if model.num_timesteps != h['total_timesteps'] or not math.isfinite(delta) or delta <= 0:
                raise ValidationError('training budget or parameter update failed')
            model.save(dest / 'trained.zip')
            (dest / 'episodes.jsonl').write_text(jsonl(episodes))
            trained[name] = {'family': family, 'seed': seed, 'actual_timesteps': model.num_timesteps,
                'episodes': len(episodes), 'seconds': time.perf_counter() - started,
                'parameters': sum(p.numel() for p in model.policy.parameters()), 'parameter_update_l2': delta,
                'initial_parameters_sha256': initial_fingerprint, 'trained_parameters_sha256': fingerprint(model),
                'original_initial_parameters_sha256': initial_hashes[str(seed)],
                'initial_sha256': sha256_file(dest / 'initial.zip'), 'trained_sha256': sha256_file(dest / 'trained.zip'),
                'optimizer': type(model.policy.optimizer).__name__, 'optimizer_defaults': model.policy.optimizer.defaults,
                'last_optimizer_metrics': {k: float(v) if math.isfinite(float(v)) else None
                    for k, v in model.logger.name_to_value.items() if k.startswith('train/')},
                'last_training_episode': episodes[-1] if episodes else None}
            model.logger.close()
            env.close()
            print(f'{name} frozen in {trained[name]["seconds"]:.1f}s; parameter delta {delta:.6f}', flush=True)
    return trained


def arms(config, spec):
    return (config['fixed_controls'] + [f'ppo_{s}' for s in spec['seeds']]
            + [f'{family}_{s}' for family in FAMILIES for s in spec['seeds']])


def checkpoint_models(root, source, spec):
    from sb3_contrib import MaskablePPO
    tr = strict_json((root / 'training_report.json').read_text())
    expected = {f'{family}_{s}' for family in FAMILIES for s in spec['seeds']}
    if tr['status'] != 'completed' or set(tr['models']) != expected:
        raise ValidationError('all six policies must be frozen before evaluation')
    models = {f'ppo_{s}': MaskablePPO.load(source / f'seed_{s}/trained.zip', device='cpu') for s in spec['seeds']}
    for name in sorted(expected):
        row = tr['models'][name]
        dest = root / 'models' / name
        for stage in ('initial', 'trained'):
            if sha256_file(dest / f'{stage}.zip') != row[f'{stage}_sha256']:
                raise ValidationError('frozen retrained checkpoint changed')
        model = MaskablePPO.load(dest / 'trained.zip', device='cpu')
        initial = MaskablePPO.load(dest / 'initial.zip', device='cpu')
        if (fingerprint(model) != row['trained_parameters_sha256'] or fingerprint(initial) != row['initial_parameters_sha256']
                or fingerprint(initial) != row['original_initial_parameters_sha256']
                or row['actual_timesteps'] != spec['ppo']['total_timesteps']
                or model.num_timesteps != row['actual_timesteps']
                or parameter_delta(model, initial.policy.state_dict()) != row['parameter_update_l2']
                or row['parameter_update_l2'] <= 0):
            raise ValidationError('paired parameter/budget evidence changed')
        models[name] = model
    for seed in spec['seeds']:
        old = MaskablePPO.load(source / f'seed_{seed}/initial.zip', device='cpu')
        if any(tr['models'][f'{f}_{seed}']['initial_parameters_sha256'] != fingerprint(old) for f in FAMILIES):
            raise ValidationError('original paired initialization differs')
    return models


def replay(root, source, data, labels, config, spec, *, reproduce=False):
    from .trading_rl_reward import overdue_seconds
    models = checkpoint_models(root, source, spec)
    result = {}
    for scenario in spec['scenarios']:
        result[scenario] = {}
        for ttl in spec['evaluation_order_ttl_seconds']:
            key = str(ttl)
            result[scenario][key] = {}
            dest = root / scenario / key
            if not reproduce:
                dest.mkdir(parents=True)
            for arm in arms(config, spec):
                value = evaluate_lifetime(data, labels, config, config['scenarios'][scenario], ttl,
                    model=models.get(arm), weekly_only=arm == 'weekly_cost_only')
                late = overdue_seconds(data.start, data.end, data.start, value['ledger'], config['environment']['cadence_seconds'])
                result[scenario][key][arm] = {**{k: value[k] for k in ('metrics', 'lifecycle')}, 'overdue_hours': str(late / 3600)}
                for field in ('ledger', 'decisions', 'equity_curve', 'pending'):
                    path = dest / f'{arm}.{field}.jsonl'
                    if reproduce:
                        if rows(path) != value[field]:
                            raise ValidationError('retrained policy replay differs: ' + str(path))
                    else:
                        path.write_text(jsonl(value[field]))
            print(f'{"Reproduced" if reproduce else "Evaluated"} {scenario}/{ttl}s: 11 policies', flush=True)
    return result


def aggregates(result, spec):
    summary = {}
    for scenario, lifetimes in result.items():
        summary[scenario] = {}
        for ttl, values in lifetimes.items():
            families = {}
            for family in ('ppo', *FAMILIES):
                selected = [values[f'{family}_{s}'] for s in spec['seeds']]
                cash = [float(v['metrics']['final_cash']) for v in selected]
                families[family] = {'mean_final_cash': statistics.fmean(cash), 'min_final_cash': min(cash),
                    'max_final_cash': max(cash), 'std_final_cash': statistics.pstdev(cash),
                    'cadence_passes': sum(v['metrics']['policy_requirements_met'] for v in selected),
                    'mean_entry_count': statistics.fmean(v['metrics']['entry_count'] for v in selected),
                    'mean_overdue_hours': statistics.fmean(float(v['overdue_hours']) for v in selected),
                    'paired_cash_difference_vs_original': {str(s): float(values[f'{family}_{s}']['metrics']['final_cash'])
                        - float(values[f'ppo_{s}']['metrics']['final_cash']) for s in spec['seeds']}}
            summary[scenario][ttl] = families
    return summary


def run(dataset, store, source, execution, config_path, output):
    if output.exists():
        raise ValidationError('retraining output already exists')
    spec, config = load_spec(source, execution, config_path)
    resources = runtime(config)
    started = time.perf_counter()
    data = prepare(dataset, store, 'train', config)
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / 'config.json')
    shutil.copytree(Path(__file__).parent, output / 'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    frozen = freeze_data(output, data)
    for name in ('train.catalog.jsonl', 'train.features.jsonl'):
        if sha256_file(output / name) != sha256_file(source / name):
            raise ValidationError('retraining features differ from original experiment')
    plan = {'spec': spec, 'original_config': config, 'resources': resources, 'training_data': frozen,
            'paths': {k: str(v.resolve()) for k, v in dict(dataset=dataset, store=store, source=source, execution=execution).items()},
            'artifact_hashes': artifacts(output)}
    (output / 'plan.json').write_text(json_text(plan))
    labels = settlements(dataset, data)
    trained = train_models(output, source, data, labels, config, spec)
    tr = {'status': 'completed', 'plan_sha256': sha256_file(output / 'plan.json'), 'models': trained,
          'validation_used_for_gradients': False, 'final_test_opened': False, 'artifact_hashes': artifacts(output)}
    (output / 'training_report.json').write_text(json_text(tr))
    # This marker is written only after all checkpoints, optimizers and training traces are frozen.
    (output / 'evaluation_plan.json').write_text(json_text({'training_report_sha256': sha256_file(output / 'training_report.json'),
        'partition': 'train', 'arms': arms(config, spec), 'scenarios': spec['scenarios'],
        'order_ttl_seconds': spec['evaluation_order_ttl_seconds']}))
    result = replay(output, source, data, labels, config, spec)
    report = {'status': 'completed', 'plan_sha256': sha256_file(output / 'plan.json'), 'partition': 'train',
        'training_report_sha256': sha256_file(output / 'training_report.json'),
        'evaluation_plan_sha256': sha256_file(output / 'evaluation_plan.json'), 'training_data': frozen,
        'scenarios': result, 'aggregates': aggregates(result, spec), 'elapsed_seconds': time.perf_counter() - started,
        'training_performed': 'six_small_masked_ppo_allocation_networks', 'validation_replayed': False, 'final_test_opened': False,
        'llm_calls': 0, 'foundation_model_updated': False, 'real_orders_sent': 0, 'default_promotion': False,
        'limitations': ['In-sample adaptive engineering experiment on six training event groups; not evidence of generalization.',
            'Repeated hours and episodes are not independent events. Earlier validation was already inspected; none is replayed here.',
            'Historical prints and hypothetical full fills/costs do not reconstruct executable order-book liquidity.',
            'Artificial scope and calendar countdowns are retained for comparison, not verified public expiry features.',
            'A one-hour training TTL equals the decision interval. Four-hour transfer can retain unobserved pending orders.',
            'Late-time penalty is a learning cost, never a cash debit or replacement for the seven-day actual-fill requirement.',
            'No LLM/LoRA parameters, agent routing, forecast probabilities or reward for forecast quality are trained.',
            'Calendar-end liquidation and later settlement remain in raw net equity; report settlement counts separately.'],
        'artifact_hashes': artifacts(output)}
    (output / 'report.json').write_text(json_text(report))
    return report


def audit(root):
    report = strict_json((root / 'report.json').read_text())
    for name in ('plan', 'training_report', 'evaluation_plan'):
        if sha256_file(root / f'{name}.json') != report[f'{name}_sha256']:
            raise ValidationError('retraining freeze changed: ' + name)
    for name, digest in report['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root / name) != digest:
            raise ValidationError('retraining artifact changed: ' + name)
    for name in SOURCE_FILES:
        if sha256_file(Path(__file__).with_name(name)) != sha256_file(root / 'source_snapshot/foretellmesh' / name):
            raise ValidationError('use frozen retraining source for audit')
    plan = strict_json((root / 'plan.json').read_text())
    tr = strict_json((root / 'training_report.json').read_text())
    ep = strict_json((root / 'evaluation_plan.json').read_text())
    paths = {k: Path(v) for k, v in plan['paths'].items()}
    spec, config = load_spec(paths['source'], paths['execution'], root / 'config.json')
    if (spec != plan['spec'] or config != plan['original_config'] or tr['plan_sha256'] != report['plan_sha256']
            or ep != {'training_report_sha256': report['training_report_sha256'], 'partition': 'train',
                'arms': arms(config, spec), 'scenarios': spec['scenarios'], 'order_ttl_seconds': spec['evaluation_order_ttl_seconds']}):
        raise ValidationError('retraining protocol changed')
    runtime(config)
    data = prepare(paths['dataset'], paths['store'], 'train', config)
    if (data.bindings != plan['training_data']['bindings'] or data.catalog != rows(root / 'train.catalog.jsonl')
            or data.feature_rows() != rows(root / 'train.features.jsonl')):
        raise ValidationError('retraining data changed')
    result = replay(root, paths['source'], data, settlements(paths['dataset'], data), config, spec, reproduce=True)
    if result != report['scenarios'] or aggregates(result, spec) != report['aggregates']:
        raise ValidationError('retraining results differ')
    return {'status': 'passed', 'report_sha256': sha256_file(root / 'report.json'), 'partition': 'train',
            'arms_reproduced': sum(len(a) for s in result.values() for a in s.values()),
            'paired_initializations_verified': True, 'parameter_updates_verified': True,
            'validation_replayed': False, 'final_test_opened': False, 'real_orders_sent': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    start = sub.add_parser('run')
    for name in ('dataset', 'store', 'source', 'execution', 'config', 'output'):
        start.add_argument('--' + name, type=Path, required=True)
    sub.add_parser('audit').add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        result = run(args.dataset, args.store, args.source, args.execution, args.config, args.output)
        print(json_text({'status': result['status'], 'seconds': result['elapsed_seconds']}))
    else:
        result = audit(args.run)
        (args.run / 'audit.json').write_text(json_text(result))
        print(json_text(result))


if __name__ == '__main__':
    main()
