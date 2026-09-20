"""Freeze and replay an order-lifetime sensitivity study on training data only."""
from collections import Counter
from datetime import timedelta
from pathlib import Path
import argparse
import shutil
import time

from .data import sha256_file, strict_json
from .evaluation import json_text
from .market_development import rows
from .schema import ValidationError, iso
from .sft_data import jsonl
from .trading_rl import artifacts, evaluate_one, freeze_data, runtime
from .trading_rl_data import prepare, settlements
from .trading_rl_diagnostic import reference_config


def load_spec(source: Path, path: Path):
    spec = strict_json(path.read_text())
    fixed = {'schema_version': '1', 'partition': 'train', 'training_performed': False,
             'final_test_opened': False, 'real_orders_sent': 0, 'order_ttl_seconds': [300, 3600, 14400],
             'closeout_ttl_seconds': 300, 'scenarios': ['frictionless_reference', 'cost_assumption', 'cost_stress'],
             'arms': ['corrected_mean_reversion', 'weekly_cost_only', 'ppo_7', 'ppo_17', 'ppo_27'],
             'primary_seed': 7, 'winner_selection': 'none_all_predeclared_arms_reported',
             'executor_version': 'pending_first_print_proxy_v1'}
    if set(spec) != set(fixed) | {'run_name', 'reference_hashes'} or any(
            type(spec[k]) is not type(v) or spec[k] != v for k, v in fixed.items()):
        raise ValidationError('unsupported execution sensitivity specification')
    diagnostic_spec = {k: spec[k] for k in ('schema_version', 'partition', 'training_performed', 'final_test_opened',
                                          'real_orders_sent', 'arms', 'run_name', 'reference_hashes')}
    diagnostic_spec['scenario'] = 'cost_assumption'
    return spec, reference_config(source, diagnostic_spec)


def evaluate_lifetime(data, labels, config, costs, ttl, *, model=None, weekly_only=False):
    from .trading_rl_execution_env import PendingAllocationEnv
    if data.partition != 'train':
        raise ValidationError('execution sensitivity is training-only')
    env = PendingAllocationEnv(data, labels, config['environment'], costs, order_ttl_seconds=ttl, record=True)
    obs, _ = env.reset()
    pending = []
    while not env.done:
        pending.append({'time': iso(env.time), 'orders': env.pending_snapshot()})
        action = env.fixed_action(weekly_only=weekly_only) if model is None else int(
            model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0])
        obs, _, _, _, _ = env.step(action)
    summary = env.summary()
    lifecycle = env.lifecycle_summary()
    lifecycle['decisions_with_pending_orders'] = sum(bool(r['orders']) for r in pending)
    lifecycle['order_count'] = sum(r['kind'] in ('buy_order', 'sell_order') for r in env.ledger)
    lifecycle['cancellation_count'] = sum(r['kind'] == 'cancel' for r in env.ledger)
    lifecycle['orders_by_reason'] = dict(sorted(Counter(r['reason'] for r in env.ledger if r['kind'] in ('buy_order', 'sell_order')).items()))
    result = {'metrics': summary, 'lifecycle': lifecycle, 'ledger': env.ledger, 'decisions': env.decisions,
              'equity_curve': env.curve, 'pending': pending}
    env.close()
    return result


def original_ledger(ledger):
    # The 300-second implementation must preserve all original economic events.
    return [{k: v for k, v in row.items() if k not in ('submitted_at', 'expires_at')} for row in ledger]


def coverage_summary(data, spec):
    """Price coverage audit, independent of actions, holdings and resolution labels."""
    windows = {w.market_id: w for w in data.windows}
    counts = Counter()
    reference_ticks = Counter()
    for t, state in zip(data.ticks[:-1], data.states[:-1]):
        ready = [mid for mid, s in state.items() if s['status'] == 'ready']
        counts['no_active_market' if not state else 'active_without_ready_quote' if not ready else 'at_least_one_ready_quote'] += 1
        for ttl in spec['order_ttl_seconds']:
            if any(data.feed.next(mid, t, min(t + timedelta(seconds=ttl), data.end, windows[mid].retire_at))
                   for mid in ready):
                reference_ticks[str(ttl)] += 1
    return {'decision_steps': len(data.ticks) - 1, 'quote_status_counts': dict(sorted(counts.items())),
            'retrospective_raw_reference_ticks_by_ttl': {str(ttl): reference_ticks[str(ttl)] for ttl in spec['order_ttl_seconds']},
            'reference_definition': 'at least one ready market with a subsequent print before TTL and scope end; no price, cash, position or holding-buffer filters',
            'policy_input': False, 'executable_liquidity_proven': False}


def replay(data, labels, config, spec, source: Path, destination: Path, *, reproduce=False):
    from sb3_contrib import MaskablePPO
    models = {f'ppo_{s}': MaskablePPO.load(source / f'seed_{s}/trained.zip', device='cpu') for s in config['seeds']}
    results = {}
    for scenario in spec['scenarios']:
        results[scenario] = {}
        for ttl in spec['order_ttl_seconds']:
            key = str(ttl)
            results[scenario][key] = {}
            dest = destination / scenario / key
            if not reproduce:
                dest.mkdir(parents=True)
            for arm in spec['arms']:
                model = models.get(arm)
                value = evaluate_lifetime(data, labels, config, config['scenarios'][scenario], ttl,
                                          model=model, weekly_only=arm == 'weekly_cost_only')
                if ttl == 300:
                    baseline = evaluate_one(data, labels, config, config['scenarios'][scenario], model=model,
                                            weekly_only=arm == 'weekly_cost_only')
                    if (value['metrics'] != baseline['metrics'] or value['decisions'] != baseline['decisions']
                            or original_ledger(value['ledger']) != baseline['ledger']
                            or value['equity_curve'] != baseline['equity_curve']):
                        raise ValidationError('300-second lifecycle differs from original executor: ' + arm)
                results[scenario][key][arm] = {k: value[k] for k in ('metrics', 'lifecycle')}
                for field in ('ledger', 'decisions', 'equity_curve', 'pending'):
                    path = dest / f'{arm}.{field}.jsonl'
                    if reproduce:
                        if rows(path) != value[field]:
                            raise ValidationError('execution replay differs: ' + str(path))
                    else:
                        path.write_text(jsonl(value[field]))
            print(f'{"Reproduced" if reproduce else "Evaluated"} {scenario} / {ttl}s: 5 frozen policies', flush=True)
    return results


def run(dataset: Path, store: Path, source: Path, config_path: Path, output: Path):
    if output.exists():
        raise ValidationError('execution experiment output already exists')
    spec, config = load_spec(source, config_path)
    resources = runtime(config)
    started = time.perf_counter()
    data = prepare(dataset, store, 'train', config)
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / 'config.json')
    shutil.copytree(Path(__file__).parent, output / 'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    frozen = freeze_data(output, data)
    for name in ('train.catalog.jsonl', 'train.features.jsonl'):
        if sha256_file(output / name) != sha256_file(source / name):
            raise ValidationError('training features differ from original experiment')
    plan = {'spec': spec, 'original_config': config, 'resources': resources, 'training_data': frozen,
            'paths': {'dataset': str(dataset.resolve()), 'store': str(store.resolve()), 'source': str(source.resolve())},
            'artifact_hashes': artifacts(output)}
    (output / 'plan.json').write_text(json_text(plan))
    results = replay(data, settlements(dataset, data), config, spec, source, output)
    report = {'status': 'completed', 'plan_sha256': sha256_file(output / 'plan.json'), 'partition': 'train',
              'training_data': frozen, 'coverage': coverage_summary(data, spec),
              'scenarios': results, 'elapsed_seconds': time.perf_counter() - started,
              'training_performed': False, 'validation_replayed': False, 'final_test_opened': False,
              'llm_calls': 0, 'foundation_model_updated': False, 'real_orders_sent': 0, 'default_promotion': False,
              'original_300s_replays_matched': len(spec['scenarios']) * len(spec['arms']),
              'limitations': ['In-sample execution sensitivity, not an out-of-sample profitability evaluation.',
                  'First-print proxy with hypothetical full fills and costs; not a historical exchange order-book simulator.',
                  'The first adverse print cancels; later favorable prices within the lifetime are not searched.',
                  'All lifetimes share a fixed 300-second terminal closeout and the original calendar and scope features.',
                  'Artificial last-observation countdowns are not verified public contract expiries.',
                  'Frozen policies retain 88 inputs and 25 actions; explicit pending-order features and policy cancellations are not trained.',
                  'Changes combine longer execution waits with necessary reservation and cancellation mechanics, not new RL skill.',
                  'All predeclared lifetimes and cost scenarios are reported without selecting a winner.'],
              'artifact_hashes': artifacts(output)}
    (output / 'report.json').write_text(json_text(report))
    return report


def audit(root: Path):
    report = strict_json((root / 'report.json').read_text())
    if sha256_file(root / 'plan.json') != report['plan_sha256']:
        raise ValidationError('execution plan changed')
    for name, digest in report['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root / name) != digest:
            raise ValidationError('execution artifact changed: ' + name)
    plan = strict_json((root / 'plan.json').read_text())
    for name in ('trading_rl_execution.py', 'trading_rl_execution_env.py'):
        if sha256_file(Path(__file__).with_name(name)) != sha256_file(root / 'source_snapshot/foretellmesh' / name):
            raise ValidationError('use frozen execution source for audit')
    paths = {k: Path(v) for k, v in plan['paths'].items()}
    spec, config = load_spec(paths['source'], root / 'config.json')
    if spec != plan['spec'] or config != plan['original_config']:
        raise ValidationError('execution configuration changed')
    runtime(config)
    data = prepare(paths['dataset'], paths['store'], 'train', config)
    if (data.bindings != plan['training_data']['bindings'] or data.catalog != rows(root / 'train.catalog.jsonl')
            or data.feature_rows() != rows(root / 'train.features.jsonl')):
        raise ValidationError('execution training data changed')
    results = replay(data, settlements(paths['dataset'], data), config, spec, paths['source'], root, reproduce=True)
    if results != report['scenarios']:
        raise ValidationError('execution summary differs')
    if coverage_summary(data, spec) != report['coverage']:
        raise ValidationError('execution price coverage differs')
    return {'status': 'passed', 'report_sha256': sha256_file(root / 'report.json'), 'partition': 'train',
            'arms_reproduced': sum(len(arms) for scenarios in results.values() for arms in scenarios.values()),
            'original_300s_replays_matched': len(spec['scenarios']) * len(spec['arms']), 'final_test_opened': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    for option in ('dataset', 'store', 'source', 'config', 'output'):
        run_parser.add_argument('--' + option, type=Path, required=True)
    sub.add_parser('audit').add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        result = run(args.dataset, args.store, args.source, args.config, args.output)
        print(json_text({'status': result['status'], 'seconds': result['elapsed_seconds']}))
    else:
        result = audit(args.run)
        (args.run / 'audit.json').write_text(json_text(result))
        print(json_text(result))


if __name__ == '__main__':
    main()
