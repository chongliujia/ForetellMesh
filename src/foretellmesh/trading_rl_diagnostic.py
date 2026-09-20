"""Training-only, read-only diagnostics of frozen allocation policies.

Retrospective fill probes are never policy inputs. They describe one-step
references on a realized account path, not existence of a feasible strategy.
"""
from collections import Counter
from datetime import timedelta
from decimal import Decimal as D
from pathlib import Path
import argparse
import shutil
import time

from .data import sha256_file, strict_json
from .evaluation import json_text
from .market_development import rows
from .schema import ValidationError, iso, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .trading_rl import artifacts, freeze_data, load_config, runtime
from .trading_rl_data import prepare, settlements


OPS = ('yes_1', 'yes_2', 'no_1', 'no_2', 'sell_half', 'sell_all')
CORE_SOURCE = ('trading_rl_env.py', 'trading_rl_data.py', 'trading_rl.py',
               'mean_reversion.py', 'paper_trading.py', 'schema.py')


def buy_blockers(env, mid: str, state: dict, side: str, budget: D) -> list[str]:
    """Explain existing gates without modifying the executor; verify equivalence."""
    reasons = []
    if mid in env.positions:
        reasons.append('already_held')
    if mid in env.resolved:
        reasons.append('resolved')
    if len(env.positions) >= env.p['max_positions']:
        reasons.append('position_limit')
    if state.get('status') != 'ready':
        reasons.append(state.get('status', 'outside_admitted_scope'))
    if env.time < env.cooldowns.get(mid, env.data.start):
        reasons.append('cooldown')
    if env.time + timedelta(seconds=env.p['weekly_holding_seconds']) > min(env.windows[mid].retire_at, env.data.end):
        reasons.append('less_than_24h_entry_scope')
    if not env.due and env.attempts[env.time.date()] >= env.p['max_entries_per_day']:
        reasons.append('daily_entry_attempt_limit')
    if state.get('status') == 'ready':
        q = D(str(state['yes_price']))
        if env.due:
            if budget != 1:
                reasons.append('weekly_budget_must_be_1')
            if side != ('yes' if q >= D('.5') else 'no'):
                reasons.append('weekly_higher_price_side_only')
        elif not D(env.p['min_yes_price']) <= q <= D(env.p['max_yes_price']):
            reasons.append('normal_entry_price_band')
        if (env._side_price(q, side) + env.premium) * (1 + env.fee) >= 1:
            reasons.append('all_in_unit_cost_ge_1')
    if env.cash < budget:
        reasons.append('cash_limit')
    if D(env.p['max_event_usd']) - env._exposure(env.windows[mid].event_group_id) < budget:
        reasons.append('event_exposure_limit')
    if D(env.p['max_portfolio_usd']) - env._exposure() < budget:
        reasons.append('portfolio_exposure_limit')
    if (not reasons) != env._can_buy(mid, state, side, budget):
        raise ValidationError('diagnostic buy gates differ from executor')
    return reasons


def reference_probe(env, mid: str, op: str) -> dict:
    """Future-aware audit only: first subsequent print, with original price gates."""
    deadline = env.time + timedelta(seconds=env.p['fill_window_seconds'])
    quote = env.data.feed.next(mid, env.time, deadline)
    result = {'market_id': mid, 'op': op, 'status': 'no_post_decision_print'}
    if quote is None:
        later = env.data.feed.next(mid, env.time, env.data.end + timedelta(seconds=env.p['fill_window_seconds']))
        result['next_recorded_print_delay_seconds'] = None if later is None else (later[1] - env.time).total_seconds()
        return result
    q, at = quote
    if not env.time < at <= deadline:
        raise ValidationError('invalid diagnostic execution reference')
    result.update(reference_time=iso(at), reference_yes_price=str(q),
                  reference_delay_seconds=(at - env.time).total_seconds())
    if env.labels[mid].time <= at:
        result['status'] = 'settled_before_reference'
    elif op.startswith(('yes_', 'no_')):
        side = op.split('_')[0]
        p = env._side_price(q, side)
        reference = env._side_price(env._quote(mid, env.time)[0], side)
        result['status'] = ('price_limit' if (p + env.premium) * (1 + env.fee) >= 1
                            or p > reference + D(env.p['entry_price_tolerance']) else 'reference_acceptable')
    else:
        result['status'] = ('reference_acceptable' if env._side_price(q, env.positions[mid]['side']) - env.premium > 0
                            else 'nonpositive_synthetic_bid')
    return result


def decision_diagnostic(env, action: int) -> dict:
    """Call after choosing the action. Output is an audit artifact, never an obs."""
    if env.done or env.pending:
        raise ValidationError('diagnostic requires a live decision with no pending orders')
    obs_hash = canonical_hash(env.obs.tolist())
    mask = env.action_masks().tolist()
    if not 0 <= action < len(mask) or not mask[action]:
        raise ValidationError('diagnostic policy selected an invalid action')
    state = env.data.states[env.i]
    markets = []
    offers = []
    blocked = Counter()
    selected = None if action == 0 else (env.slots[(action - 1) // 6], OPS[(action - 1) % 6])
    for mid in sorted(set(state) | set(env.positions)):
        s = state.get(mid, {})
        quote = env._quote(mid, env.time)
        gates = {}
        for op in OPS[:4]:
            side, amount = op.split('_')
            reasons = buy_blockers(env, mid, s, side, D(amount))
            gates[op] = reasons
            blocked.update(reasons)
            if not reasons:
                offers.append({'market_id': mid, 'op': op, 'visible': mid in env.slots, 'automatic': False})
        markets.append({'market_id': mid, 'in_slots': mid in env.slots, 'state_status': s.get('status', 'outside_admitted_scope'),
                        'quote_age_seconds': None if quote is None else (env.time - quote[1]).total_seconds(),
                        'hours_to_scope_end': (env.windows[mid].retire_at - env.time).total_seconds() / 3600,
                        'group_exposure_usd': str(env._exposure(env.windows[mid].event_group_id)), 'buy_blockers': gates})
    for j, mid in enumerate(env.slots):
        for offset in (4, 5):
            if mask[1 + j * 6 + offset]:
                offers.append({'market_id': mid, 'op': OPS[offset], 'visible': True, 'automatic': False})
    visible = [o for o in offers if o['visible']]
    expected = {(env.slots[(a - 1) // 6], OPS[(a - 1) % 6]) for a, ok in enumerate(mask) if a and ok}
    if {(o['market_id'], o['op']) for o in visible} != expected:
        raise ValidationError('diagnostic visible actions differ from mask')
    for mid, reason in sorted(env.auto.items()):
        if reason:
            offers.append({'market_id': mid, 'op': 'sell_all', 'visible': False, 'automatic': True, 'reason': reason})
    causal = {'time': iso(env.time), 'next_tick': iso(env.data.ticks[env.i + 1]), 'action': action,
              'observation_sha256': obs_hash, 'last_fill_or_start': iso(env.last_fill),
              'seconds_until_deadline': env.p['cadence_seconds'] - (env.time - env.last_fill).total_seconds(),
              'weekly_due': env.due, 'cash_usd': str(env.cash), 'exposure_usd': str(env._exposure()),
              'position_count': len(env.positions), 'active_markets': len(state), 'slots': list(env.slots),
              'outside_scope_markets': len(env.windows) - len(state), 'visible_legal_actions': len(visible),
              'all_legal_actions': len(offers) - sum(o['automatic'] for o in offers),
              'automatic_exit_reasons': {m: r for m, r in env.auto.items() if r},
              'buy_gate_counts': dict(sorted(blocked.items())), 'markets': markets}
    # Everything below is future-aware; no callback can consume this output.
    probes = [{**o, **reference_probe(env, o['market_id'], o['op'])} for o in offers]
    accepted = [o for o in probes if o['status'] == 'reference_acceptable']
    chosen = [o for o in accepted if o['automatic'] or (o['market_id'], o['op']) == selected]
    if chosen:
        category = 'selected_or_automatic_reference_available'
    elif any(o['visible'] for o in accepted):
        category = 'visible_alternative_reference_available'
    elif accepted:
        category = 'truncated_alternative_reference_available'
    elif offers:
        category = 'legal_actions_but_no_acceptable_reference'
    else:
        category = 'no_legal_action'
    if canonical_hash(env.obs.tolist()) != obs_hash or env.action_masks().tolist() != mask:
        raise ValidationError('diagnostic mutated policy inputs')
    return {'causal': causal, 'retrospective': {'category': category, 'offers': probes,
            'all_acceptable_references': len(accepted),
            'visible_acceptable_references': sum(o['visible'] for o in accepted),
            'automatic_acceptable_references': sum(o['automatic'] for o in accepted)}}


def summarize_steps(steps: list[dict]) -> dict:
    categories = Counter()
    gates = Counter()
    references = Counter()
    for row in steps:
        categories[row['retrospective']['category']] += 1
        gates.update(row['causal']['buy_gate_counts'])
        references.update(o['status'] for o in row['retrospective']['offers'])
    return {'decision_count': len(steps), 'categories': dict(sorted(categories.items())),
            'overlapping_buy_gate_action_counts': dict(sorted(gates.items())),
            'hypothetical_reference_action_counts': dict(sorted(references.items()))}


def violation_details(steps: list[dict], cadence: dict, lead_seconds: int) -> list[dict]:
    out = []
    for violation in cadence['violations']:
        a = timestamp(violation['from'], 'gap start')
        b = timestamp(violation['to'], 'gap end')
        deadline = timestamp(violation['overdue_from'], 'deadline')
        during = [r for r in steps if a <= timestamp(r['causal']['time'], 'decision') < b]
        warning = [r for r in during if deadline - timedelta(seconds=lead_seconds) <= timestamp(r['causal']['time'], 'decision') <= deadline]
        accepted = [o for r in warning for o in r['retrospective']['offers']
                    if o['status'] == 'reference_acceptable' and timestamp(o['reference_time'], 'reference') <= deadline]
        out.append({**violation, 'overdue_hours': max(0., (b - deadline).total_seconds() / 3600),
                    'entire_gap': summarize_steps(during), 'warning_period': summarize_steps(warning),
                    'warning_reference_actions_arriving_by_deadline': len(accepted),
                    'warning_visible_reference_actions_arriving_by_deadline': sum(o['visible'] for o in accepted),
                    'warning_automatic_reference_actions_arriving_by_deadline': sum(o['automatic'] for o in accepted)})
    return out


def diagnose(data, labels, config: dict, *, model=None, weekly_only: bool = False) -> dict:
    from .trading_rl_env import AllocationEnv
    if data.partition != 'train':
        raise ValidationError('cadence diagnosis is training-only')
    env = AllocationEnv(data, labels, config['environment'], config['scenarios']['cost_assumption'], record=True)
    obs, _ = env.reset()
    steps = []
    while not env.done:
        action = env.fixed_action(weekly_only=weekly_only) if model is None else int(
            model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0])
        row = decision_diagnostic(env, action)
        offset = len(env.ledger)
        obs, _, _, _, _ = env.step(action)
        events = env.ledger[offset:]
        row['actual_transition'] = {
            'orders': sum(r['kind'] in ('buy_order', 'sell_order') for r in events),
            'fills_in_calendar': sum(r['kind'] in ('buy_fill', 'sell_fill') and timestamp(r['time'], 'fill') <= data.end for r in events),
            'cancellations': dict(sorted(Counter(r['reason'] for r in events if r['kind'] == 'cancel').items()))}
        steps.append(row)
    metrics = env.summary()
    violations = violation_details(steps, metrics['cadence'], config['environment']['cadence_lead_seconds'])
    report = {'metrics': metrics, 'all_decisions': summarize_steps(steps),
              'due_decisions': summarize_steps([r for r in steps if r['causal']['weekly_due']]),
              'violations': violations, 'total_overdue_hours': sum(v['overdue_hours'] for v in violations)}
    result = {'summary': report, 'steps': steps, 'ledger': env.ledger, 'decisions': env.decisions}
    env.close()
    return result


def reference_config(source: Path, spec: dict) -> dict:
    fixed = {'schema_version': '1', 'partition': 'train', 'scenario': 'cost_assumption',
             'training_performed': False, 'final_test_opened': False, 'real_orders_sent': 0,
             'arms': ['corrected_mean_reversion', 'weekly_cost_only', 'ppo_7', 'ppo_17', 'ppo_27']}
    if set(spec) != set(fixed) | {'run_name', 'reference_hashes'} or any(type(spec[k]) is not type(v) or spec[k] != v for k, v in fixed.items()):
        raise ValidationError('unsupported diagnostic specification')
    if set(spec['reference_hashes']) != {'config.json', 'training_report.json', 'report.json'}:
        raise ValidationError('missing diagnostic source bindings')
    for name, digest in spec['reference_hashes'].items():
        if sha256_file(source / name) != digest:
            raise ValidationError('diagnostic reference changed: ' + name)
    trained = strict_json((source / 'training_report.json').read_text())
    for name in CORE_SOURCE:
        key = 'source_snapshot/foretellmesh/' + name
        expected = trained['artifact_hashes'][key]
        if sha256_file(source / key) != expected or sha256_file(Path(__file__).with_name(name)) != expected:
            raise ValidationError('diagnostic executor differs from frozen experiment: ' + name)
    config = load_config(source / 'config.json')
    if config['seeds'] != [7, 17, 27]:
        raise ValidationError('diagnostic source seeds differ')
    for seed in config['seeds']:
        if sha256_file(source / f'seed_{seed}/trained.zip') != trained['seeds'][str(seed)]['trained_sha256']:
            raise ValidationError('diagnostic checkpoint changed')
    return config


def replay_arms(data, labels, config: dict, source: Path, destination: Path, *, reproduce: bool = False) -> dict:
    from sb3_contrib import MaskablePPO
    results = {}
    for arm in config['fixed_controls'] + [f'ppo_{s}' for s in config['seeds']]:
        model = MaskablePPO.load(source / f'seed_{arm[4:]}/trained.zip', device='cpu') if arm.startswith('ppo_') else None
        result = diagnose(data, labels, config, model=model, weekly_only=arm == 'weekly_cost_only')
        results[arm] = result['summary']
        for name in ('steps', 'ledger', 'decisions'):
            path = destination / f'{arm}.{name}.jsonl'
            if reproduce:
                if rows(path) != result[name]:
                    raise ValidationError('diagnostic replay differs: ' + str(path))
            else:
                path.write_text(jsonl(result[name]))
        print(f'{"Reproduced" if reproduce else "Diagnosed"} training {arm}: '
              f'{len(result["summary"]["violations"])} gaps, {result["summary"]["total_overdue_hours"]:.2f} overdue hours', flush=True)
    return results


def run(dataset: Path, store: Path, source: Path, spec_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError('diagnostic output already exists')
    spec = strict_json(spec_path.read_text())
    config = reference_config(source, spec)
    resources = runtime(config)
    started = time.perf_counter()
    data = prepare(dataset, store, 'train', config)
    output.mkdir(parents=True)
    shutil.copyfile(spec_path, output / 'config.json')
    shutil.copytree(Path(__file__).parent, output / 'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    frozen = freeze_data(output, data)
    for name in ('train.catalog.jsonl', 'train.features.jsonl'):
        if sha256_file(output / name) != sha256_file(source / name):
            raise ValidationError('diagnostic training features differ from original run')
    plan = {'spec': spec, 'executor_config': config, 'resources': resources, 'training_data': frozen,
            'paths': {'dataset': str(dataset.resolve()), 'store': str(store.resolve()), 'source': str(source.resolve())},
            'artifact_hashes': artifacts(output)}
    (output / 'plan.json').write_text(json_text(plan))
    results = replay_arms(data, settlements(dataset, data), config, source, output)
    report = {'status': 'completed', 'plan_sha256': sha256_file(output / 'plan.json'), 'partition': 'train',
              'training_data': frozen, 'arms': results, 'elapsed_seconds': time.perf_counter() - started,
              'training_performed': False, 'validation_replayed': False, 'final_test_opened': False,
              'real_orders_sent': 0, 'foundation_model_updated': False, 'llm_calls': 0, 'default_promotion': False,
              'scope_provenance': {'initialized_at': 'historical_onchain_initialization',
                  'retire_at': 'last_admitted_observation_not_public_contract_expiry',
                  'calendar': 'first_to_last_admitted_training_observation',
                  'scope_features_are_retrospective_experiment_boundaries': True},
              'limitations': ['Training replay is in-sample, not a new profitability evaluation.',
                  'Future references are post-decision audit artifacts, not deployable candidate-selection inputs.',
                  'Action-level gate counts overlap; no single causal attribution or globally infeasible strategy is inferred.',
                  'Acceptable first prints assume full fills; historical bid/ask, depth, queue and minimum lot remain unproven.',
                  'A missing 300-second reference does not establish absence of market liquidity.',
                  'Frozen v1 policies see artificial replay scope countdowns, not verified public contract expiry features.'],
              'artifact_hashes': artifacts(output)}
    (output / 'report.json').write_text(json_text(report))
    return report


def audit(root: Path) -> dict:
    report = strict_json((root / 'report.json').read_text())
    if sha256_file(root / 'plan.json') != report['plan_sha256']:
        raise ValidationError('diagnostic plan changed')
    for name, digest in report['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root / name) != digest:
            raise ValidationError('diagnostic artifact changed: ' + name)
    plan = strict_json((root / 'plan.json').read_text())
    if strict_json((root / 'config.json').read_text()) != plan['spec']:
        raise ValidationError('diagnostic config changed')
    if sha256_file(Path(__file__)) != sha256_file(root / 'source_snapshot/foretellmesh/trading_rl_diagnostic.py'):
        raise ValidationError('use frozen diagnostic source for audit')
    paths = {k: Path(v) for k, v in plan['paths'].items()}
    config = reference_config(paths['source'], plan['spec'])
    if config != plan['executor_config']:
        raise ValidationError('diagnostic executor config changed')
    runtime(config)
    data = prepare(paths['dataset'], paths['store'], 'train', config)
    if (data.bindings != plan['training_data']['bindings'] or data.catalog != rows(root / 'train.catalog.jsonl')
            or data.feature_rows() != rows(root / 'train.features.jsonl')):
        raise ValidationError('diagnostic training data changed')
    results = replay_arms(data, settlements(paths['dataset'], data), config, paths['source'], root, reproduce=True)
    if results != report['arms']:
        raise ValidationError('diagnostic summary differs')
    return {'status': 'passed', 'report_sha256': sha256_file(root / 'report.json'),
            'arms_reproduced': len(results), 'partition': 'train', 'final_test_opened': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    for option in ('dataset', 'store', 'source', 'config', 'output'):
        run_parser.add_argument('--' + option, type=Path, required=True)
    sub.add_parser('audit').add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        report = run(args.dataset, args.store, args.source, args.config, args.output)
        print(json_text({'status': report['status'], 'arms': len(report['arms']), 'seconds': report['elapsed_seconds']}))
    else:
        result = audit(args.run)
        (args.run / 'audit.json').write_text(json_text(result))
        print(json_text(result))


if __name__ == '__main__':
    main()
