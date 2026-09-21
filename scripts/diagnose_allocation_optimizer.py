"""Read frozen training artifacts; diagnose decisions without training or opening holdouts."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal as D
from pathlib import Path
import statistics

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.schema import ValidationError, timestamp


def decision_summary(decisions):
    counts = Counter(); uniform_wait = []; sizes = Counter()
    for row in decisions:
        mask = row['mask']; action = row['executed_action']
        if (len(mask) != 25 or any(type(x) is not bool for x in mask) or not mask[0]
                or type(action) is not int or not 0 <= action < len(mask) or not mask[action]):
            raise ValidationError('invalid action/mask in frozen decision')
        legal = sum(mask); sizes[legal] += 1; counts['ticks'] += 1
        counts['wait_ticks'] += action == 0
        if legal == 1:
            counts['only_wait_legal_ticks'] += 1
        else:
            counts['choice_ticks'] += 1
            counts['voluntary_wait_ticks'] += action == 0
            uniform_wait.append(1 / legal)
        counts['buy_choice_ticks'] += any(mask[a] for a in range(1, 25) if (a - 1) % 6 < 4)
        if action:
            counts['buy_actions' if (action - 1) % 6 < 4 else 'sell_actions'] += 1
    return {**dict(counts), 'legal_action_histogram': dict(sorted(sizes.items())),
        'voluntary_wait_fraction': counts['voluntary_wait_ticks'] / counts['choice_ticks'] if counts['choice_ticks'] else None,
        'uniform_legal_action_wait_probability_on_choice_ticks': statistics.fmean(uniform_wait) if uniform_wait else None}


def ledger_summary(ledger, costs, threshold):
    counts = Counter(); pnl = defaultdict(lambda: D(0)); exits = Counter(); holds = []; opened = {}
    for r in ledger:
        if r['kind'] == 'buy_order':
            counts['buy_orders'] += 1
            if r['target'] is None:
                counts['buy_orders_without_historical_mean'] += 1
            else:
                q, target = D(r['reference_yes']), D(r['target'])
                if r['side'] == 'no':
                    q, target = 1 - q, 1 - target
                fee, premium = D(costs['fee_fraction']), D(costs['entry_price_premium'])
                edge = max(D(0), target - premium) * (1 - fee) - (q + premium) * (1 + fee)
                counts['buy_orders_positive_mean_reversion_edge'] += edge > 0
                counts['buy_orders_meeting_rule_edge_threshold'] += edge >= D(str(threshold))
        elif r['kind'] == 'buy_fill':
            counts['buy_fills'] += 1
            if r['market'] in opened:
                raise ValidationError('overlapping position')
            opened[r['market']] = timestamp(r['time'], 'fill')
        elif r['kind'] in ('sell_fill', 'settle'):
            reason = r.get('exit_reason', 'settlement'); exits[reason] += 1
            pnl[reason] += D(r['net_pnl'])
            if r.get('full_exit', r['kind'] == 'settle'):
                start = opened.pop(r['market'])
                holds.append((timestamp(r['time'], 'exit') - start).total_seconds() / 3600)
        elif r['kind'] == 'cancel':
            counts['cancel_' + r['reason']] += 1
    return {**dict(counts), 'exit_fill_counts': dict(exits), 'net_pnl_by_exit_reason': {k: str(v) for k, v in pnl.items()},
        'mean_completed_position_hours': statistics.fmean(holds) if holds else None,
        'open_positions_at_ledger_end': len(opened),
        'edge_semantics': 'Historical mean price target, not an event probability or a verified expected return. Counts are orders, including unfilled orders.'}


def summarize(root):
    consumed = {}
    def read(name, hashes=None, *, jsonl=False):
        path = root / name; digest = sha256_file(path)
        if hashes is not None and hashes.get(name) != digest:
            raise ValidationError('artifact changed: ' + name)
        consumed[name] = digest
        return rows(path) if jsonl else strict_json(path.read_text())
    report = read('report.json'); audit = read('audit.json')
    if report['partition'] != 'train' or report['final_test_opened'] or report['validation_replayed']:
        raise ValidationError('training-only report required')
    if audit['status'] != 'passed' or audit['report_sha256'] != consumed['report.json']:
        raise ValidationError('matching replay audit required')
    plan = read('plan.json', {'plan.json': report['plan_sha256']})
    training = read('training_report.json', {'training_report.json': report['training_report_sha256']})
    spec = plan['spec']; env = plan['original_config']['environment']; arms = {}
    for name, trained in training['models'].items():
        prefix = f'cost_assumption/{spec["order_ttl_seconds"]}/{name}'
        dd = read(prefix + '.decisions.jsonl', report['artifact_hashes'], jsonl=True)
        ll = read(prefix + '.ledger.jsonl', report['artifact_hashes'], jsonl=True)
        episodes = read(f'models/{name}/episodes.jsonl', training['artifact_hashes'], jsonl=True)
        progress = f'models/{name}/progress.csv'; digest = sha256_file(root / progress)
        if training['artifact_hashes'][progress] != digest:
            raise ValidationError('optimizer log changed')
        consumed[progress] = digest
        with (root / progress).open() as stream:
            logs = list(csv.DictReader(stream))
        ev = [float(r['train/explained_variance']) for r in logs if r.get('train/explained_variance')]
        by_fee = defaultdict(list)
        for episode in episodes:
            by_fee[episode['execution_costs']['fee_fraction']].append({k: episode[k] for k in ('timesteps', 'final_cash', 'entry_count')})
        arms[name] = {'decisions': decision_summary(dd),
            'ledger': ledger_summary(ll, spec['scenarios']['cost_assumption'], env['min_round_trip_edge']),
            'training': {'timesteps': trained['actual_timesteps'], 'complete_episodes': trained['episodes'],
                'rollout_iterations': len(logs), 'episodes_by_fee': dict(by_fee),
                'csv_explained_variance_min': min(ev) if ev else None,
                'csv_explained_variance_max': max(ev) if ev else None,
                'csv_explained_variance_negative_count': sum(v < 0 for v in ev),
                'csv_explained_variance_count': len(ev),
                'last_optimizer_metrics': trained['last_optimizer_metrics']}}
        metrics = report['scenarios']['cost_assumption'][str(spec['order_ttl_seconds'])][name]['metrics']
        if (arms[name]['ledger'].get('buy_fills', 0) != metrics['entry_count']
                or abs(sum(D(v) for v in arms[name]['ledger']['net_pnl_by_exit_reason'].values()) - D(metrics['net_pnl'])) > D('.000001')):
            raise ValidationError('ledger/report mismatch')
    signals_root = Path(plan['paths']['signals'])
    def signal_read(name, digest):
        path = signals_root / name
        if sha256_file(path) != digest:
            raise ValidationError('signal artifact changed: ' + name)
        consumed['signals/' + name] = digest
        return rows(path) if name.endswith('.jsonl') else strict_json(path.read_text())
    gen = signal_read('generation_report.json', plan['signal_generation_report_sha256'])
    gp = signal_read('plan.json', gen['plan_sha256'])
    inputs = signal_read('inputs.jsonl', gp['artifact_hashes']['inputs.jsonl'])
    results = signal_read('results.jsonl', gen['results_sha256'])
    inp = {r['sample_id']: r['input'] for r in inputs}
    evidence_counts = Counter(); ages = []; sources = set(); empty = 0; deltas = []
    for r in inputs:
        x = r['input']; t = timestamp(x['observation_time'], 'cutoff'); external = []
        for e in x['evidence']:
            published = timestamp(e['published_at'], 'published'); available = timestamp(e['available_at'], 'available')
            if not published <= available <= t:
                raise ValidationError('future/invalid evidence timing')
            if not e['source'].startswith('foretellmesh:'):
                external.append(e); sources.add(e['evidence_id']); ages.append((t - published).total_seconds() / 86400)
        evidence_counts[len(external)] += 1; empty += not external
    for r in results:
        if r['result']['status'] == 'completed':
            deltas.append(abs(r['result']['prediction']['probability'] - inp[r['sample_id']]['market']['probability']))
    h = spec['ppo']; decay = h['gamma'] * h['gae_lambda']
    return {'status': 'completed', 'kind': 'frozen_training_artifact_diagnostic',
        'source_report_sha256': consumed['report.json'], 'consumed_artifact_hashes': consumed,
        'script_sha256': sha256_file(Path(__file__)), 'arms': arms,
        'signals': {'jobs': len(inputs), 'successful': len(deltas), 'external_evidence_count_histogram': dict(evidence_counts),
            'unique_external_evidence_ids': len(sources), 'jobs_without_external_evidence': empty,
            'external_evidence_age_days_min': min(ages) if ages else None,
            'external_evidence_age_days_median': statistics.median(ages) if ages else None,
            'external_evidence_age_days_max': max(ages) if ages else None,
            'forecast_market_delta_max': max(deltas) if deltas else None},
        'setup': {'ppo': h, 'independent_event_groups': plan['training_data']['event_groups'],
            'episode_decision_steps': plan['training_data']['decision_steps'],
            'action_count': 25, 'max_holding_hours': env['max_holding_seconds'] / 3600,
            'gae_direct_td_residual_weights': {str(hours): decay ** (hours * 3600 / env['step_seconds']) for hours in (24, 48, 72)},
            'gae_note': 'Weights on future TD residuals, not a hard memory horizon; critic bootstrapping can carry longer-term value.',
            'optimizer_note': 'CSV is dumped before train(); final CSV optimizer metrics lag the final post-training metrics. Explained variance is batch-specific, not evidence of profit/convergence.'},
        'limitations': ['No optimizer/causal method comparison performed.',
            'Choice masks and exits are on each policy trajectory, not identical counterfactual states.',
            'Uniform-action probability is an encoding diagnostic, not measured PPO probability.',
            'Historical mean-reversion edge is a proxy; a nonpositive value does not prove a trade lacks all possible advantages.',
            'Training episodes use evolving stochastic policies and alternating fees; they are not frozen checkpoint evaluations.'],
        'new_historical_training': False, 'llm_calls': 0, 'final_test_opened': False, 'validation_replayed': False,
        'default_promotion': False, 'real_orders_sent': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValidationError('diagnostic output already exists')
    result = summarize(args.run)
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json_text(result))
    print(json_text({'status': result['status'], 'signals': result['signals'], 'setup': result['setup']}))


if __name__ == '__main__':
    main()
