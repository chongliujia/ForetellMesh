"""Independent signal timing, transaction-fee, and Agent-usage checks."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal as D, ROUND_DOWN
from pathlib import Path
import statistics

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.schema import ValidationError, timestamp


def summarize(root):
    report = strict_json((root / 'report.json').read_text())
    audit = strict_json((root / 'audit.json').read_text())
    if audit['status'] != 'passed' or audit['report_sha256'] != sha256_file(root / 'report.json'):
        raise ValidationError('matching replay audit required')
    plan = strict_json((root / 'plan.json').read_text())
    if sha256_file(root / 'plan.json') != report['plan_sha256']:
        raise ValidationError('plan changed')
    def checked_rows(name):
        if sha256_file(root / name) != report['artifact_hashes'][name]:
            raise ValidationError('artifact changed: ' + name)
        return rows(root / name)
    signals = checked_rows('agent_signals.jsonl'); by_market = defaultdict(list)
    for s in signals:
        by_market[s['market_id']].append(s)
    times = {mid: [timestamp(s['available_at'], 'ready') for s in rr] for mid, rr in by_market.items()}
    counts = Counter(); coverage = {}
    for scenario, ttls in report['scenarios'].items():
        fee_rate = D(plan['execution_scenarios'][scenario]['fee_fraction'])
        for ttl, arms in ttls.items():
            for arm, result in arms.items():
                prefix = f'{scenario}/{ttl}/{arm}'
                decisions = checked_rows(prefix + '.decisions.jsonl')
                available = candidates = ticks = attempts = 0
                for d in decisions:
                    t = timestamp(d['time'], 'decision'); expected = []
                    for mid in d['slots']:
                        candidates += 1
                        i = bisect_right(times.get(mid, []), t) - 1
                        s = by_market[mid][i] if i >= 0 else None
                        sid = (s['sample_id'] if s is not None and s['content'] is not None
                               and t < timestamp(s['expires_at'], 'expiry') else None)
                        expected.append(sid); available += sid is not None
                    if (expected != d['agent_signal_ids'] or not d['mask'][0] or d['weekly_due']
                            or d['agent_content_enabled'] != arm.startswith('multi_agent_')
                            or d['market_anchor_only'] != arm.startswith('market_anchor_')):
                        raise ValidationError('signal timing/content ablation or cash action differs')
                    ticks += any(expected); counts['decisions_checked'] += 1
                    if d['executed_action']:
                        j, op = divmod(d['executed_action'] - 1, 6)
                        attempts += op < 4 and expected[j] is not None
                m = result['metrics']
                if (ticks != m['decision_ticks_with_signal'] or available != m['candidate_slots_with_signal']
                        or candidates != m['total_candidate_slots'] or attempts != m['entry_attempts_with_signal']
                        or abs(m['reward_sum_usd'] - (float(m['final_cash']) - 100)) > 1e-7):
                    raise ValidationError('coverage/reward identity differs')
                if arm == 'cash_only' and (D(m['final_cash']) != 100 or m['entry_count'] != 0):
                    raise ValidationError('cash baseline failed')
                for fill in checked_rows(prefix + '.ledger.jsonl'):
                    if fill['kind'] not in ('buy_fill', 'sell_fill'):
                        continue
                    notional = (D(fill['shares']) * D(fill['price'])).quantize(D('.000001'), rounding=ROUND_DOWN)
                    expected_fee = (notional * fee_rate).quantize(D('.000001'), rounding=ROUND_DOWN)
                    if D(fill['fee']) != expected_fee:
                        raise ValidationError('fill fee incorrect')
                    counts['fill_fees_checked'] += 1
                coverage[f'{scenario}/{arm}'] = {'decision_ticks': len(decisions), 'ticks_with_signal': ticks,
                    'candidate_slots': candidates, 'slots_with_signal': available,
                    'slot_coverage': available / candidates if candidates else 0, 'entry_attempts_with_signal': attempts}
                counts['arms_checked'] += 1
    signals_root = Path(plan['paths']['signals'])
    gen = strict_json((signals_root / 'generation_report.json').read_text())
    if sha256_file(signals_root / 'generation_report.json') != plan['signal_generation_report_sha256']:
        raise ValidationError('generation report changed')
    if sha256_file(signals_root / 'results.jsonl') != gen['results_sha256']:
        raise ValidationError('raw signals changed')
    results = rows(signals_root / 'results.jsonl')
    gp = strict_json((signals_root / 'plan.json').read_text())
    if (sha256_file(signals_root / 'plan.json') != gen['plan_sha256']
            or sha256_file(signals_root / 'inputs.jsonl') != gp['artifact_hashes']['inputs.jsonl']):
        raise ValidationError('Agent input provenance changed')
    inputs = {r['sample_id']: r['input'] for r in rows(signals_root / 'inputs.jsonl')}
    success = [r for r in results if r['result']['status'] == 'completed']
    deltas = [r['result']['prediction']['probability'] - inputs[r['sample_id']]['market']['probability'] for r in success]
    calls = [call for r in results for call in r['calls']]
    valid = sum(t['status'] == 'valid' for r in results for t in r['result']['trace'])
    invalid = sum(t['status'] == 'invalid_output' for r in results for t in r['result']['trace'])
    generated = sum((c['usage'] or {}).get('output_tokens', 0) for c in calls)
    return {'status': 'passed', 'source_report_sha256': sha256_file(root / 'report.json'),
        'source_script_sha256': sha256_file(Path(__file__)), 'checks': dict(counts), 'coverage': coverage,
        'agents': {'jobs': len(results), 'successful_jobs': len(success), 'calls': len(calls),
            'schema_valid_calls': valid, 'schema_invalid_calls': invalid,
            'failure_stages': dict(Counter(r['result'].get('stage') for r in results if r['result']['status'] != 'completed')),
            'forecast_exact_market_copies': sum(abs(d) < 1e-12 for d in deltas),
            'forecasts_within_1e_5_of_market': sum(abs(d) <= 1e-5 for d in deltas),
            'forecast_max_absolute_difference_from_market': max((abs(d) for d in deltas), default=None),
            'forecast_mean_absolute_difference_from_market': statistics.fmean(abs(d) for d in deltas) if deltas else None,
            'quant_views': dict(Counter(r['result']['stages']['market_quant']['market_view'] for r in success)),
            'game_views': dict(Counter(r['result']['stages']['game_theory']['market_view'] for r in success)),
            'mean_workflow_seconds': statistics.fmean(r['seconds'] for r in results),
            'total_workflow_seconds': sum(r['seconds'] for r in results), 'generated_tokens': generated,
            'peak_allocated_gib': max(r['peak_allocated_bytes'] for r in results) / 2**30}}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); output = args.run / 'agent_checks.json'
    if output.exists():
        raise ValidationError('Agent checks already exist')
    result = summarize(args.run); output.write_text(json_text(result))
    print(json_text({k: result[k] for k in ('status', 'checks', 'agents')}))


if __name__ == '__main__':
    main()
