"""Input coverage diagnostic, separate from model research and efficacy tests."""
import argparse
from collections import Counter
from datetime import timedelta
from pathlib import Path
from foretellmesh.data import strict_json, sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.schema import timestamp, iso
from foretellmesh.team_executable_methods import fresh_price
from foretellmesh.team_research_loop import prepare
from foretellmesh.team_learning import require


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    require(not args.output.exists(), 'output already exists')
    _, _, cat, feed, labels, protocol, _, inventory = prepare(strict_json(args.config.read_text()))
    cutoff = timestamp(protocol['as_of'], 'cutoff'); dates = {}; rows = []
    # Settlement times are private to this offline inventory. They are NOT sent
    # to the running team or used to choose candidates or alter its fixed grid.
    for mid, entry in sorted(cat.items()):
        start = timestamp(entry['initialized_at'], 'init').replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=1)
        resolution = timestamp(labels[mid]['resolution_time'], 'resolution')
        points = set(); attempted = 0
        for day in range(protocol['max_days_per_target']):
            at = start+timedelta(days=day)
            if at > cutoff or at >= resolution: break
            attempted += 1
            q, error = fresh_price(feed, mid, at, protocol['max_quote_age_seconds'])
            if not error: points.add(at)
        dates[mid] = points
        rows.append({'market_id': mid, 'event_group_id': entry['event_group_id'],
            'live_days_on_fixed_grid': attempted, 'fresh_daily_inputs': len(points),
            'adjacent_fresh_day_pairs': sum(p+timedelta(days=1) in points for p in points)})
    pairs = []
    for target in sorted(cat):
        for peer in sorted(cat):
            if peer <= target: continue
            shared = dates[target] & dates[peer]
            pairs.append({'market_ids': [target, peer], 'same_event_group': cat[target]['event_group_id'] == cat[peer]['event_group_id'],
                          'shared_fresh_live_daily_inputs': len(shared)})
    result = {'kind': 'training_input_coverage_diagnostic_v1', 'config_sha256': sha256_file(args.config),
        'script_sha256': sha256_file(Path(__file__)), 'inventory': inventory, 'protocol': protocol,
        'contracts': rows, 'pairs': pairs,
        'summary': {'contracts_with_any_fresh_input': sum(bool(d) for d in dates.values()),
                    'contracts_with_at_least_8_fresh_inputs': sum(len(d) >= 8 for d in dates.values()),
                    'total_pairs': len(pairs),
                    'pairs_with_any_shared_input': sum(p['shared_fresh_live_daily_inputs'] > 0 for p in pairs),
                    'cross_group_pairs_with_any_shared_input': sum(not p['same_event_group'] and p['shared_fresh_live_daily_inputs'] > 0 for p in pairs),
                    'cross_group_pairs_with_at_least_8_shared_inputs': sum(not p['same_event_group'] and p['shared_fresh_live_daily_inputs'] >= 8 for p in pairs)},
        'interpretation': 'Offline post-hoc inventory only; not supplied to the team, no model or PnL scores. Counts depend on midnight grid, initial 90-day windows, 3-hour freshness and live boundaries. Other event times, lags and windows may have different coverage.',
        'development_opened': False, 'final_test_opened': False, 'new_model_calls': 0}
    args.output.mkdir(parents=True); (args.output/'coverage.json').write_text(json_text(result))
    print(json_text(result['summary']))

if __name__ == '__main__': main()
