"""Read-only descriptive analysis of a completed, audited RSI pilot.

No candidate selection, prompt tuning, further fitting or final-test access.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import statistics

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.metrics import score_predictions
from foretellmesh.schema import timestamp
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.team_learning import require, feedback_for
from foretellmesh.team_learning_experiment import read_rows
from foretellmesh.team_rsi_experiment import make_feed


def summarize(root, output):
    require(not output.exists(), 'analysis output exists')
    report = strict_json((root/'report.json').read_text()); audit = strict_json((root/'audit.json').read_text())
    require(report['status'] == 'completed' and audit['status'] == 'passed'
            and audit['report_sha256'] == sha256_file(root/'report.json'), 'completed audit required')
    for name, digest in report['artifact_hashes'].items(): require(sha256_file(root/name) == digest, 'changed run')
    config = strict_json((root/'config.json').read_text()); jobs = read_rows(root/'development.jobs.jsonl')
    train_by_market = {}
    for row in read_rows(root/'training_examples.jsonl'):
        key = (row['event_group_id'], row['market_id'])
        require(key not in train_by_market or train_by_market[key] == row['outcome'], 'inconsistent training outcome')
        train_by_market[key] = row['outcome']
    train_groups = defaultdict(list)
    for (group, _), outcome in train_by_market.items(): train_groups[group].append(outcome)
    empirical_prior = statistics.fmean(statistics.fmean(v) for v in train_groups.values())
    feed = make_feed(config, jobs); arms = {}; tables = {}
    for arm in ('base', 'base_memory', 'lora', 'lora_memory'):
        episodes = read_rows(root/(arm+'.episodes.jsonl')); table = {}; holdings = []; drawdowns = []
        raw_ps = []; raw_ys = []; failed_roles = defaultdict(int)
        for ep in episodes:
            valid_forecast_calls = {canonical_hash(c['request']) for c in ep['calls']
                                    if c['request']['agent'] == 'forecast' and c['error'] is None}
            raw = {r['market_id']: r['probability'] for r in ep['readouts'] if r['request_sha256'] in valid_forecast_calls}
            for call in ep['calls']:
                if call['error'] is not None: failed_roles[call['request']['agent']] += 1
            forecasts = {} if ep['decision'] is None else {f['market_id']: f['probability'] for f in ep['decision']['forecast']['forecasts']}
            for m in ep['context']['markets']:
                key = ep['episode_id']+':'+m['market_id']
                table[key] = {'p': forecasts.get(m['market_id']), 'y': ep['feedback']['outcomes'][m['market_id']],
                              'q': m['input']['market']['probability'], 'group': m['event_group_id']}
                raw_ps.append(raw.get(m['market_id'])); raw_ys.append(table[key]['y'])
            account = ep['feedback']['account']; fills = {}
            for row in account['ledger']:
                if row['kind'] == 'fill': fills[row['order_id']] = timestamp(row['time'], 'fill')
                if row['kind'] == 'settle' and row['order_id'] in fills:
                    holdings.append((timestamp(row['time'], 'settlement')-fills[row['order_id']]).total_seconds()/86400)
            drawdowns.append(float(account['sampled_equity_proxy_max_drawdown_usd']))
        tables[arm] = table; fees = {}
        # Declared here as diagnostic only: same decisions, no new policy/model evaluation.
        for fee in ('0', '0.01', '0.03'):
            policy = deepcopy(config['execution_policy']); policy['fee_fraction'] = fee
            accounts = [feedback_for(j['context'], ep['decision'], j['labels'], feed, policy,
                                     ep['decision_seconds'])['account'] for j, ep in zip(jobs, episodes)]
            fees[fee] = {'mean_final_cash': statistics.fmean(float(a['final_cash']) for a in accounts),
                         'fills': sum(a['filled_trades'] for a in accounts)}
        arms[arm] = {'mean_sampled_drawdown_usd': statistics.fmean(drawdowns), 'max_sampled_drawdown_usd': max(drawdowns),
                     'filled_holding_days': holdings, 'mean_filled_holding_days': statistics.fmean(holdings) if holdings else None,
                     'forecast_readout_before_risk_pipeline_failure': score_predictions(raw_ps, raw_ys),
                     'invalid_calls_by_role': dict(failed_roles),
                     'fee_counterfactual_fixed_decisions': fees}
    common = sorted(k for k in tables['base'] if all(k in t and t[k]['p'] is not None for t in tables.values()))
    scores = {}; deltas = {}
    for arm, table in tables.items():
        rows = [table[k] for k in common]
        scores[arm] = score_predictions([r['p'] for r in rows], [r['y'] for r in rows])
    scores['market'] = score_predictions([tables['base'][k]['q'] for k in common], [tables['base'][k]['y'] for k in common])
    outcomes = [tables['base'][k]['y'] for k in common]
    scores['constant_half'] = score_predictions([.5]*len(common), outcomes)
    scores['training_event_prior'] = score_predictions([empirical_prior]*len(common), outcomes)
    for left, right in (('base_memory','base'), ('lora','base'), ('lora_memory','base_memory'), ('lora_memory','base')):
        groups = defaultdict(list)
        for k in common:
            a, b = tables[left][k], tables[right][k]
            require(a['y'] == b['y'] and a['group'] == b['group'], 'paired identity differs')
            groups[a['group']].append((a['p']-a['y'])**2-(b['p']-b['y'])**2)
        per_group = {g: statistics.fmean(v) for g, v in groups.items()}
        deltas[left+' minus '+right] = {'event_mean_brier_difference': statistics.fmean(per_group.values()) if per_group else None,
                                      'per_event_brier_difference': per_group, 'negative_means_improvement': True}
    train_episodes = read_rows(root/'train.episodes.jsonl')
    research = [e['decision']['research'] for e in train_episodes if e['decision'] is not None]
    exploration = {'training_available_market_counts': dict(Counter(len(j['context']['markets'])
                         for j in read_rows(root/'train.jobs.jsonl'))),
                   'completed_training_queries_by_tool': dict(Counter(q['tool'] for r in research for q in r['queries'])),
                   'completed_training_hypothesis_count': sum(len(r['hypotheses']) for r in research),
                   'nonempty_training_reflections': sum(bool(e['reflection'] and e['reflection']['lessons']) for e in train_episodes)}
    result = {'kind': 'team_rsi_descriptive_comparison_v1', 'source_report_sha256': sha256_file(root/'report.json'),
              'common_covered_predictions': len(common), 'common_covered_keys': common, 'common_coverage_scores': scores,
              'paired_event_differences': deltas, 'execution_diagnostics': arms,
              'empirical_prior_from_training_only': empirical_prior,
              'training_unique_markets': len(train_by_market), 'training_independent_groups': len(train_groups),
              'exploration_coverage': exploration,
              'limitations': ['Descriptive tiny development sample; no significance or generalization claim.',
                  'Report full-arm coverage alongside common coverage to avoid hiding failures.',
                  'Fees replay fixed model decisions; agents are not regenerated at counterfactual fees.',
                  'Sampled mark-to-market drawdown is not a continuous executable liquidation value.',
                  'Independent accounts cannot be summed into a single $100 portfolio return.']}
    output.mkdir(parents=True); (output/'comparison.json').write_text(json_text(result)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    print(json_text(summarize(a.run, a.output)))
