"""Rebuild causal evidence/price diagnostics for the frozen training pilot."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

from foretellmesh.data import sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.evidence_forecast import ARMS, audit
from foretellmesh.market_development import rows
from foretellmesh.schema import ValidationError, timestamp


def summarize(root):
    checked = audit(root)
    jobs = rows(root / 'inputs.jsonl')
    results = rows(root / 'results.jsonl')
    selected = rows(root / 'selected.jsonl')
    originals = {r['sample_id']: r for r in selected}
    indexed = {(j['original_sample_id'], j['arm']): (j, r) for j, r in zip(jobs, results)}
    arms = {}
    for arm in ARMS:
        pairs = [indexed[sid, arm] for sid in originals]
        ages, evidence_counts, differences, probability_values, failures = [], [], [], [], []
        selected_counts, cited_counts, confidence = [], [], []
        for job, result in pairs:
            value = job['input']; cutoff = timestamp(value['observation_time'], 'cutoff')
            for e in value['evidence']:
                if not timestamp(e['published_at'], 'published') <= timestamp(e['available_at'], 'available') <= cutoff:
                    raise ValidationError('evidence cutoff violated')
            evidence_counts.append(len(value['evidence']))
            if value['evidence']:
                ages.append(min((cutoff - timestamp(e['published_at'], 'published')).total_seconds() / 86400
                                for e in value['evidence']))
            for call in result['calls']:
                request = call['request']
                if request['adapter'] is not None:
                    raise ValidationError('adapter in Base diagnostic')
                if arm != 'original_market' and request['input']['market'] is not None:
                    raise ValidationError('market visible to blind arm')
                expected = {e['evidence_id']: e for e in value['evidence']}
                if any(expected.get(e['evidence_id']) != e for e in request['input']['evidence']):
                    raise ValidationError('model request has unadmitted evidence')
            if result['result']['status'] == 'completed':
                prediction = result['result']['prediction']
                p = prediction['probability']; probability_values.append(p)
                research = result['result']['stages']['research']
                selected_counts.append(len(research['evidence_ids']) + len(research['counter_evidence_ids']))
                cited_counts.append(len(prediction['key_evidence']) + len(prediction['counter_evidence']))
                confidence.append(prediction['confidence'])
                anchor = originals[job['original_sample_id']]['input']['market']['probability']
                differences.append(abs(p - anchor))
            else:
                failures.append({'sample_id': job['original_sample_id'], 'result': result['result']})
        calls = [c for _, r in pairs for c in r['calls']]
        seconds = sum(r['seconds'] for _, r in pairs)
        tokens = sum((c['usage'] or {}).get('output_tokens', 0) for c in calls)
        generation_seconds = sum((c['usage'] or {}).get('seconds', 0) for c in calls)
        arms[arm] = {'eligible': len(pairs), 'valid': len(probability_values), 'failures': failures,
                     'evidence_count_distribution': dict(Counter(evidence_counts)),
                     'newest_evidence_age_days_median': statistics.median(ages) if ages else None,
                     'mean_absolute_probability_minus_market': statistics.fmean(differences) if differences else None,
                     'copied_market_within_1e_5': sum(d <= 1e-5 for d in differences),
                     'probability_distribution': dict(Counter(probability_values)),
                     'research_selected_evidence_counts': dict(Counter(selected_counts)),
                     'forecast_cited_evidence_counts': dict(Counter(cited_counts)),
                     'self_reported_confidence_counts': dict(Counter(confidence)),
                     'model_calls': len(calls), 'workflow_seconds': seconds,
                     'output_tokens': tokens, 'output_tokens_per_generation_second': tokens / generation_seconds if generation_seconds else None,
                     'peak_allocated_bytes': max(r['peak_allocated_bytes'] for _, r in pairs)}
        arms[arm]['cached_workflows'] = sum('cached_from' in r for _, r in pairs)
        arms[arm]['new_model_calls'] = sum(len(r['calls']) for _, r in pairs if 'cached_from' not in r)
        arms[arm]['new_workflow_seconds'] = sum(r['seconds'] for _, r in pairs if 'cached_from' not in r)
    unchanged = []
    changed = []
    for sid in originals:
        a, ar = indexed[sid, 'original_blind']; b, br = indexed[sid, 'refreshed_blind']
        if a['input'] == b['input']:
            unchanged.append({'sample_id': sid,
                              'calls_identical': [(c['request'], c['output']) for c in ar['calls']] ==
                                                 [(c['request'], c['output']) for c in br['calls']]})
        else:
            changed.append(sid)
    return {'status': 'completed', 'audit': checked, 'arms': arms,
            'enriched_observations': changed, 'identical_input_controls': unchanged,
            'source_script_sha256': sha256_file(Path(__file__)),
            'source_scores_sha256': sha256_file(root / 'scores.json'),
            'final_test_opened': False, 'forecast_quality_is_not_pnl': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    (args.run / 'diagnostic.json').write_text(json_text(result))
    print(json_text(result))
