"""Use-specific historical admission, replayed from raw archives before release.

This preflight never runs a model. Teacher probabilities are needed for an SFT
release, not for scoring against isolated outcomes. Every other source blocker
is retained; unknown blocker names are rejected conservatively, not ignored.
"""
import argparse
from collections import Counter
from pathlib import Path
import tempfile

from .data import sha256_file, strict_json
from .evaluation import json_text
from .historical_market import build_historical_markets
from .schema import ValidationError, parse_record, timestamp

ARTIFACTS = {'candidates.jsonl', 'outcomes.jsonl', 'review_inputs.jsonl', 'evidence.jsonl', 'exclusions.json', 'plan.json'}


def blockers(candidate, purpose):
    if purpose not in ('sft', 'development_replay'):raise ValidationError('unsupported admission purpose')
    record = parse_record(candidate['record'])
    reasons = set(candidate['blockers'])
    if purpose == 'development_replay':reasons.discard('historical_teacher_target_missing')
    # Re-derive critical checks even if a caller tampers with a ready flag.
    if record.label is None:reasons.add('exact_settlement_proof_missing')
    if record.forecast_input.market is None:reasons.add('historical_price_unusable')
    if not record.forecast_input.evidence:reasons.add('pre_cutoff_evidence_missing')
    if not record.forecast_input.observation_time < timestamp(candidate['first_public_result_time'], 'public outcome time'):
        reasons.add('observation_after_public_outcome')
    if not record.event_group_id:reasons.add('event_group_missing')
    return sorted(reasons)


def verify_staging(staging, capture, heldout_index):
    report = strict_json((staging/'report.json').read_text())
    if report.get('kind') != 'historical_prediction_market_staging' or set(report['artifact_hashes']) != ARTIFACTS:
        raise ValidationError('invalid historical staging')
    for name, digest in report['artifact_hashes'].items():
        if sha256_file(staging/name) != digest:raise ValidationError('historical artifact changed')
    # Hashes alone do not certify a manually changed blocker or readiness flag.
    with tempfile.TemporaryDirectory(prefix='foretellmesh-replay-') as tmp:
        rebuilt = Path(tmp)/'rebuilt'; actual = build_historical_markets(capture, heldout_index, rebuilt)
        if actual['artifact_hashes'] != report['artifact_hashes']:
            raise ValidationError('historical staging does not replay from raw sources')
        for key in ('counts', 'dataset_version', 'heldout_index_sha256', 'config_sha256', 'capture_requests_sha256'):
            if report[key] != actual[key]:raise ValidationError('historical provenance changed')
    return report, [strict_json(s) for s in (staging/'candidates.jsonl').read_text().splitlines()]


def preflight(staging, capture, heldout_index, config_path, chain_bundle=None):
    config = strict_json(config_path.read_text())
    if (config.get('schema_version') != '1' or config.get('purpose') != 'development_replay'
            or type(config.get('minimum_event_groups')) is not int or config['minimum_event_groups'] < 2
            or config.get('training') is not False or config.get('default_promotion') is not False):
        raise ValidationError('invalid admission policy')
    report, candidates = verify_staging(staging, capture, heldout_index)
    chain_audit = None
    if chain_bundle is not None:
        from .historical_chain_supplement import apply_bundle
        candidates, chain_audit = apply_bundle(staging, capture, candidates, chain_bundle)
    decisions = [{'sample_id': c['record']['sample_id'], 'event_group_id': c['record']['event_group_id'],
                  'platform': c['record']['dataset_source'],
                  'replay_blockers': blockers(c, 'development_replay'), 'sft_blockers': blockers(c, 'sft')}
                 for c in candidates]
    eligible = [d for d in decisions if not d['replay_blockers']]
    groups = {d['event_group_id'] for d in eligible}; platforms = {d['platform'] for d in eligible}
    cohort_blockers = []
    if len(groups) < config['minimum_event_groups']:cohort_blockers.append('insufficient_independent_event_groups')
    if set(config['required_platforms']) - platforms:cohort_blockers.append('required_platform_coverage_missing')
    # A later release must additionally freeze real splits and per-run weights.
    cohort_blockers.append('chronological_split_and_checkpoint_manifest_not_frozen')
    return {'schema_version': '1', 'kind': 'historical_replay_admission', 'status': 'not_admitted',
            'staging_report_sha256': sha256_file(staging/'report.json'),
            'dataset_version': chain_audit['dataset_version'] if chain_audit else report['dataset_version'],
            'chain_supplement': chain_audit,
            'exact_settled_rows': sum(c['record']['label'] is not None for c in candidates),
            'evaluation_policy': config, 'policy_sha256': sha256_file(config_path),
            'candidate_rows': len(candidates), 'candidate_event_groups': len({d['event_group_id'] for d in decisions}),
            'eligible_replay_rows': len(eligible), 'eligible_replay_event_groups': len(groups),
            'eligible_sft_rows': sum(not d['sft_blockers'] for d in decisions),
            'replay_blocker_counts': dict(Counter(b for d in decisions for b in d['replay_blockers'])),
            'cohort_blockers': cohort_blockers, 'decisions': decisions,
            'model_calls': 0, 'score_metrics': None, 'training_started': False,
            'notes': ['Missing teacher answers alone do not block replay or outcome scoring.',
                      'Twenty groups is a development admission floor, not proof of statistical power.',
                      'Role/contract/observation variants are not independent events.',
                      'No model ranking or forecasting claim is possible before admission.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('staging', 'capture', 'heldout-index', 'config', 'output'):p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--chain-bundle', type=Path)
    a = p.parse_args()
    if a.output.exists():raise ValidationError('admission report exists')
    r = preflight(a.staging, a.capture, a.heldout_index, a.config, a.chain_bundle)
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json_text(r))
    print(json_text({k: r[k] for k in ('status', 'candidate_rows', 'eligible_replay_rows', 'eligible_sft_rows', 'cohort_blockers')}))


if __name__ == '__main__':main()
