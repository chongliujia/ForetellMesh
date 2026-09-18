"""Rebuild chain supplements without changing historical model-visible input."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile

from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .polygon_rules_audit import audit_rules
from .polygon_settlement_audit import audit_settlements
from .schema import ValidationError, fields, parse_record
from .sft_data import jsonl
from .synthetic_sft import canonical_hash


def apply_bundle(staging, capture, candidates, bundle_path):
    """Caller must first rebuild staging from raw, including its outcome ledger."""
    config = strict_json(bundle_path.read_text())
    names = {'oracle_archive', 'ctf_archive', 'initial_archive', 'code_archive', 'explorer_archive', 'rules_policy'}
    fields(config, names | {'schema_version'}, 'chain supplement bundle')
    if config['schema_version'] != '1':raise ValidationError('unsupported chain bundle')
    paths = {key: (bundle_path.parent/config[key]).resolve() for key in names}
    settlements = audit_settlements(staging, capture, candidates, paths['oracle_archive'], paths['ctf_archive'])
    rules = audit_rules(staging, candidates, paths['initial_archive'], paths['code_archive'], paths['explorer_archive'], paths['rules_policy'])
    audit = {'settlements': settlements, 'rules': rules, 'bundle_sha256': sha256_file(bundle_path),
             'parent_staging_report_sha256': sha256_file(staging/'report.json')}
    version = 'sha256:'+canonical_hash(audit)
    rows = deepcopy(candidates)
    for old, new in zip(candidates, rows):
        record = new['record']; record['dataset_version'] = version; sid = record['sample_id']
        if sid in settlements['labels']:
            record['label'] = settlements['labels'][sid]
            new['blockers'].remove('exact_settlement_proof_missing')
            if new['question_proof']['question_id'] not in rules['question_ids']:
                raise ValidationError('missing corresponding rule proof')
            new['blockers'].remove('rule_update_history_review_required')
        if parse_record(old['record']).forecast_input.to_payload() != parse_record(record).forecast_input.to_payload():
            raise ValidationError('chain supplement changed model-visible input')
        # Semantic benchmark review and training supervision are independent.
        new['ready_for_sft'] = False; new['ready_for_benchmark'] = False
    audit.update(dataset_version=version, exact_settled_rows=sum(c['record']['label'] is not None for c in rows),
                 model_input_payloads_unchanged=True, samples_admitted=0,
                 blocker_counts=dict(Counter(b for c in rows for b in c['blockers'])))
    return rows, audit


def build(staging, capture, heldout_index, bundle_path, output):
    from .historical_replay_gate import verify_staging
    if output.exists():raise ValidationError('chain supplement output exists')
    _, candidates = verify_staging(staging, capture, heldout_index)
    rows, audit = apply_bundle(staging, capture, candidates, bundle_path)
    outcomes = [dict(sample_id=c['record']['sample_id'], event_group_id=c['record']['event_group_id'],
                     label=c['record']['label']) for c in rows]
    inputs = [{'sample_id': c['record']['sample_id'], 'input': parse_record(c['record']).forecast_input.to_payload(),
               'status': 'review_only; semantic_review_and_cohort_pending'} for c in rows if c['question_proof']]
    artifacts = {'candidates.jsonl': jsonl(rows), 'outcomes.jsonl': jsonl(outcomes),
                 'review_inputs.jsonl': jsonl(inputs), 'proofs.json': json_text(audit)}
    report = {'schema_version': '1', 'kind': 'historical_chain_supplement_staging', 'dataset_version': audit['dataset_version'],
              'candidate_rows': len(rows), 'exact_settled_rows': audit['exact_settled_rows'],
              'blocker_counts': audit['blocker_counts'], 'samples_admitted': 0, 'code': code_provenance(),
              'artifact_hashes': {name: sha256_bytes(value.encode()) for name, value in artifacts.items()},
              'parent_staging_report_sha256': audit['parent_staging_report_sha256'],
              'bundle_sha256': audit['bundle_sha256']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.chain-supplement-', dir=output.parent) as tmp:
        stage = Path(tmp)/'release'; stage.mkdir()
        for name, value in artifacts.items():(stage/name).write_text(value)
        (stage/'report.json').write_text(json_text(report)); stage.rename(output)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('staging', 'capture', 'heldout-index', 'bundle', 'output'):p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json_text(build(a.staging, a.capture, a.heldout_index, a.bundle, a.output)))


if __name__ == '__main__':main()
