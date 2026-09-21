"""Reprocess the V3 recorded prefix after the conservative null transport fix.

This is a versioned CPU compatibility check, not a new GPU run or historical audit.
The old run, failed call and unused repair remain immutable.
"""
import argparse
from pathlib import Path
import shutil

from foretellmesh import team_variable_contracts as typed
from foretellmesh.data import strict_json, sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_executable_experiment import prepare
from foretellmesh.team_learning import require
from foretellmesh.team_lifecycle_experiment import RecordedBackend
from foretellmesh.team_staged_methods import run_staged_workflow


def run(source, output):
    require(not output.exists(), 'compatibility output exists')
    report = strict_json((source/'report.json').read_text())
    audit = strict_json((source/'audit.json').read_text())
    require(audit['status'] == 'passed' and audit['report_sha256'] == sha256_file(source/'report.json'), 'source audit changed')
    require(report['workflow_status'] == 'data_selection_failed' and report['model_calls'] == 3, 'unexpected source run')
    for name, digest in report['artifact_hashes'].items():
        path = (source/name).resolve()
        require(path.is_relative_to(source.resolve()) and sha256_file(path) == digest, 'source artifact changed')
    calls = strict_json((source/'calls.json').read_text())
    config = strict_json((source/'config.json').read_text())
    _, _, catalogue, _, _, protocol, context = prepare(config)
    require(context == strict_json((source/'context.json').read_text()), 'source context changed')
    require(calls[0]['request']['agent'] == 'relation_researcher' and calls[1]['request']['agent'] == 'method_data_member'
            and calls[1]['error'] == 'unknown data source', 'unexpected recorded prefix')
    class NoAccess:
        def latest(self, *args): raise AssertionError('price data accessed for missing source')
        def __getitem__(self, key): raise AssertionError('target labels accessed for missing source')
    def no_freeze(value): raise AssertionError('missing source was promoted to executable plan')
    backend = RecordedBackend(calls[:2]); runner = DiscoveryRunner(backend, None)
    workflow = run_staged_workflow(runner, context, catalogue, NoAccess(), NoAccess(), protocol, no_freeze,
                                  variable_contract=typed)
    require(backend.index == 2 and workflow['status'] == 'unavailable_variables'
            and workflow['result'] is None and workflow['proposal'] is None, 'unexpected compatibility result')
    require(all(b['source_id'] is None for b in workflow['data_request']['bindings']), 'missing source became usable')
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), output/'replay_script.py')
    shutil.copytree(Path(typed.__file__).parent, output/'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    (output/'workflow.json').write_text(json_text(workflow))
    (output/'calls.json').write_text(json_text(runner.calls))
    result = {'kind': 'recorded_variable_null_transport_check_v1', 'status': 'passed',
              'source_run': str(source), 'source_report_sha256': sha256_file(source/'report.json'),
              'source_calls_sha256': sha256_file(source/'calls.json'),
              'recorded_call_indices_consumed': [0, 1], 'old_repair_call_not_needed': 2,
              'original_run_reclassified': False, 'new_model_calls': 0, 'gpu_required': False,
              'workflow_status': workflow['status'], 'normalization': 'string null plus explicit missing_reason -> JSON null',
              'price_reads': 0, 'forecast_observations_scored': 0, 'effectiveness_verified': False,
              'fine_tuning_admitted': False, 'missing_variables': workflow['data_check']['unavailable_variables'],
              'artifact_hashes': {str(p.relative_to(output)): sha256_file(p) for p in output.rglob('*') if p.is_file()}}
    (output/'report.json').write_text(json_text(result)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    print(json_text(run(args.source, args.output)))
