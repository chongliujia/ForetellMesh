"""Frozen Base turns an archived candidate into a tool-executable training test."""
import argparse
from copy import deepcopy
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
import shutil

from .data import sha256_file, strict_json
from .evaluation import json_text
from .lora_probe import verify_model_manifest
from .peft_runtime import SharedPeftExecutor, PeftTextBackend
from .schema import timestamp, iso
from .synthetic_sft import canonical_hash
from .team_discovery import DiscoveryRunner
from .team_discovery_experiment import prepare as prepare_training
from .team_experience_gate import validate_admitted
from .team_executable_methods import run_workflow, fresh_price, validate_protocol
from .team_learning import require
from .team_lifecycle_experiment import AuditRunner, RecordedBackend
from .team_rsi_experiment import now, load_partition


def workflow_for(config):
    mode = config.get('method_workflow', 'single_planner')
    require(mode in ('single_planner', 'staged_team', 'typed_staged_team'), 'unknown method workflow')
    if mode == 'typed_staged_team':
        from functools import partial
        from . import team_variable_contracts
        from .team_staged_methods import run_staged_workflow
        return partial(run_staged_workflow, variable_contract=team_variable_contracts)
    if mode == 'staged_team':
        from .team_staged_methods import run_staged_workflow
        return run_staged_workflow
    return run_workflow


def prepare(config):
    workflow_for(config)
    require(config['kind'] == 'team_executable_method_diagnostic_v1'
            and config['automatic_fine_tuning'] is False and config['final_test_opened'] is False
            and config['independent_validation'] is False, 'invalid executable diagnostic scope')
    require(sha256_file(Path(config['source_config'])) == config['source_config_sha256'], 'training config changed')
    base = strict_json(Path(config['source_config']).read_text())
    _, catalogue, feed = prepare_training(base); jobs = load_partition(base, 'train')
    source = Path(config['candidate_run']); report = strict_json((source/'report.json').read_text())
    require(sha256_file(source/'report.json') == config['candidate_report_sha256'], 'candidate run changed')
    require(sha256_file(source/'reviews.json') == report['artifact_hashes']['reviews.json'], 'candidate reviews changed')
    rows = strict_json((source/'reviews.json').read_text())
    candidates = [r for r in rows if r['admitted_to_exploratory_memory']]
    validate_admitted(candidates, config['feedback_cutoff'])
    require(len(candidates) == 1, 'v1 diagnostic expects the one preserved exploratory candidate')
    labels = {}
    for job in jobs:
        for mid, label in job['labels'].items():
            require(mid not in labels or labels[mid] == label, 'inconsistent labels')
            labels[mid] = deepcopy(label)
    require(set(labels) == set(catalogue.rows), 'labels/catalogue population differs')
    protocol = {k: config[k] for k in ('as_of', 'feedback_cutoff', 'max_days_per_target', 'max_quote_age_seconds',
                                      'min_event_groups', 'min_coverage', 'partition', 'independent_validation')}
    validate_protocol(protocol)
    require(protocol['as_of'] == max(j['context']['observation_time'] for j in jobs), 'training cutoff differs')
    visible = catalogue.visible(protocol['as_of']); cutoff = timestamp(protocol['as_of'], 'as of')
    coverage = {}
    for mid, row in visible.items():
        init = timestamp(row['initialized_at'], 'init')
        start = init.replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=1)
        sampled = []
        for day in range(protocol['max_days_per_target']):
            at = start+timedelta(days=day)
            if at > cutoff: break
            quote, _ = fresh_price(feed, mid, at, protocol['max_quote_age_seconds'])
            if quote: sampled.append(day)
        coverage[mid] = {'daily_grid_fresh_quotes': len(sampled),
                         'adjacent_fresh_day_pairs': sum(d+1 in sampled for d in sampled),
                         'counts_do_not_guarantee_joint_feature_coverage': True}
    candidate = candidates[0]
    context = {'phase': 'offline_training_method_design', 'observation_time': protocol['as_of'],
               'built_at': config['feedback_cutoff'], 'historically_deployed': False,
               'candidate': candidate['output']['lesson'],
               'candidate_source_reflection_sha256': candidate['reflection_sha256'],
               'catalogue': list(visible.values()), 'price_coverage': coverage, 'protocol': protocol,
               'no_target_labels_supplied': True,
               'limitations': ['Prior candidate and source trades are reused training experience.',
                              'Rule text and historical trade prices only; absent external evidence must remain missing.']}
    if config.get('method_workflow') == 'typed_staged_team':
        from .team_variable_contracts import source_registry
        context['variable_sources'] = source_registry(base)
    return base, source/'model_manifest.json', visible, feed, labels, protocol, context


def run(config_path, output):
    config = strict_json(config_path.read_text())
    require(not output.exists() and output.name == config['run_name'], 'output exists/name differs')
    base, manifest, catalogue, feed, labels, protocol, context = prepare(config)
    model_path, _ = verify_model_manifest(manifest, base)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'BF16 GPU unavailable')
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output/'config.json'); shutil.copyfile(manifest, output/'model_manifest.json')
    shutil.copytree(Path(__file__).parent, output/'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    (output/'context.json').write_text(json_text(context))
    registration = {'registered_at': now(), 'config_sha256': sha256_file(config_path),
                    'context_sha256': sha256_file(output/'context.json'), 'labels_sha256': canonical_hash(labels),
                    'source_hashes': {str(p.relative_to(output)): sha256_file(p) for p in (output/'source_snapshot').rglob('*.py')},
                    'no_score_based_plan_retries': True, 'independent_validation': False}
    (output/'registration.json').write_text(json_text(registration))
    report = {'kind': config['kind'], 'status': 'running', 'started_at': now(),
              'registration_sha256': sha256_file(output/'registration.json'),
              'base_model': base['model'], 'model_revision': base['model_revision'], 'seed': base['seed'],
              'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
              'packages': {p: version(p) for p in ('torch', 'transformers', 'peft', 'langgraph')},
              'effectiveness_verified': False, 'fine_tuning_triggered': False, 'final_test_opened': False,
              'real_orders_sent': 0, 'independent_validation': False, 'net_profit_evaluated': False}
    def save(): (output/'report.json').write_text(json_text(report))
    save(); runner = None
    try:
        set_seed(base['seed']); torch.backends.cuda.matmul.allow_tf32 = False
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', use_safetensors=True)
        backend = PeftTextBackend(SharedPeftExecutor(model), tok, max_context_tokens=12288,
            max_new_tokens=config['max_new_tokens'], sampling=None, stop_on_json_object=True)
        runner = DiscoveryRunner(backend, None)
        def freeze(proposal):
            require(not (output/'proposal.json').exists(), 'proposal already frozen')
            (output/'proposal.json').write_text(json_text(proposal))
            (output/'proposal_freeze.json').write_text(json_text({'frozen_at': now(),
                'proposal_sha256': sha256_file(output/'proposal.json'), 'scoring_calls_so_far': 0,
                'planner_calls_sha256': canonical_hash(runner.calls)}))
            print('Executable proposal frozen; no future targets returned to planner.', flush=True)
        workflow = workflow_for(config)(runner, context, catalogue, feed, labels, protocol, freeze)
        (output/'workflow.json').write_text(json_text(workflow)); (output/'calls.json').write_text(json_text(runner.calls))
        require(all(not p.requires_grad for p in model.parameters()), 'weights unfrozen')
        result = workflow['result']
        report.update(status='completed', finished_at=now(), all_parameters_frozen=True, base_model_loads=1,
                      model_calls=len(runner.calls), invalid_calls=sum(c['error'] is not None for c in runner.calls),
                      model_call_seconds=sum(c['seconds'] for c in runner.calls),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      proposal_error=workflow['proposal_error'], review_error=workflow['review_error'],
                      result=None if result is None else {k: v for k, v in result.items() if k != 'observations'},
                      next_stage='independent_validation_not_run', strategy_promoted=False)
        if config.get('method_workflow') in ('staged_team', 'typed_staged_team'):
            report.update(workflow_status=workflow['status'], stop_reason=workflow['stop_reason'],
                input_check=None if workflow['data_check'] is None else
                    {k: v for k, v in workflow['data_check'].items() if k != 'observations'})
        report['artifact_hashes'] = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != 'report.json'}
        save()
    except BaseException as exc:
        if runner is not None: (output/'calls.json').write_text(json_text(runner.calls))
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=now()); save(); raise
    return report


def audit(output):
    report = strict_json((output/'report.json').read_text()); config = strict_json((output/'config.json').read_text())
    registration = strict_json((output/'registration.json').read_text())
    require(report['status'] == 'completed' and sha256_file(output/'registration.json') == report['registration_sha256'],
            'unfinished/changed registration')
    for name, digest in {**registration['source_hashes'], **report['artifact_hashes']}.items():
        path = (output/name).resolve()
        require(path.is_relative_to(output.resolve()) and sha256_file(path) == digest, 'artifact changed')
    _, _, catalogue, feed, labels, protocol, context = prepare(config)
    require(context == strict_json((output/'context.json').read_text()) and canonical_hash(labels) == registration['labels_sha256'],
            'source inputs changed')
    calls = strict_json((output/'calls.json').read_text()); backend = RecordedBackend(calls)
    runner = AuditRunner(backend, None, {'min_review_seconds': 3600, 'max_review_seconds': 2592000,
        'max_signal_valid_seconds': 604800, 'max_order_ttl_seconds': 86400, 'failure_review_seconds': 86400})
    def frozen(proposal):
        require(proposal == strict_json((output/'proposal.json').read_text()), 'registered proposal changed')
        record = strict_json((output/'proposal_freeze.json').read_text())
        require(record['proposal_sha256'] == sha256_file(output/'proposal.json') and record['scoring_calls_so_far'] == 0
                and record['planner_calls_sha256'] == canonical_hash(runner.calls), 'registration order differs')
    rebuilt = workflow_for(config)(runner, context, catalogue, feed, labels, protocol, frozen)
    require(rebuilt == strict_json((output/'workflow.json').read_text()) and backend.index == len(calls), 'workflow replay differs')
    expected = None if rebuilt['result'] is None else {k: v for k, v in rebuilt['result'].items() if k != 'observations'}
    require(report['result'] == expected and report['proposal_error'] == rebuilt['proposal_error']
            and report['review_error'] == rebuilt['review_error'], 'report summary differs from replay')
    if config.get('method_workflow') in ('staged_team', 'typed_staged_team'):
        expected_check = None if rebuilt['data_check'] is None else {k: v for k, v in rebuilt['data_check'].items() if k != 'observations'}
        require(report['workflow_status'] == rebuilt['status'] and report['stop_reason'] == rebuilt['stop_reason']
                and report['input_check'] == expected_check, 'stage report differs from replay')
    result = {'status': 'passed', 'report_sha256': sha256_file(output/'report.json'), 'model_calls_replayed': len(calls),
              'new_model_calls': 0, 'gpu_required': False, 'observation_count': len((rebuilt['result'] or {}).get('observations', []))}
    (output/'audit.json').write_text(json_text(result)); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('run', 'audit', 'preflight'))
    parser.add_argument('--config', type=Path); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'preflight':
        _, _, cat, _, _, protocol, context = prepare(strict_json(args.config.read_text()))
        print(json_text({'contracts': len(cat), 'protocol': protocol, 'coverage': context['price_coverage']}))
    else: print(json_text(run(args.config, args.output) if args.action == 'run' else audit(args.output)))


if __name__ == '__main__': main()
