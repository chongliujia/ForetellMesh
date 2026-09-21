"""Bounded, training-only team research with an append-only attempt history.

Adaptation inside this loop is training research, never an independent test.
The frozen base is loaded once; failed and duplicate hypotheses consume budget.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
import shutil

from .data import strict_json, sha256_file
from .evaluation import json_text
from .schema import fields, timestamp
from .synthetic_sft import canonical_hash
from .team_learning import require
from .team_executable_methods import bounded_text
from .team_executable_experiment import prepare as prepare_original
from .team_discovery import DiscoveryRunner
from .team_staged_methods import run_staged_workflow
from . import team_variable_contracts as typed
from .mean_reversion import HistoricalBars
from .team_rsi_experiment import now
from .team_lifecycle_experiment import RecordedBackend, AuditRunner

SELECT = '''You coordinate an offline TRAINING research team. Choose 1..8 market_ids from the supplied title index
whose FULL historical rules should be retrieved for the next hypothesis. Discover relationships yourself; no fixed
strategy categories. Review prior failed attempts and choose a different investigation or explicitly justify a revision.
Return ONLY JSON with exactly market_ids and reason (short explanation). Use only exact supplied market IDs.
The titles help retrieval, not establish a relation. Do not infer outcomes, current news or profitability.
This is bounded training research; every attempt is retained and nothing is independently validated.'''
REFLECT = '''You are the team's learning member. Review the supplied deterministic attempt feedback.
Return ONLY JSON with exactly feedback_sha256, next_focus, data_requests.
Copy feedback_sha256 exactly. next_focus is a concrete next investigation or change under 400 characters.
data_requests is a list of at most 3 objects each exactly quantity, unit, why, historical_timestamp_requirement.
Each field is a nonempty string under 240 characters. Request only observations worth obtaining, explain WHY they
could test the hypothesis, and specify publication/as-of timestamps needed. Use [] when no worthwhile request.
Do not claim unavailable evidence exists, invent results, or call schema success predictive skill.
Missing evidence and failed tests are experience. You may abandon the hypothesis and propose another next focus.
This is training feedback; no promotion, fine-tuning or independent efficacy claim is authorized by a training score.'''


def admit_catalogue(rows, groups, as_of):
    """Filter partition BEFORE projecting rules/labels; never expose held-out rules."""
    catalogue = {}; labels = {}
    for row in rows:
        if groups.get(row['event_group_id']) != 'train': continue
        proof = row['proof']
        if timestamp(proof['initialized_at'], 'init') > timestamp(as_of, 'as of'): continue
        mid = row['market_id']; require(mid not in catalogue, 'duplicate admitted market')
        record = {'market_id': mid, 'event_group_id': row['event_group_id'],
                  'question': proof['historical_question'], 'initialized_at': proof['initialized_at']}
        record['record_sha256'] = canonical_hash(record); catalogue[mid] = record
        labels[mid] = {k: proof[k] for k in ('outcome', 'resolution_time', 'available_at')}
    return catalogue, labels


def loop_for(config):
    mode = config.get('handoff_protocol', 'legacy_free_text')
    if mode in ('executable_review_boundary_v1', 'executable_data_catalogue_v1', 'executable_structured_history_v1', 'executable_grounded_revision_v1'):
        from functools import partial
        from .team_research_handoff import run_loop as run_handoff, summary
        return partial(run_handoff, derive_task_change=True, multi_input=True, named_prediction_kind=True,
                       semantic_gate=True, task_negotiation=True, semantic_advisory=True,
                       data_discovery=mode in ('executable_data_catalogue_v1','executable_structured_history_v1', 'executable_grounded_revision_v1'),
                       structured_history=mode in ('executable_structured_history_v1', 'executable_grounded_revision_v1'),
                       grounded_revision=mode=='executable_grounded_revision_v1'), summary
    require(mode in ('legacy_free_text', 'executable_task_v1', 'executable_task_v2', 'executable_multi_input_v1', 'executable_multi_input_v2', 'executable_semantic_consistency_v1', 'executable_task_negotiation_v1'), 'unknown handoff protocol')
    if mode in ('executable_task_v1', 'executable_task_v2', 'executable_multi_input_v1', 'executable_multi_input_v2', 'executable_semantic_consistency_v1', 'executable_task_negotiation_v1'):
        from functools import partial
        from .team_research_handoff import run_loop as run_handoff, summary
        if mode in ('executable_multi_input_v1', 'executable_multi_input_v2', 'executable_semantic_consistency_v1', 'executable_task_negotiation_v1'):
            return partial(run_handoff, derive_task_change=True, multi_input=True,
                           named_prediction_kind=mode in ('executable_multi_input_v2', 'executable_semantic_consistency_v1', 'executable_task_negotiation_v1'),
                           semantic_gate=mode in ('executable_semantic_consistency_v1', 'executable_task_negotiation_v1'),
                           task_negotiation=mode=='executable_task_negotiation_v1'), summary
        return (partial(run_handoff, derive_task_change=True) if mode == 'executable_task_v2' else run_handoff), summary
    return run_loop, summarize


def prepare(config):
    loop_for(config)
    require(config['kind'] == 'team_autonomous_research_loop_v1', 'invalid loop kind')
    require(type(config['max_attempts']) is int and 1 <= config['max_attempts'] <= 10, 'invalid attempt budget')
    template = Path(config['template_config'])
    require(sha256_file(template) == config['template_sha256'], 'template changed')
    prior = strict_json(template.read_text())
    base, manifest, old, _, _, protocol, context = prepare_original(prior)
    root = Path(base['cohort']); report = strict_json((root/'report.json').read_text())
    require(sha256_file(root/'markets.jsonl') == report['artifact_hashes']['markets.jsonl'], 'cohort markets changed')
    rows = [strict_json(line) for line in (root/'markets.jsonl').read_text().splitlines()]
    catalogue, labels = admit_catalogue(rows, report['groups'], protocol['as_of'])
    require(set(old) <= set(catalogue), 'expanded catalogue lost old training markets')
    require(all(old[k] == catalogue[k] for k in old), 'historical catalogue changed')
    start = min(timestamp(r['initialized_at'], 'init') for r in catalogue.values())-timedelta(days=30)
    feed = HistoricalBars.from_store(Path(base['price_store'])/'prices.sqlite', set(catalogue), start,
                                    timestamp(protocol['as_of'], 'cutoff'))
    inventory = {'previous_job_catalogue_contracts': len(old), 'admitted_research_contracts': len(catalogue),
        'admitted_event_groups': len({r['event_group_id'] for r in catalogue.values()}),
        'added_market_ids': sorted(set(catalogue)-set(old)),
        'cohort_partition_counts': dict(Counter(report['groups'][r['event_group_id']] for r in rows)),
        'cohort_report_sha256': base['cohort_report_sha256'],
        'markets_sha256': report['artifact_hashes']['markets.jsonl'],
        'limitations': ['Expansion removes old fixed-entry-time restriction, not proof or split requirements.',
                       'Raw archive inventory is not an admitted training set. No new external source captured.']}
    selection_path = Path(report['sources']['selection'])/'report.json'
    selection = strict_json(selection_path.read_text())
    archive_path = Path(config['archive_catalog_report'])
    require(sha256_file(archive_path) == selection['catalog_report_sha256'], 'archive inventory changed')
    archive = strict_json(archive_path.read_text())
    inventory['archive_inventory'] = {
        'catalog_report_sha256': sha256_file(archive_path),
        'discovery_report_sha256': sha256_file(selection_path),
        'catalog_markets': archive['counts']['markets_rows'],
        'archived_trade_rows': archive['counts']['trades_rows'],
        'eligible_discovery_candidates': selection['counts']['eligible_discovery_candidates'],
        'fixed_discovery_sample': selection['counts']['selected'],
        'counts_are_inventory_not_training_admission': True}
    context = {k: deepcopy(v) for k, v in context.items() if k not in ('catalogue', 'price_coverage', 'candidate', 'candidate_source_reflection_sha256')}
    context['phase'] = 'offline_adaptive_training_research'
    context['historically_deployed'] = False
    if config.get('handoff_protocol') in ('executable_semantic_consistency_v1', 'executable_task_negotiation_v1', 'executable_review_boundary_v1', 'executable_data_catalogue_v1', 'executable_structured_history_v1', 'executable_grounded_revision_v1'):
        from .team_semantic_consistency import build_probes
        context['semantic_probe_cases'] = build_probes(config, catalogue)
    if config.get('handoff_protocol') in ('executable_task_negotiation_v1', 'executable_review_boundary_v1', 'executable_data_catalogue_v1', 'executable_structured_history_v1', 'executable_grounded_revision_v1'):
        if config.get('handoff_protocol') == 'executable_grounded_revision_v1':
            from .team_grounded_revision import load_reference
        else:
            from .team_task_negotiation import load_reference
        context['task_negotiation_reference'] = load_reference(config, catalogue)
    return base, manifest, catalogue, feed, labels, protocol, context, inventory


def validate_selection(value, catalogue):
    fields(value, {'market_ids', 'reason'}, 'research selection'); bounded_text(value['reason'], 400)
    ids = value['market_ids']
    require(isinstance(ids, list) and 1 <= len(ids) <= 8 and all(isinstance(k, str) for k in ids)
            and len(set(ids)) == len(ids) and set(ids) <= set(catalogue), 'invalid retrieval market IDs')
    return deepcopy(value)


def relation_fingerprint(relation):
    if "predictions" in relation:
        from .team_multi_input import relation_fingerprint as graph_fingerprint
        return graph_fingerprint(relation)
    # Wording/variable aliases are not a new data experiment. A change of measured
    # inputs, lag, direction, horizon or contract bindings remains a new attempt.
    body = {k: deepcopy(relation[k]) for k in ('forecast_target', 'horizon_days')}
    body['bindings'] = sorted(relation['bindings'], key=canonical_hash)
    body['measurements'] = sorted([{k: v[k] for k in ('quantity', 'unit', 'role', 'lag_days')}
                                   for v in relation['variables']], key=canonical_hash)
    return canonical_hash(body)


class UniqueResearchRunner:
    def __init__(self, runner, seen): self.runner = runner; self.seen = seen; self.duplicate = None
    def structured(self, role, instruction, context, upstream, validator):
        value = self.runner.structured(role, instruction, context, upstream, validator)
        if role == 'relation_researcher' and value['relation'] is not None:
            key = relation_fingerprint(value['relation'])
            if key in self.seen:
                self.duplicate = deepcopy(value)
                raise ValueError('duplicate_relation: same bindings, target, horizon and measured inputs already attempted')
            self.seen.add(key)
        return value


def feedback_for(workflow):
    relation = (workflow.get('relation') or {}).get('relation')
    check = workflow.get('data_check') or {}; result = workflow.get('result') or {}
    value = {'status': workflow['status'], 'stop_reason': workflow.get('stop_reason'),
        'relation': relation, 'input_check': {k: v for k, v in check.items() if k != 'observations'},
        'test': {k: v for k, v in result.items() if k != 'observations'},
        'independent_validation': False, 'effectiveness_verified': False}
    value['feedback_sha256'] = canonical_hash(value); return value


def validate_reflection(value, feedback):
    fields(value, {'feedback_sha256', 'next_focus', 'data_requests'}, 'research reflection')
    require(value['feedback_sha256'] == feedback['feedback_sha256'], 'reflection feedback mismatch')
    bounded_text(value['next_focus'], 400)
    require(isinstance(value['data_requests'], list) and len(value['data_requests']) <= 3, 'unbounded data requests')
    for row in value['data_requests']:
        fields(row, {'quantity', 'unit', 'why', 'historical_timestamp_requirement'}, 'data acquisition request')
        for text in row.values(): bounded_text(text, 240)
    return deepcopy(value)


def history_view(attempts):
    # Full immutable records are retained on disk; model memory is bounded.
    return {'attempts': [{'attempt': a['attempt'], 'status': a['feedback']['status'],
        'market_ids': (a['selection'] or {}).get('market_ids', []),
        'hypothesis': ((a['feedback'].get('relation') or {}).get('hypothesis') or '')[:200],
        'feedback_sha256': a['feedback']['feedback_sha256']} for a in attempts],
        'recent_feedback': [{k: deepcopy(a[k]) for k in ('feedback', 'reflection', 'reflection_error')}
                            for a in attempts[-2:]],
        'feedback_is_reused_training_experience': True}


def run_loop(runner, context, catalogue, feed, labels, protocol, max_attempts, freeze, save_attempt):
    require(type(max_attempts) is int and 1 <= max_attempts <= 10, 'invalid attempt budget')
    index = [{'market_id': k, 'title': row['question'].split(', description:', 1)[0],
              'initialized_at': row['initialized_at']} for k, row in sorted(catalogue.items())]
    attempts = []; seen = set()
    for number in range(1, max_attempts+1):
        start = len(runner.calls); history = history_view(attempts); selection = None
        try:
            selection = runner.structured('research_coordinator', SELECT,
                {'phase': context['phase'], 'index': index, 'attempt': number, 'budget': max_attempts}, history,
                lambda v: validate_selection(v, catalogue))
            selected = {k: catalogue[k] for k in selection['market_ids']}
            local = {**deepcopy(context), 'catalogue': list(selected.values()),
                     'retrieval_reason': selection['reason'], 'training_history': history,
                     'attempt': number, 'max_attempts': max_attempts}
            unique = UniqueResearchRunner(runner, seen)
            workflow = run_staged_workflow(unique, local, selected, feed, labels, protocol,
                lambda p: freeze(number, p), variable_contract=typed)
            if unique.duplicate is not None:
                workflow.update(status='duplicate_relation', relation=unique.duplicate)
        except (ValueError, TypeError, KeyError) as exc:
            workflow = {'status': 'selection_failed', 'stop_reason': str(exc), 'relation': None,
                        'data_check': None, 'result': None}
        feedback = feedback_for(workflow); reflection = None; error = None
        try:
            reflection = runner.structured('research_learning_member', REFLECT,
                {'phase': 'after_training_attempt', 'attempt': number}, {'feedback': feedback},
                lambda v: validate_reflection(v, feedback))
        except (ValueError, TypeError, KeyError) as exc: error = str(exc)
        record = {'attempt': number, 'selection': selection, 'workflow': workflow, 'feedback': feedback,
            'reflection': reflection, 'reflection_error': error, 'call_start': start, 'call_end': len(runner.calls),
            'previous_attempt_sha256': attempts[-1]['attempt_sha256'] if attempts else None}
        record['attempt_sha256'] = canonical_hash(record); save_attempt(record); attempts.append(record)
        print('Research attempt', number, '/', max_attempts, feedback['status'], flush=True)
    return attempts


def summarize(attempts):
    qualified = [a['attempt'] for a in attempts if (a['workflow'].get('result') or {}).get('eligible_for_independent_validation')]
    return {'attempts': len(attempts), 'statuses': dict(Counter(a['workflow']['status'] for a in attempts)),
        'scored_attempts': sum(a['workflow'].get('result') is not None for a in attempts),
        'qualified_attempts': qualified, 'successful_reflections': sum(a['reflection'] is not None for a in attempts),
        'unique_relation_specs': len({relation_fingerprint(a['feedback']['relation']) for a in attempts if a['feedback']['relation']}),
        'data_requests': [{'attempt': a['attempt'], **r} for a in attempts if a['reflection'] for r in a['reflection']['data_requests']],
        'next_stage': 'freeze_validation_design_required' if qualified else 'no_qualified_candidate_for_independent_validation'}


def run(config_path, output):
    config = strict_json(config_path.read_text())
    require(not output.exists() and output.name == config['run_name'], 'output exists/name differs')
    base, manifest, catalogue, feed, labels, protocol, context, inventory = prepare(config)
    from .lora_probe import verify_model_manifest
    from .peft_runtime import SharedPeftExecutor, PeftTextBackend
    model_path, _ = verify_model_manifest(manifest, base)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'BF16 GPU unavailable')
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output/'config.json'); shutil.copyfile(manifest, output/'model_manifest.json')
    shutil.copytree(Path(__file__).parent, output/'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    for name, value in [('context', context), ('catalogue', catalogue), ('inventory', inventory)]:
        (output/(name+'.json')).write_text(json_text(value))
    registration = {'registered_at': now(), 'config_sha256': sha256_file(config_path),
        'context_sha256': canonical_hash(context), 'catalogue_sha256': canonical_hash(catalogue),
        'labels_sha256': canonical_hash(labels), 'protocol_sha256': canonical_hash(protocol),
        'source_hashes': {str(p.relative_to(output)): sha256_file(p) for p in (output/'source_snapshot').rglob('*.py')},
        'adaptive_training_only': True, 'max_attempts': config['max_attempts'], 'score_based_run_restarts': False}
    (output/'registration.json').write_text(json_text(registration))
    report = {'kind': config['kind'], 'status': 'running', 'started_at': now(),
        'registration_sha256': sha256_file(output/'registration.json'), 'inventory': inventory,
        'base_model': base['model'], 'model_revision': base['model_revision'], 'seed': base['seed'],
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'packages': {p: version(p) for p in ('torch', 'transformers', 'peft', 'langgraph')},
        'effectiveness_verified': False, 'fine_tuning_triggered': False, 'final_test_opened': False,
        'development_opened': False, 'independent_validation': False, 'net_profit_evaluated': False,
        'real_orders_sent': 0, 'strategy_promoted': False}
    if config.get('handoff_protocol') in ('executable_review_boundary_v1', 'executable_data_catalogue_v1', 'executable_structured_history_v1', 'executable_grounded_revision_v1'):
        from .team_review_boundary import POLICY
        report['review_boundary'] = deepcopy(POLICY)
    def save(): (output/'report.json').write_text(json_text(report))
    save(); runner = None
    try:
        set_seed(base['seed']); torch.backends.cuda.matmul.allow_tf32 = False
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', use_safetensors=True)
        runner = DiscoveryRunner(PeftTextBackend(SharedPeftExecutor(model), tok,
            max_context_tokens=12288, max_new_tokens=config['max_new_tokens'], sampling=None, stop_on_json_object=True), None)
        def freeze(number, proposal):
            path = output/f'proposal_{number:02}.json'; require(not path.exists(), 'proposal already frozen')
            path.write_text(json_text({'proposal': proposal, 'frozen_at': now(), 'calls_sha256': canonical_hash(runner.calls)}))
        def save_attempt(record):
            path = output/f'attempt_{record["attempt"]:02}.json'; require(not path.exists(), 'attempt already written')
            path.write_text(json_text(record)); (output/'calls.json').write_text(json_text(runner.calls))
            report['completed_attempts'] = record['attempt']; save()
        loop, summary = loop_for(config)
        live_context = deepcopy(context)
        if 'semantic_probe_cases' in live_context:
            from .team_semantic_consistency import run_probes
            probes = run_probes(runner, live_context.pop('semantic_probe_cases'), catalogue)
            (output/'semantic_probes.json').write_text(json_text(probes))
            live_context['semantic_probe_qualified'] = probes['qualified_for_this_bounded_run']
            report['semantic_probe_qualified'] = probes['qualified_for_this_bounded_run']
        attempts = loop(runner, live_context, catalogue, feed, labels, protocol, config['max_attempts'], freeze, save_attempt)
        require(all(not p.requires_grad for p in model.parameters()), 'weights unfrozen')
        report.update(status='completed', finished_at=now(), all_parameters_frozen=True, base_model_loads=1,
            model_calls=len(runner.calls), invalid_calls=sum(c['error'] is not None for c in runner.calls),
            model_call_seconds=sum(c['seconds'] for c in runner.calls), summary=summary(attempts),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report['artifact_hashes'] = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != 'report.json'}
        save()
    except BaseException as exc:
        if runner is not None: (output/'calls.json').write_text(json_text(runner.calls))
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=now()); save(); raise
    return report


def audit(output):
    report = strict_json((output/'report.json').read_text()); config = strict_json((output/'config.json').read_text())
    registration = strict_json((output/'registration.json').read_text())
    require(report['status'] == 'completed' and report['registration_sha256'] == sha256_file(output/'registration.json'), 'registration changed')
    for name, digest in {**registration['source_hashes'], **report['artifact_hashes']}.items():
        path = (output/name).resolve(); require(path.is_relative_to(output.resolve()) and sha256_file(path) == digest, 'artifact changed')
    _, _, catalogue, feed, labels, protocol, context, inventory = prepare(config)
    for key, value in [('context', context), ('catalogue', catalogue), ('labels', labels), ('protocol', protocol)]:
        require(canonical_hash(value) == registration[key+'_sha256'], 'input changed: '+key)
    require(inventory == report['inventory'], 'inventory changed')
    if config.get('handoff_protocol') in ('executable_review_boundary_v1', 'executable_data_catalogue_v1', 'executable_structured_history_v1', 'executable_grounded_revision_v1'):
        from .team_review_boundary import POLICY
        require(report['review_boundary'] == POLICY, 'review authority changed')
    calls = strict_json((output/'calls.json').read_text()); backend = RecordedBackend(calls)
    runner = AuditRunner(backend, None, {'min_review_seconds': 3600, 'max_review_seconds': 2592000,
        'max_signal_valid_seconds': 604800, 'max_order_ttl_seconds': 86400, 'failure_review_seconds': 86400})
    frozen_numbers = set()
    def freeze(number, proposal):
        expected = strict_json((output/f'proposal_{number:02}.json').read_text())
        require(expected['proposal'] == proposal and expected['calls_sha256'] == canonical_hash(runner.calls), 'proposal/order changed')
        frozen_numbers.add(number)
    def verify_attempt(record):
        require(record == strict_json((output/f'attempt_{record["attempt"]:02}.json').read_text()), 'attempt replay differs')
    loop, summary = loop_for(config)
    live_context = deepcopy(context)
    if 'semantic_probe_cases' in live_context:
        from .team_semantic_consistency import run_probes
        probes = run_probes(runner, live_context.pop('semantic_probe_cases'), catalogue)
        require(probes == strict_json((output/'semantic_probes.json').read_text()), 'semantic probes replay differs')
        require(report['semantic_probe_qualified'] == probes['qualified_for_this_bounded_run'], 'probe qualification differs')
        live_context['semantic_probe_qualified'] = probes['qualified_for_this_bounded_run']
    attempts = loop(runner, live_context, catalogue, feed, labels, protocol, config['max_attempts'], freeze, verify_attempt)
    require(backend.index == len(calls) and canonical_hash(runner.calls) == canonical_hash(calls), 'calls replay differs')
    require(summary(attempts) == report['summary'], 'summary differs')
    require(len(frozen_numbers) == len(list(output.glob('proposal_*.json'))), 'extra proposal')
    result = {'status': 'passed', 'report_sha256': sha256_file(output/'report.json'), 'attempts_replayed': len(attempts),
        'model_calls_replayed': len(calls), 'new_model_calls': 0, 'gpu_required': False}
    (output/'audit.json').write_text(json_text(result)); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('run', 'audit', 'preflight'))
    parser.add_argument('--config', type=Path); parser.add_argument('--output', type=Path); args = parser.parse_args()
    if args.action == 'preflight':
        *_, inventory = prepare(strict_json(args.config.read_text())); print(json_text(inventory))
    else: print(json_text(run(args.config, args.output) if args.action == 'run' else audit(args.output)))

if __name__ == '__main__': main()
