"""Tool-grounded research feedback; citations do not prove a learned method works."""
from copy import deepcopy
from pathlib import Path

from .data import strict_json, sha256_file
from .schema import fields
from .synthetic_sft import canonical_hash
from .team_learning import require
from .team_executable_methods import bounded_text

LEARNING = '''\nFor this protocol also return diagnosis, in addition to the four usual reflection fields.
diagnosis has exactly fact_ids, interpretation, proposed_change. fact_ids is 1..4 distinct IDs copied from
feedback.tool_facts; cite at least one requires_response fact when any exists. interpretation and proposed_change
are nonempty strings under 400 characters. Explain the concrete observed obstacle and how next_task or data_requests
addresses it. A citation alone does not prove that your explanation or revision is correct.
tool_facts are recorded execution/coverage results; reviewer_advice is fallible model opinion; candidate_hypothesis
is untested. Topic difference is not disproof. If joint inputs are unavailable, inspect the declared lags, source
coverage and initialization times instead of merely repeating a critic's topic judgment. Do not backfill future
prices or assume stale prices are executable. Choose contracts, hypotheses and revisions yourself; no prescribed
strategy. You may request missing measurements. Marginal time ranges do not establish joint or target coverage.
Only actual subsequent tool checks can establish whether the proposed repair worked; do not claim success now.'''


def current_catalogue(task, catalogue, registry, feed, protocol):
    """Refresh exact effective IDs after negotiation, independent of the old query."""
    from .team_data_catalogue import query_catalogue
    query = {'source_ids':[s['source_id'] for s in registry], 'market_ids':list(task['market_ids']),
        'as_of':protocol['as_of'], 'reason':'Verify coverage of the effective task after negotiation.'}
    result = query_catalogue(query, catalogue, registry, feed, protocol['as_of'])
    record = {'task_sha256':canonical_hash(task), 'result':result,
        'kind':'effective_task_catalogue_v1', 'model_selected_task':True,
        'joint_coverage_established':False, 'historical_feature':False}
    record['record_sha256'] = canonical_hash(record)
    return record


def facts(feedback):
    rows = []
    def add(kind, value, required=False):
        row = {'kind':kind, 'value':deepcopy(value), 'requires_response':required,
            'feedback_sha256':feedback['feedback_sha256']}
        row['fact_id'] = canonical_hash(row)[:16]; rows.append(row)
    status = feedback['status']
    add('workflow_status', {'status':status}, status not in ('training_screen_passed','abstained'))
    check = feedback.get('input_check') or {}
    if check:
        add('historical_input_check', {k:check[k] for k in ('status','jointly_available','failures',
            'unavailable_variables','check_sha256','protocol_sha256') if k in check},
            check.get('status') != 'ready_for_calculation')
    current = feedback.get('current_task_catalogue')
    catalogue = current['result'] if current else (feedback.get('data_catalogue') or {}).get('result')
    if catalogue:
        add('catalogue_query_scope', {'market_ids':catalogue['query']['market_ids'],
            'as_of':catalogue['query']['as_of'], 'research_cutoff':catalogue['research_cutoff'],
            'effective_task_refreshed':current is not None, 'result_sha256':catalogue['result_sha256']})
        for row in catalogue['markets']:
            member = row['market_id'] in (feedback.get('active_task') or {}).get('market_ids', [])
            add('marginal_input_coverage', {**row, 'belongs_to_effective_task':member},
                member and row['historical_blocks'] == 0)
    for failure in feedback.get('failure_details', []):
        from .team_context_comparison import error_code
        add('validation_failure', {'role':failure['role'], 'code':error_code(failure['error']),
            'error':failure['error']}, True)
    if feedback.get('test'):
        add('training_test', {k:v for k,v in feedback['test'].items() if k != 'observations'})
    return rows


def learning_view(feedback):
    value = {'feedback_sha256':feedback['feedback_sha256'], 'feedback_hash_refers_to_full_archived_record':True,
        'tool_facts':facts(feedback), 'active_task':deepcopy(feedback.get('active_task')),
        'candidate_hypothesis':{'relation':deepcopy(feedback.get('relation')), 'verified':False},
        'reviewer_advice':{'findings':deepcopy(feedback.get('semantic_consistency', [])),
            'verified':False, 'authority':'advisory_only'},
        'scope':'Offline training feedback, not evidence available at earlier historical forecast times.',
        'effectiveness_verified':False}
    value['learning_view_sha256'] = canonical_hash(value)
    return value


def validate_reflection(value, feedback, catalogue, previous):
    from .team_research_handoff import validate_reflection as validate_base
    fields(value, {'feedback_sha256','next_focus','data_requests','next_task','diagnosis'}, 'grounded reflection')
    base = validate_base({k:v for k,v in value.items() if k != 'diagnosis'}, feedback, catalogue, previous,
        derive_task_change=True)
    diagnosis = value['diagnosis']
    fields(diagnosis, {'fact_ids','interpretation','proposed_change'}, 'failure diagnosis')
    ids = diagnosis['fact_ids']; available = {r['fact_id']:r for r in facts(feedback)}
    require(isinstance(ids,list) and 1 <= len(ids) <= 4 and all(isinstance(k,str) for k in ids)
        and len(set(ids)) == len(ids) and set(ids) <= set(available), 'diagnosis must cite 1..4 actual tool fact IDs')
    required = {k for k,r in available.items() if r['requires_response']}
    # Status alone cannot substitute for a more specific available failure.
    specific = {k for k in required if available[k]['kind'] != 'workflow_status'}
    require(not required or bool(set(ids) & (specific or required)), 'diagnosis must cite a concrete failure requiring response')
    bounded_text(diagnosis['interpretation'],400); bounded_text(diagnosis['proposed_change'],400)
    return {**base, 'diagnosis':deepcopy(diagnosis)}


def extend_history(view, full_history):
    """Retain only prior facts with original provenance; never current/future feedback."""
    result = deepcopy(view); result.pop('history_view_sha256')
    previous = []
    if full_history.get('prior_training_failure'):
        previous.append(full_history['prior_training_failure']['feedback'])
        relation = previous[-1].get('relation')
        if relation:
            from .team_research_loop import relation_fingerprint
            result['reference_experiment_fingerprint'] = relation_fingerprint(relation)
    previous.extend(r['feedback'] for r in full_history.get('recent_feedback', []))
    result['past_tool_facts'] = [{'feedback_sha256':f['feedback_sha256'], 'facts':facts(f)} for f in previous]
    result['past_tool_facts_are_not_current_task'] = True
    result['history_view_sha256'] = canonical_hash(result)
    return result


def load_reference(config, catalogue):
    """Continue one exposed, unscored input-coverage failure, not a fresh evaluation."""
    from .team_research_handoff import validate_derived_task
    root = Path(config['grounded_reference_run']); report = strict_json((root/'report.json').read_text())
    require(sha256_file(root/'report.json') == config['grounded_reference_report_sha256'], 'grounded reference changed')
    require(report['status'] == 'completed' and report['all_parameters_frozen'] and not report['development_opened']
        and not report['final_test_opened'], 'grounded reference must be frozen training research')
    name = 'attempt_01.json'
    require(sha256_file(root/name) == report['artifact_hashes'][name], 'grounded reference attempt changed')
    attempt = strict_json((root/name).read_text()); feedback = attempt['feedback']
    require(feedback['status'] == 'no_joint_observations' and feedback['input_check']['jointly_available'] == 0
        and not feedback['test'], 'reference must be an unscored joint-input failure')
    validate_derived_task(attempt['active_task'], catalogue)
    return {'source_report_sha256':config['grounded_reference_report_sha256'],
        'source_attempt_sha256':report['artifact_hashes'][name], 'task':deepcopy(attempt['active_task']),
        'feedback':deepcopy(feedback), 'already_exposed_training_experience':True}
