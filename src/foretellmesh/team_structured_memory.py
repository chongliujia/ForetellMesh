"""Bounded model-facing history views with complete, separate audit records.

Past response prose is not replayed as a new assignment. Current task, data and
validated tool feedback remain visible. This is context management, not training.
"""
from copy import deepcopy

from .synthetic_sft import canonical_hash


def negotiation_view(record):
    return {'status':record['status'],'negotiation_sha256':record['negotiation_sha256'],
        'effective_task':deepcopy(record['effective_task']),
        'decision':(record['decision'] or {}).get('decision'),
        'original_task_sha256':record.get('original_task_sha256') or canonical_hash(record['original_task']),
        'original_assessment_and_reason_retained_in_audit':True}


def tool_constraints(feedback):
    check=feedback.get('input_check') or {};test=feedback.get('test') or {}
    missing=[]
    for row in check.get('unavailable_variables',[]):
        var=row['variable'];item={'quantity':var['quantity'],'unit':var['unit'],'registered_source_missing':True}
        if item not in missing:missing.append(item)
    return {'input_status':check.get('status'),'missing_measurements':missing,
        'input_failure_counts':deepcopy(check.get('failures',{})),
        'jointly_available':check.get('jointly_available'),
        'test':{k:deepcopy(test[k]) for k in ('status','valid_observations','target_event_groups','coverage',
            'mean_group_improvement','eligible_for_independent_validation','result_sha256') if k in test},
        'predictive_effectiveness_verified':False}


def history_view(full_history,past_calls,catalogue,attempts):
    # Keep the previously tested compactor intact for exact historical replay.
    from .team_context_comparison import compact_history
    result=compact_history(full_history,past_calls,catalogue)
    result['kind']='structured_research_history_v1'
    result['current_task_authority']='Current role input: pending_task, original_task or active_task; history is never an assignment.'
    result['past_tool_constraints']=[{'attempt':a['attempt'],'feedback_sha256':a['feedback']['feedback_sha256'],
        **tool_constraints(a['feedback'])} for a in attempts]
    result['attempted_experiments']=[]
    from .team_research_loop import relation_fingerprint
    for a in attempts:
        rel=a['feedback'].get('relation')
        if rel is not None:
            result['attempted_experiments'].append({'attempt':a['attempt'],
                'fingerprint':relation_fingerprint(rel),'status':a['workflow']['status']})
    result['history_view_sha256']=canonical_hash(result);return result


def learning_view(feedback):
    """Keep current facts, remove raw replies and superseded task narratives.

    feedback_sha256 explicitly refers to the full archived record and is copied
    by the existing reflection validator; this is a projection, not a rehash.
    """
    from .team_context_comparison import error_code
    result=deepcopy(feedback)
    result['feedback_hash_refers_to_full_archived_record']=True
    result['failure_details']=[{'role':r['role'],'error_code':error_code(r['error']),
        'error_sha256':canonical_hash(r['error']),'raw_output_excerpt_sha256':canonical_hash(r['raw_output_excerpt'])}
        for r in feedback.get('failure_details',[])]
    if feedback.get('task_negotiation'):
        result['task_negotiation']=negotiation_view(feedback['task_negotiation'])
    # A query predates task negotiation. Keep measured coverage, not the old task
    # narrative in its model-written query reason.
    discovery=feedback.get('data_catalogue')
    if discovery:
        tool=discovery.get('result') or {}
        result['data_catalogue']={'status':discovery['status'],'discovery_sha256':discovery['discovery_sha256'],
            'result_sha256_reference':tool.get('result_sha256'),
            **{k:deepcopy(tool[k]) for k in ('sources','markets','time_semantics','limitations') if k in tool}}
    result['learning_view_sha256']=canonical_hash(result);return result
