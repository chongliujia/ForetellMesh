"""Bounded, explicit renegotiation of a research task before relation design.

The program checks routing, never decides that different topics cannot relate.
No source/label access, silent contract deletion, or unbounded retry loop.
"""
from copy import deepcopy

from .schema import fields
from .synthetic_sft import canonical_hash
from .team_learning import require
from .team_executable_methods import bounded_text

ASSESS = '''You are the research member assessing an assigned TRAINING task before designing a relation.
Read every supplied historical contract rule and the prior failure feedback. Return ONLY JSON with exactly
contract_assessments, proposed_task, reason. reason is under 400 characters.
contract_assessments contains exactly one object for EVERY original task contract: market_id, disposition, reason.
disposition is retain or release; each reason under 400 characters states its role in the proposed test or why you
cannot state a faithful role. Do not invent a connection merely to include every ID. Different topics ALONE do not
justify release: explicitly testable cross-topic hypotheses are allowed, with no assumed predictive edge.
proposed_task is either null (abstain), or exactly market_ids and objective: 1..4 distinct STRING IDs from the index,
objective under 400 characters explaining the tentative link to test. You may keep the task, narrow it, replace
contracts, or revise its objective. Retain precisely the original IDs included in proposed_task; release the others.
If proposed_task is null, release all IDs with reasons. New contracts have only index titles here; their full rules
will be retrieved if the coordinator accepts. Missing data is not proof a relationship is false. Do not invent
sources, outcomes, financial-market identities or profits. Your proposal cannot change the task without the
coordinator's decision. An honest abstention is preferable to inventing a relationship.'''
DECIDE = '''You coordinate one bounded task negotiation. Review the research member's proposal, the original task,
full historical rules supplied, and concrete previous failure feedback. Return ONLY JSON with exactly decision and
reason. decision is accept_proposal, retain_original, or abstain; reason under 400 characters explains the choice.
Accept a clearer testable proposed task when justified, retain the original only if the concerns are addressed,
or abstain. Do not force a contract into an invented relationship just to fill an assigned list. Cross-topic tests
are allowed; topic distance alone is not a reason for rejection. Acceptance is permission to research, not evidence
of semantic truth, source availability, predictive validity or profit. There is no further negotiation this attempt.'''


def validate_assessment(value, task, catalogue):
    from .team_research_handoff import validate_derived_task
    fields(value, {'contract_assessments','proposed_task','reason'}, 'task assessment')
    bounded_text(value['reason'],400)
    proposed=value['proposed_task']
    if proposed is not None:
        fields(proposed,{'market_ids','objective'},'proposed task')
        validate_derived_task(proposed,catalogue,task)
    selected=set(proposed['market_ids']) if proposed else set()
    rows=value['contract_assessments'];seen=set()
    require(isinstance(rows,list) and len(rows)==len(task['market_ids']), 'assess every original contract exactly once')
    for row in rows:
        fields(row,{'market_id','disposition','reason'},'contract assessment')
        mid=row['market_id'];bounded_text(row['reason'],400)
        require(isinstance(mid,str) and mid in task['market_ids'] and mid not in seen,'unknown/duplicate assessed contract')
        seen.add(mid)
        require(row['disposition']==('retain' if mid in selected else 'release'),'disposition must match proposed task IDs')
    return deepcopy(value)


def validate_decision(value, assessment):
    fields(value,{'decision','reason'},'negotiation decision');bounded_text(value['reason'],400)
    require(value['decision'] in ('accept_proposal','retain_original','abstain'),'unknown negotiation decision')
    require(value['decision']!='accept_proposal' or assessment['proposed_task'] is not None,'cannot accept absent proposal')
    return deepcopy(value)


def negotiate(runner, task, catalogue, index, history, *, data_catalogue=None):
    from .team_research_handoff import validate_derived_task
    # Explicit projection: no source registry, private labels, price coverage or probe controls.
    context={'phase':'pre_research_task_negotiation','original_task':deepcopy(task),
        'catalogue':[deepcopy(catalogue[k]) for k in task['market_ids']], 'index':deepcopy(index)}
    extra=''
    if data_catalogue is not None:
        from .team_data_catalogue import GUIDANCE
        context['data_catalogue']=deepcopy(data_catalogue);extra=GUIDANCE
    record={'kind':'bounded_task_negotiation_v1','original_task':deepcopy(task),
        'original_task_sha256':canonical_hash(task),'assessment':None,'decision':None,
        'effective_task':None,'status':'task_negotiation_failed','error':None,
        'added_market_ids':[],'removed_market_ids':[],'objective_changed':False}
    try:
        assessment=runner.structured('research_task_assessor',ASSESS+extra,context,{'training_history':history},
            lambda v:validate_assessment(v,task,catalogue))
        record['assessment']=assessment
        decision=runner.structured('task_negotiation_coordinator',DECIDE+extra,context,
            {'assessment':assessment,'training_history':history},lambda v:validate_decision(v,assessment))
        record['decision']=decision
        if decision['decision']=='abstain':record['status']='task_negotiation_abstained'
        else:
            effective=deepcopy(task) if decision['decision']=='retain_original' else validate_derived_task(assessment['proposed_task'],catalogue,task)
            record.update(effective_task=effective,status='task_negotiated')
            record['added_market_ids']=sorted(set(effective['market_ids'])-set(task['market_ids']))
            record['removed_market_ids']=sorted(set(task['market_ids'])-set(effective['market_ids']))
            record['objective_changed']=effective['objective']!=task['objective']
    except (ValueError,TypeError,KeyError) as exc:record['error']=str(exc)
    record['negotiation_sha256']=canonical_hash(record);return record


def feedback_view(record):
    return {k:deepcopy(record[k]) for k in ('status','error','assessment','decision','original_task',
        'effective_task','added_market_ids','removed_market_ids','objective_changed','negotiation_sha256')}


def load_reference(config, catalogue):
    """Replay one explicitly registered failed training task, never its outcomes."""
    from pathlib import Path
    from .data import strict_json,sha256_file
    from .team_research_handoff import validate_derived_task
    root=Path(config['negotiation_reference_run'])
    require(sha256_file(root/'report.json')==config['negotiation_reference_report_sha256'],'negotiation reference changed')
    report=strict_json((root/'report.json').read_text())
    require(report['status']=='completed' and report['development_opened'] is False and report['final_test_opened'] is False,
        'negotiation reference must be training only')
    name='attempt_01.json'
    require(sha256_file(root/name)==report['artifact_hashes'][name],'reference attempt changed')
    attempt=strict_json((root/name).read_text());feedback=attempt['feedback']
    require(feedback['status']=='relation_failed' and 'task handoff mismatch:' in feedback['stop_reason'],
        'reference must be a failed binding task')
    require(feedback['relation'] is None and not feedback['input_check'] and not feedback['test'], 'reference must precede data or scores')
    validate_derived_task(attempt['active_task'],catalogue)
    return {'source_report_sha256':config['negotiation_reference_report_sha256'],
        'source_attempt_sha256':report['artifact_hashes'][name], 'task':deepcopy(attempt['active_task']),
        'feedback':deepcopy(feedback),'already_exposed_training_experience':True}
