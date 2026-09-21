"""Executable task handoffs for the bounded frozen-team research loop.

Enforces contract routing and duplicate repair, not free-form semantic truth.
No strategy classes or predictive formulas are supplied by the orchestrator.
"""
from copy import deepcopy

from .schema import fields
from .synthetic_sft import canonical_hash
from .team_learning import require
from .team_executable_methods import bounded_text
from .team_research_loop import relation_fingerprint, feedback_for, history_view, summarize
from .team_staged_methods import run_staged_workflow
from . import team_variable_contracts as typed

TASK = '''You coordinate the next offline TRAINING investigation. Return ONLY a JSON object with exactly
market_ids, objective, change. market_ids is 1..4 DISTINCT STRING IDs from the supplied catalogue index,
for example ["123", "456"] (example IDs are not real selections). Never output numbers as IDs or unknown IDs.
objective is a concrete research question under 400 characters. change is new_contracts or revise_current.
For new_contracts include at least one market not used in the previous investigated relation/task.
For revise_current use exactly the previous task's market IDs and explain the substantive revision in objective.
Choose a relationship yourself; no strategy category is prescribed. Select only contracts that the research member
must actually use. Full historical rules are retrieved for these exact IDs next. Previous failure details are supplied.
The task controls the next investigation, not a trade. Do not claim any relation profitable or verified.'''
REFLECT = '''You are the team's learning member. Convert this attempt's concrete feedback into the NEXT executable
task, not just a suggestion. Return ONLY JSON with exactly feedback_sha256, next_focus, data_requests, next_task.
Copy feedback_sha256 exactly. next_focus is a concise explanation under 400 characters.
next_task is an object with exactly market_ids, objective, change. Use 1..4 distinct STRING market IDs from the
supplied full catalogue index. objective is a concrete next research question under 400 characters.
change is new_contracts or revise_current. new_contracts must include at least one different contract from the
previous investigated relation/task; revise_current must keep exactly the current task's contract IDs.
The next research member will be REQUIRED to use every selected ID; choose only contracts actually needed.
Do not name unavailable contracts as if they were available. Do not retry identical failed measurements merely by
changing wording. Correct a specific measurement/source mistake, or investigate another relationship.
data_requests is a list of at most 3 objects, each exactly quantity, unit, why, historical_timestamp_requirement,
all nonempty strings under 240 characters. Use [] if no worthwhile acquisition is identified. Requests are unverified;
missing data remain missing. Do not request existing error logs: the concrete failures are already supplied.
Reading feedback and producing valid JSON do not establish learning effectiveness, forecasting edge or net profit.'''
TASK_V2 = '''You coordinate the next offline TRAINING investigation. Return ONLY JSON with exactly market_ids and objective.
market_ids is 1..4 distinct STRING IDs from the supplied catalogue index. Use exact quoted IDs, never numeric IDs.
objective is a concrete research question under 400 characters. Choose a relationship yourself, without prescribed
strategy categories. Select only contracts actually needed: the next researcher must use ALL and ONLY these IDs.
You may revise the current experiment or investigate different contracts, based on concrete previous failures.
Do not merely conjoin unrelated event questions. Explain the tentative predictive link to test. Do not claim profit.
The program derives whether the selected IDs change; you do not need to classify this routing metadata.'''
REFLECT_V2 = '''You are the team's learning member. Convert concrete feedback into the NEXT executable investigation.
Return ONLY JSON with exactly feedback_sha256, next_focus, data_requests, next_task. Copy feedback_sha256 exactly.
next_focus is a concise explanation under 400 characters. next_task has exactly market_ids and objective.
Use 1..4 distinct STRING market IDs from the full supplied index. objective is a concrete next research question under
400 characters. The next researcher must use ALL and ONLY these IDs. You may revise the current experiment or select
other contracts; the program derives this distinction. Correct the specific failed measurement or data assumption,
or investigate a different predictive link. Do not merely conjoin unrelated event questions or repeat failed wording.
Unknown contracts cannot be selected. Do not claim that reading feedback proves predictive or trading effectiveness.
data_requests is at most 3 objects, each exactly quantity, unit, why, historical_timestamp_requirement. All fields are
nonempty strings under 240 characters. Use [] when no worthwhile acquisition is identified. These requests remain
unverified; no external observations are silently replaced with prices. Concrete error logs are already supplied.'''

RESEARCH_TASK = '''\nYou are executing active_task, an immutable handoff from the preceding team member.
Read its objective and full rules. If proposing a relation, the union of target and non-null peer IDs in bindings
MUST equal active_task.market_ids: use EVERY assigned contract, and no other contract. Do not silently switch tasks.
If no faithful hypothesis fits these rules, abstain with relation=null and explain why. Prior rejected candidate
specifications cannot be repeated. A duplicate receives one explicit repair opportunity within the role's existing
repair budget. Repairs must change the actual experiment, not only variable names or prose.'''


def validate_task(value, catalogue, previous=None):
    fields(value, {'market_ids', 'objective', 'change'}, 'research task')
    bounded_text(value['objective'], 400)
    ids = value['market_ids']
    require(isinstance(ids, list) and 1 <= len(ids) <= 4
            and all(isinstance(k, str) for k in ids) and len(set(ids)) == len(ids)
            and set(ids) <= set(catalogue),
            'market_ids must be 1..4 distinct JSON STRINGS from: '+', '.join(sorted(catalogue)))
    require(value['change'] in ('new_contracts', 'revise_current'), 'invalid task change')
    if previous is None:
        require(value['change'] == 'new_contracts', 'initial task must use new_contracts')
    elif value['change'] == 'new_contracts':
        require(set(ids)-set(previous['market_ids']), 'new_contracts must add at least one different market ID')
    else:
        require(set(ids) == set(previous['market_ids']), 'revise_current must retain exactly the assigned market IDs')
    return deepcopy(value)


def validate_derived_task(value, catalogue, previous=None):
    # Selection is model-owned; change classification is deterministic routing
    # metadata. An optional legacy tag is archived raw but cannot veto valid IDs.
    require(isinstance(value, dict), 'research task must be an object')
    require(set(value) in ({'market_ids', 'objective'}, {'market_ids', 'objective', 'change'}),
            'derived task requires market_ids and objective only')
    ids = value['market_ids']
    if 'change' in value:
        require(value['change'] in ('new_contracts', 'revise_current'), 'unknown legacy task tag')
    # First use the original validator to check text and IDs, without accepting
    # the model's redundant change label as an authority over those exact IDs.
    checked = validate_task({'market_ids': ids, 'objective': value['objective'], 'change': 'new_contracts'}, catalogue)
    checked['change'] = ('revise_current' if previous is not None and set(ids) == set(previous['market_ids'])
                         else 'new_contracts')
    return checked


def validate_reflection(value, feedback, catalogue, previous, *, derive_task_change=False):
    from .team_research_loop import validate_reflection as validate_base
    fields(value, {'feedback_sha256', 'next_focus', 'data_requests', 'next_task'}, 'task reflection')
    validate_base({k: value[k] for k in ('feedback_sha256', 'next_focus', 'data_requests')}, feedback)
    result = deepcopy(value)
    result['next_task'] = (validate_derived_task if derive_task_change else validate_task)(value['next_task'], catalogue, previous)
    return result


class TaskRunner:
    def __init__(self, runner, task, seen, *, multi_input=False):
        self.runner = runner; self.task = deepcopy(task); self.seen = seen; self.multi_input = multi_input
        self.duplicates = []; self.last_relation = None; self.semantic_revision_used = False
    def structured(self, role, instruction, context, upstream, validator):
        if role != 'relation_researcher':
            return self.runner.structured(role, instruction, context, upstream, validator)
        revision = context.get('semantic_revision_of')
        authorized_revision = revision is not None and self.last_relation is not None and revision == canonical_hash(self.last_relation) and not self.semantic_revision_used
        if revision is not None:
            require(authorized_revision, 'semantic revision must refer to the current accepted relation, once only')
            self.semantic_revision_used = True
        def checked(value):
            result = validator(value); relation = result['relation']
            if relation is not None:
                if self.multi_input:
                    from .team_multi_input import used_markets
                    used = used_markets(relation)
                else:
                    used = {k for b in relation['bindings'] for k in b.values() if k is not None}
                require(used == set(self.task['market_ids']),
                        'task handoff mismatch: bindings must use ALL and ONLY assigned market_ids '+str(self.task['market_ids']))
                key = relation_fingerprint(relation)
                if key in self.seen and not (authorized_revision and key == relation_fingerprint(self.last_relation)):
                    self.duplicates.append(deepcopy(result))
                    raise ValueError('duplicate_relation: already attempted these bindings, target, horizon and measurements. '
                                     'Change the actual measured experiment to address its feedback, or abstain. '
                                     'Renaming variables or rewording the hypothesis is not a revision.')
            return result
        task_instruction = ('\nExecute active_task. The union of variable market_ids and prediction targets must equal its market_ids. '
                            'Abstain when no faithful relation is possible; do not repeat a rejected experiment under new variable names.'
                            if self.multi_input else RESEARCH_TASK)
        value = self.runner.structured(role, instruction+task_instruction,
            {**deepcopy(context), 'active_task': deepcopy(self.task), 'active_task_sha256': canonical_hash(self.task)}, upstream, checked)
        if value['relation'] is not None:
            self.seen.add(relation_fingerprint(value['relation']))
            self.last_relation = deepcopy(value['relation'])
        return value


def failure_details(calls):
    # The raw calls remain intact in the archive; expose bounded concrete errors
    # so "learn from failure" is more than an opaque final status string.
    return [{'role': c['request']['agent'], 'error': c['error'], 'raw_output_excerpt': (c['output'] or '')[:1800]}
            for c in calls if c['error']][-4:]


def run_loop(runner, context, catalogue, feed, labels, protocol, max_attempts, freeze, save_attempt, *, derive_task_change=False, multi_input=False, named_prediction_kind=False, semantic_gate=False, task_negotiation=False, semantic_advisory=False, data_discovery=False, structured_history=False, grounded_revision=False):
    require(type(max_attempts) is int and 1 <= max_attempts <= 10, 'invalid attempt budget')
    index = [{'market_id': k, 'title': r['question'].split(', description:', 1)[0], 'initialized_at': r['initialized_at']}
             for k, r in sorted(catalogue.items())]
    task_validator = validate_derived_task if derive_task_change else validate_task
    task_instruction = TASK_V2 if derive_task_change else TASK
    reflection_instruction = REFLECT_V2 if derive_task_change else REFLECT
    if semantic_advisory:
        require(semantic_gate and multi_input, 'advisory mode requires multi-input review')
        from .team_review_boundary import LEARNING
        reflection_instruction += LEARNING
    if data_discovery:
        require(task_negotiation and semantic_advisory, 'data discovery requires negotiated advisory workflow')
        from .team_data_catalogue import GUIDANCE
        reflection_instruction += GUIDANCE
    if structured_history: require(data_discovery, 'structured history requires the full data-catalogue workflow')
    if grounded_revision:
        require(structured_history and derive_task_change, 'grounded revision requires structured derived-task workflow')
        from .team_grounded_revision import LEARNING as GROUNDED_LEARNING
        reflection_instruction += GROUNDED_LEARNING
    attempts = []; seen = set(); pending = None; previous = None
    context = deepcopy(context)
    reference = context.pop('task_negotiation_reference', None) if task_negotiation else None
    if reference is not None: pending = deepcopy(reference['task'])
    if grounded_revision and reference is not None and reference['feedback'].get('relation'):
        seen.add(relation_fingerprint(reference['feedback']['relation']))
    for number in range(1, max_attempts+1):
        start = len(runner.calls); history = history_view(attempts); task = None; selection = None
        origin = 'preceding_reflection' if pending is not None else 'coordinator'
        if reference is not None:
            history['prior_training_failure'] = deepcopy(reference)
            if number == 1: origin = 'archived_training_failure'
        full_history = deepcopy(history)
        if structured_history:
            from .team_structured_memory import history_view as compact_view
            history = compact_view(full_history,runner.calls[:start],catalogue,attempts)
        if grounded_revision:
            from .team_grounded_revision import extend_history
            history = extend_history(history, full_history)
        duplicates = []; negotiation = None; discovery = None; data_catalogue = None
        current_task_catalogue = None
        try:
            if data_discovery:
                from .team_data_catalogue import discover
                discovery = discover(runner,catalogue,context['variable_sources'],feed,protocol,index,history,pending)
                if discovery['error'] is not None: raise ValueError(discovery['error'])
                data_catalogue = discovery['result']
            if pending is not None:
                task = task_validator(pending, catalogue, previous)
            else:
                task = runner.structured('research_coordinator', task_instruction,
                    {'phase': context['phase'], 'index': index, 'attempt': number, 'budget': max_attempts,
                     'previous_task': previous, **({'data_catalogue':data_catalogue} if data_discovery else {})}, history, lambda v: task_validator(v, catalogue, previous))
            pending = None
            effective_task = task
            if task_negotiation:
                from .team_task_negotiation import negotiate
                negotiation = negotiate(runner, task, catalogue, index, history, data_catalogue=data_catalogue)
                effective_task = negotiation['effective_task']
            if effective_task is None:
                workflow = {'status': negotiation['status'], 'stop_reason': negotiation['error'] or 'Coordinator abstained.',
                            'relation': None, 'data_check': None, 'result': None}
            else:
                task = effective_task
                selection = {'market_ids': task['market_ids'], 'reason': task['objective']}
                selected = {k: catalogue[k] for k in task['market_ids']}
                if grounded_revision:
                    from .team_grounded_revision import current_catalogue
                    current_task_catalogue = current_catalogue(task, catalogue, context['variable_sources'], feed, protocol)
                    data_catalogue = current_task_catalogue['result']
                local = {**deepcopy(context), 'catalogue': list(selected.values()), 'training_history': history,
                         'active_task': task, 'attempt': number, 'max_attempts': max_attempts}
                if negotiation is not None:
                    from .team_task_negotiation import feedback_view
                    if structured_history:
                        from .team_structured_memory import negotiation_view
                        local['task_negotiation'] = negotiation_view(negotiation)
                    else: local['task_negotiation'] = feedback_view(negotiation)
                if data_discovery: local['data_catalogue'] = deepcopy(data_catalogue)
                bound = TaskRunner(runner, task, seen, multi_input=multi_input)
                if multi_input:
                    from .team_multi_input import run_workflow
                    workflow = run_workflow(bound, local, selected, feed, labels, protocol, lambda p: freeze(number, p), named_kind=named_prediction_kind,
                        semantic_gate=semantic_gate, probe_qualified=context.get('semantic_probe_qualified', False),
                        semantic_advisory=semantic_advisory)
                else:
                    workflow = run_staged_workflow(bound, local, selected, feed, labels, protocol,
                                                   lambda p: freeze(number, p), variable_contract=typed)
                duplicates = bound.duplicates
                if workflow['status'] == 'relation_failed' and 'duplicate_relation:' in (workflow.get('stop_reason') or ''):
                    workflow.update(status='duplicate_relation', relation=duplicates[-1])
        except (ValueError, TypeError, KeyError) as exc:
            workflow = {'status': 'selection_failed', 'stop_reason': str(exc), 'relation': None,
                        'data_check': None, 'result': None}
        if discovery is not None and discovery['error'] is not None:
            workflow['status'] = 'data_catalogue_failed'
        feedback = feedback_for(workflow)
        feedback.pop('feedback_sha256')
        feedback['failure_details'] = failure_details(runner.calls[start:])
        feedback['active_task'] = deepcopy(task)
        if data_discovery: feedback['data_catalogue'] = deepcopy(discovery)
        if grounded_revision: feedback['current_task_catalogue'] = deepcopy(current_task_catalogue)
        if negotiation is not None:
            from .team_task_negotiation import feedback_view
            feedback['task_negotiation'] = feedback_view(negotiation)
        if semantic_gate:
            from .team_semantic_consistency import feedback_summary
            feedback['semantic_consistency'] = feedback_summary(workflow.get('semantic_reviews', []))
            feedback['data_requests_are_unverified'] = True
            if semantic_advisory:
                from .team_review_boundary import POLICY
                feedback['review_boundary'] = deepcopy(POLICY)
                feedback['semantic_consistency_is_unverified_advice'] = True
        feedback['feedback_sha256'] = canonical_hash(feedback)
        learning_feedback = feedback
        if structured_history:
            from .team_structured_memory import learning_view
            learning_feedback = learning_view(feedback)
        if grounded_revision:
            from .team_grounded_revision import learning_view
            learning_feedback = learning_view(feedback)
        reflection = None; error = None
        if task is not None: previous = deepcopy(task)
        try:
            if grounded_revision:
                from .team_grounded_revision import validate_reflection as validate_grounded
                reflection_validator = lambda v: validate_grounded(v, feedback, catalogue, previous)
            else:
                reflection_validator = lambda v: validate_reflection(v, feedback, catalogue, previous, derive_task_change=derive_task_change)
            reflection = runner.structured('research_learning_member', reflection_instruction,
                {'phase': 'after_training_attempt', 'attempt': number, 'index': index, 'previous_task': previous},
                {'feedback': learning_feedback}, reflection_validator)
            pending = deepcopy(reflection['next_task'])
        except (ValueError, TypeError, KeyError) as exc: error = str(exc)
        record = {'attempt': number, 'selection': selection, 'workflow': workflow, 'feedback': feedback,
            'reflection': reflection, 'reflection_error': error, 'active_task': task, 'task_origin': origin,
            'active_task_sha256': canonical_hash(task) if task is not None else None,
            'duplicate_repairs': duplicates, 'call_start': start, 'call_end': len(runner.calls),
            'previous_attempt_sha256': attempts[-1]['attempt_sha256'] if attempts else None}
        if task_negotiation: record['task_negotiation'] = negotiation
        if data_discovery: record['data_catalogue'] = discovery
        if grounded_revision: record['current_task_catalogue'] = current_task_catalogue
        if structured_history:
            record.update(history_archive=full_history,history_view=history,learning_feedback_view=learning_feedback)
        record['attempt_sha256'] = canonical_hash(record); save_attempt(record); attempts.append(record)
        print('Task handoff attempt', number, '/', max_attempts, feedback['status'], 'origin', origin, flush=True)
    return attempts


def summary(attempts):
    result = {**summarize(attempts), 'tasks_received_from_reflection': sum(a['task_origin'] == 'preceding_reflection' for a in attempts),
        'valid_relations_matching_active_task': sum(bool(a['workflow'].get('relation') and a['workflow']['relation']['relation'])
            and a['workflow']['status'] != 'duplicate_relation' for a in attempts),
        'duplicate_outputs_rejected': sum(len(a['duplicate_repairs']) for a in attempts),
        'last_next_task_not_executed_due_to_budget': (attempts[-1]['reflection'] or {}).get('next_task') if attempts else None}

    if any('task_negotiation' in a for a in attempts):
        negotiations = [a['task_negotiation'] for a in attempts if a.get('task_negotiation')]
        result['task_negotiation'] = {
            'attempts': len(negotiations),
            'accepted_proposals': sum((n['decision'] or {}).get('decision') == 'accept_proposal' for n in negotiations),
            'effective_tasks_changed': sum(n['effective_task'] is not None and bool(n['added_market_ids'] or n['removed_market_ids'] or n['objective_changed']) for n in negotiations),
            'abstentions': sum(n['status'] == 'task_negotiation_abstained' for n in negotiations),
            'failures': sum(n['status'] == 'task_negotiation_failed' for n in negotiations),
            'effectiveness_verified': False}
    if any('data_catalogue' in a for a in attempts):
        result['data_catalogue'] = {
            'queries_completed': sum((a.get('data_catalogue') or {}).get('status')=='data_catalogue_read' for a in attempts),
            'queries_failed': sum((a.get('data_catalogue') or {}).get('status')=='data_catalogue_failed' for a in attempts),
            'new_sources_added': 0, 'values_or_labels_exposed': False}
    if any('history_view' in a for a in attempts):
        result['structured_history'] = {'attempts_with_audited_projection':sum('history_view' in a for a in attempts),
            'full_records_retained':True, 'parameter_learning':False}
    if any('current_task_catalogue' in a for a in attempts):
        result['grounded_revision'] = {
            'effective_task_catalogue_refreshes':sum(a.get('current_task_catalogue') is not None for a in attempts),
            'reflections_with_valid_fact_references':sum('diagnosis' in (a['reflection'] or {}) for a in attempts),
            'repair_effectiveness_verified':False}
    return result
