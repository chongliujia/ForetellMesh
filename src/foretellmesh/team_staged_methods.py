"""Small team handoffs: relation, data request, coverage, calculation, scoring.

Each model owns one bounded decision. Tools own timestamps, bindings and scores.
Natural-language meaning is still an unverified hypothesis, never a proof.
"""
from collections import Counter
from copy import deepcopy
from datetime import timedelta

from .schema import fields, timestamp, iso
from .synthetic_sft import canonical_hash
from .team_learning import require
from .team_executable_methods import (bounded_text, compile_expression, fresh_price,
    validate_protocol, validate_plan, execute_plan, REVIEWER, validate_review)


RELATION = '''You are the research member. Propose ONE tentative relationship worth testing from the supplied
contract rules and prior candidate. Choose the contracts yourself; do not assume adjacent catalogue entries are related.
Only design the relation now; a data member will check observations and a quant member will later write the formula.
Return one JSON object with exactly relation and reason. relation is null if no worthwhile hypothesis; explain in reason.
Otherwise reason=null and relation is an object with exactly hypothesis, forecast_target, horizon_days, bindings.
hypothesis is a concrete tentative relationship, not a profitability assertion, under 600 characters.
forecast_target is future_yes_price or resolves_yes. horizon_days is integer 1..14 for future_yes_price, null for resolves_yes.
bindings is 1..4 objects, each exactly target and peer. Use supplied market IDs; peer may be null.
Targets must be distinct; a peer differs from its target. All bindings use a peer or all use null.
Contract Yes prices are different from external quantities such as box office revenue, sports results or tweet counts.
You may propose hypotheses needing external evidence, but missing evidence will stop execution until it is available.
Rules alone do not establish statistical prediction. Same-event contracts do not count as independent event groups.
No formula, threshold or invented evidence is needed in this step.'''

DATA = '''You are the data member. Translate the supplied hypothesis into the minimal observable inputs it actually
needs. Do not change the relation, contracts, forecast target or horizon. Return ONLY JSON with exactly inputs and missing_data.
inputs is a list of up to 8 objects, each with exactly role, quantity, lag_days. role is target or peer.
lag_days is integer 0..30; 0 means observation time. All roles refer to the supplied contract bindings.
Use quantity=yes_price ONLY for historical Yes contract trade prices, not external asset prices or event measurements.
Any other needed quantity should have its own short descriptive name; the tool will explicitly report it unsupported.
missing_data is a list of up to 4 short strings for unmet evidence needs; empty is allowed.
Do not substitute market prices for absent movie revenue, tweets or other required event observations.
For a cross-contract hypothesis, request at least one peer input. There is no peer input for a target-only hypothesis.
Request only inputs the later formula must actually use. Do not write a formula here. Do not claim data are present:
the next tool checks actual joint availability and timestamps. If no inputs can be specified, explain in missing_data.'''

QUANT = '''You are the quant member. Write ONLY the computation for the fixed relation and verified data request.
Do not change the hypothesis, contract bindings, forecast target, horizon or declared inputs.
Return one JSON object with exactly expression, min_observations, min_improvement, reason.
expression is a string under 400 characters predicting the stated Yes price or probability in [0,1].
Read each declared input as p('target',N) or p('peer',N), where N is its literal lag_days.
Use ALL and ONLY the declared role/lag inputs. Available arithmetic: + - * /, parentheses, abs(x), min(x,y), max(x,y).
Constants may parameterize your method; the program does not choose a trading rule for you.
The baseline is current target Yes price. Evaluation measures future-price squared error or event Brier, NOT net profit.
min_observations is integer 8..100 for future_yes_price, 3..100 for resolves_yes.
min_improvement is numeric >0 and <=1: required reduction in event-group-averaged error versus baseline.
reason=null when an expression is specified. If no faithful computation is possible, expression=null,
min_observations=null, min_improvement=null, and explain briefly in reason.
Missing inputs are never guessed and out-of-range predictions fail; use bounds explicitly only if your method requires them.
This training diagnostic is not an independent validation; do not call a hypothesis proven.'''


def validate_relation(value, catalogue):
    fields(value, {'relation', 'reason'}, 'relation design')
    relation = value['relation']
    if relation is None:
        bounded_text(value['reason']); return deepcopy(value)
    require(value['reason'] is None, 'relation and abstention conflict')
    fields(relation, {'hypothesis', 'forecast_target', 'horizon_days', 'bindings'}, 'relation')
    bounded_text(relation['hypothesis'])
    require(relation['forecast_target'] in ('future_yes_price', 'resolves_yes'), 'invalid forecast target')
    horizon = relation['horizon_days']
    require((type(horizon) is int and 1 <= horizon <= 14) if relation['forecast_target'] == 'future_yes_price'
            else horizon is None, 'target/horizon mismatch')
    bindings = relation['bindings']
    require(isinstance(bindings, list) and 1 <= len(bindings) <= 4, 'unbounded relation bindings')
    seen = set(); peer_modes = set()
    for b in bindings:
        fields(b, {'target', 'peer'}, 'relation binding'); target, peer = b['target'], b['peer']
        require(isinstance(target, str) and target in catalogue and target not in seen, 'unknown/repeated relation target')
        require(peer is None or isinstance(peer, str) and peer in catalogue and peer != target, 'unknown/self relation peer')
        seen.add(target); peer_modes.add(peer is not None)
    require(len(peer_modes) == 1, 'inconsistent peer mode across bindings')
    return deepcopy(value)


def validate_data(value, relation):
    fields(value, {'inputs', 'missing_data'}, 'data requirements')
    require(isinstance(value['inputs'], list) and len(value['inputs']) <= 8, 'unbounded data inputs')
    require(isinstance(value['missing_data'], list) and len(value['missing_data']) <= 4, 'unbounded missing data')
    for text in value['missing_data']: bounded_text(text)
    has_peer = relation['bindings'][0]['peer'] is not None; seen = set()
    for item in value['inputs']:
        fields(item, {'role', 'quantity', 'lag_days'}, 'data input')
        require(item['role'] in ('target', 'peer') and (has_peer or item['role'] != 'peer'), 'unbound data role')
        bounded_text(item['quantity'], 80)
        require(type(item['lag_days']) is int and 0 <= item['lag_days'] <= 30, 'future/unbounded data lag')
        key = (item['role'], item['quantity'], item['lag_days'])
        require(key not in seen, 'duplicate requested input'); seen.add(key)
    require(value['inputs'] or value['missing_data'], 'empty data request without explanation')
    if has_peer and not value['missing_data']:
        require(any(x['role'] == 'peer' for x in value['inputs']), 'cross-contract relation lacks peer data')
    return deepcopy(value)


def check_data(request, relation, catalogue, feed, protocol):
    """Input-only check: no labels, future target prices or scores are accepted."""
    validate_protocol(protocol); validate_data(request, relation)
    unsupported = [i for i in request['inputs'] if i['quantity'] != 'yes_price']
    rows = []; summaries = []
    if not unsupported and not request['missing_data']:
        refs = {(i['role'], i['lag_days']) for i in request['inputs']} | {('target', 0)}
        cutoff = timestamp(protocol['as_of'], 'cutoff')
        for binding in relation['bindings']:
            init = timestamp(catalogue[binding['target']]['initialized_at'], 'initialization')
            start = init.replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=1)
            attempted = 0; available = 0
            horizon = relation['horizon_days']
            for day in range(0, protocol['max_days_per_target'], horizon or 1):
                at = start+timedelta(days=day)
                if at+timedelta(days=horizon or 0) > cutoff: break
                attempted += 1; evidence = []; errors = []
                for role, lag in sorted(refs):
                    mid = binding[role]; point = at-timedelta(days=lag)
                    if point < timestamp(catalogue[mid]['initialized_at'], 'initialization'):
                        errors.append(role+':before_initialization'); continue
                    quote, error = fresh_price(feed, mid, point, protocol['max_quote_age_seconds'])
                    if error: errors.append(role+':'+error)
                    else: evidence.append({**quote, 'role': role, 'lag_days': lag})
                if not errors: available += 1
                rows.append({'binding': deepcopy(binding), 'observation_time': iso(at), 'evidence': evidence, 'errors': errors})
                if horizon is None: break
            summaries.append({'binding': deepcopy(binding), 'observations_attempted': attempted,
                              'jointly_available': available})
    available = sum(s['jointly_available'] for s in summaries)
    result = {'kind': 'method_input_check_v1', 'request_sha256': canonical_hash(request),
              'relation_sha256': canonical_hash(relation), 'protocol_sha256': canonical_hash(protocol),
              'status': 'unsupported_data' if unsupported or request['missing_data'] else
                        'no_joint_observations' if not available else 'ready_for_calculation',
              'unsupported_inputs': unsupported, 'declared_missing_data': deepcopy(request['missing_data']),
              'bindings': summaries, 'jointly_available': available,
              'failures': dict(Counter(error for r in rows for error in r['errors'])), 'observations': rows,
              'interpretation': 'Input-only availability; does not check future targets, settlement boundaries or predictive quality.'}
    result['check_sha256'] = canonical_hash(result); return result


def validate_calculation(value, relation, request, catalogue):
    fields(value, {'expression', 'min_observations', 'min_improvement', 'reason'}, 'calculation')
    if value['expression'] is None:
        bounded_text(value['reason'])
        require(value['min_observations'] is None and value['min_improvement'] is None, 'abstention has thresholds')
        return deepcopy(value)
    require(value['reason'] is None, 'calculation and abstention conflict')
    _, refs = compile_expression(value['expression'])
    require(all(i['quantity'] == 'yes_price' for i in request['inputs']) and not request['missing_data'], 'unavailable formula inputs')
    declared = {(i['role'], i['lag_days']) for i in request['inputs']}
    require(set(refs) == declared, 'formula inputs differ: required '+str(sorted(declared))+'; used '+str(refs))
    validate_plan(assemble(relation, value), catalogue)
    return deepcopy(value)


def assemble(relation, calculation):
    """Mechanical assembly only: no rewriting of bindings, hypothesis or formula."""
    return {'plan': {**deepcopy(relation), **{k: calculation[k] for k in
                    ('expression', 'min_observations', 'min_improvement')}}, 'no_plan_reason': None}


def run_staged_workflow(runner, context, catalogue, feed, labels, protocol, freeze, *, variable_contract=None):
    from langgraph.graph import StateGraph, START, END
    typed = variable_contract
    registry = context.get('variable_sources') if typed else None
    if typed: typed.validate_registry(registry)
    def stopped(s): return s.get('stop_reason') is not None
    def failure(s, stage, exc):
        return {**s, 'status': stage+'_failed', 'stop_reason': str(exc), 'proposal_error': str(exc)}
    def research(s):
        try:
            research_context = typed.research_context(context) if typed else context
            value = runner.structured('relation_researcher', typed.RELATION if typed else RELATION, research_context, {},
                lambda v: (typed.validate_relation if typed else validate_relation)(v, catalogue))
            print('Relation stage:', 'proposed' if value['relation'] else 'abstained', flush=True)
            return {**s, 'relation': value, 'status': 'relation_proposed' if value['relation'] else 'no_relation',
                    'stop_reason': value['reason']}
        except (ValueError, TypeError, KeyError) as exc: return failure(s, 'relation', exc)
    def data(s):
        if stopped(s): return s
        relation = s['relation']['relation']; mids = {v for b in relation['bindings'] for v in b.values() if v is not None}
        try:
            data_context = {'phase': 'data_selection', 'relation': relation,
                'catalogue': [catalogue[k] for k in sorted(mids)],
                'available_numeric_tool': 'historical_yes_contract_trade_price', 'labels_visible': False}
            if typed: data_context['source_registry'] = registry
            value = runner.structured('method_data_member', typed.DATA if typed else DATA, data_context, {},
                lambda v: typed.validate_mapping(v, relation, registry) if typed else validate_data(v, relation))
            return {**s, 'data_request': value}
        except (ValueError, TypeError, KeyError) as exc: return failure(s, 'data_selection', exc)
    def availability(s):
        if stopped(s): return s
        check = (typed.check_sources(s['data_request'], s['relation']['relation'], registry, catalogue, feed, protocol)
                 if typed else check_data(s['data_request'], s['relation']['relation'], catalogue, feed, protocol))
        ready = check['status'] == 'ready_for_calculation'
        print('Data stage:', check['status'], 'joint inputs', check['jointly_available'], flush=True)
        return {**s, 'data_check': check, 'status': check['status'],
                'stop_reason': None if ready else check['status']}
    def quant(s):
        if stopped(s): return s
        relation = s['relation']['relation']; request = s['data_request']
        extra = {}
        if typed:
            extra = {'declared_variables': relation['variables'], 'source_bindings': request,
                     'source_registry_sha256': canonical_hash(registry)}
            request = typed.compiler_request(request, relation, registry)
            relation = typed.project_relation(relation)
        summary = {k: v for k, v in s['data_check'].items() if k != 'observations'}
        try:
            value = runner.structured('method_quant_member', QUANT,
                {'phase': 'calculation_design', 'relation': relation, 'data_request': request, **extra},
                {'input_check': summary}, lambda v: validate_calculation(v, relation, request, catalogue))
            if value['expression'] is None:
                return {**s, 'calculation': value, 'status': 'no_calculation', 'stop_reason': value['reason']}
            proposal = assemble(relation, value)
            freeze(proposal)
            return {**s, 'calculation': value, 'proposal': proposal, 'status': 'plan_frozen'}
        except (ValueError, TypeError, KeyError) as exc: return failure(s, 'calculation', exc)
    def execute(s):
        if stopped(s): return s
        result = execute_plan(s['proposal'], catalogue, feed, labels, protocol)
        print('Scoring stage:', result['status'], 'valid', result['valid_observations'], flush=True)
        return {**s, 'result': result, 'status': result['status']}
    def review(s):
        if s['result'] is None: return s
        summary = {k: v for k, v in s['result'].items() if k != 'observations'}
        try:
            value = runner.structured('experiment_auditor', REVIEWER, {'phase': 'after_training_tool_feedback'},
                {'result': summary}, lambda v: validate_review(v, s['result']))
            return {**s, 'review': value}
        except (ValueError, TypeError, KeyError) as exc: return {**s, 'review_error': str(exc)}
    graph = StateGraph(dict)
    nodes = [('relation', research), ('data', data), ('availability', availability), ('quant', quant), ('execute', execute), ('review', review)]
    for name, fn in nodes: graph.add_node(name, fn)
    path = [START]+[n for n, _ in nodes]+[END]
    for a, b in zip(path, path[1:]): graph.add_edge(a, b)
    initial = {'kind': 'typed_staged_method_workflow_v1' if typed else 'staged_method_workflow_v1', 'status': 'started', 'stop_reason': None,
               'relation': None, 'data_request': None, 'data_check': None, 'calculation': None,
               'proposal': None, 'proposal_error': None, 'result': None, 'review': None, 'review_error': None}
    return graph.compile().invoke(initial, {'max_concurrency': 1, 'recursion_limit': 10})
