"""Explicit multi-contract variables and one forecast per target/time.

An opt-in research protocol; the prior single-peer interpreter remains intact.
Only registered, time-bounded sources may supply variables. Unknown external
measurements remain missing, and no model-written expression executes as code.
"""
import ast
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta
import math
import re
import statistics

from .schema import fields, timestamp, iso, ValidationError
from .synthetic_sft import canonical_hash
from .team_learning import require
from .team_executable_methods import (bounded_text, compile_expression, calculate, fresh_price,
                                     validate_protocol, REVIEWER, validate_review)
from . import team_variable_contracts as typed

RELATION = '''You are the research member. Design one tentative, testable multi-contract relationship from the
active task and supplied historical rules. Return ONLY JSON with exactly relation and reason.
If no faithful hypothesis is possible, relation=null and reason is a short explanation. Otherwise reason=null.
relation has exactly hypothesis, forecast_target, horizon_days, variables, predictions.
hypothesis describes the proposed predictive link under 600 characters, not an assertion of profit.
forecast_target is future_yes_price or resolves_yes. horizon_days is integer 1..14 for future_yes_price, null for resolves_yes.
variables is 1..8 objects with exactly variable_id, market_id, definition, quantity, unit, lag_days.
variable_id is a unique short identifier starting with a letter. market_id is an exact STRING ID from the supplied
catalogue, identifying the contract whose measurement this variable describes. definition is under 240 characters.
quantity and unit describe the ACTUAL measurement. lag_days is integer 0..30, relative to forecast observation time.
Multiple different market_ids may supply variables for ONE target. There are no target/peer roles or pair bindings.
Do not relabel movie revenue, external asset prices or sport results as Yes contract prices. External measurements may
be declared accurately; a data member will report them missing when no exact historical source is registered.
predictions is 1..4 objects with exactly target and input_ids. target is a supplied STRING market ID, appearing ONCE.
input_ids is a nonempty list of declared variable_id strings used together to forecast that target. Put ALL inputs for
one target in ONE list; never duplicate the target once for each input. Variables may be shared across different targets.
Every variable must be used by at least one prediction. Different variables cannot duplicate an identical measurement
(same market_id, quantity, unit and lag_days). Union of variable market_ids and prediction targets must use all and only
the active task's market_ids. No formula is required yet. Missing data is a valid result, not permission to invent it.
A proposed relation is not verified by valid JSON. Do not mention a prediction target that has no supplied market ID.'''
QUANT = '''You are the quant member. Calculate the fixed, explicitly bound multi-input hypothesis.
Return ONLY JSON with exactly expressions, min_observations, min_improvement, reason.
expressions is a list of objects with exactly target and expression, one for each fixed prediction target.
Read a declared variable only as v('variable_id'). Its contract, quantity, unit and historical lag are already fixed.
Each target's expression must use ALL and ONLY its declared input_ids. Multiple peers jointly produce ONE scalar forecast.
Allowed arithmetic: + - * /, parentheses, finite numeric constants, abs(x), min(x,y), max(x,y). No other functions or code.
Predict the stated Yes price/event probability in [0,1]; missing inputs/division by zero/out-of-range outputs fail.
Do not change the hypothesis, variables, targets or input assignments. Do not invent a price proxy for absent external data.
min_observations is integer 8..100 for future_yes_price or 3..100 for resolves_yes.
min_improvement is numeric >0 and <=1, the required reduction in event-group-averaged error versus current target price.
reason=null with expressions. If no faithful computation is possible, expressions=null, min_observations=null,
min_improvement=null, and explain in reason. The screen requires at least 3 target event groups and 80% valid coverage.
This is reused training research, not independent validation or net profitability. Do not optimize by writing ground truth.'''


RELATION_NAMED_KIND = RELATION.replace('forecast_target', 'prediction_kind') +     "\nprediction_kind is a TYPE ENUM, never a market ID: choose exactly future_yes_price or resolves_yes. "     "Put contract IDs only in variables.market_id and predictions.target. Do not copy a target ID into prediction_kind."


def validate_named_relation(value, catalogue):
    fields(value, {'relation','reason'}, 'named-kind relation')
    if value['relation'] is None:return validate_relation(value,catalogue)
    relation=deepcopy(value['relation'])
    fields(relation, {'hypothesis','prediction_kind','horizon_days','variables','predictions'}, 'named-kind relation graph')
    require(relation['prediction_kind'] in ('future_yes_price','resolves_yes'),
            'prediction_kind must be the TYPE future_yes_price or resolves_yes, never a market ID; got '+str(relation['prediction_kind']))
    relation['forecast_target']=relation.pop('prediction_kind')
    # Only an explicit schema-key translation; no inferred or repaired values.
    return validate_relation({'relation':relation,'reason':value['reason']},catalogue)


def validate_relation(value, catalogue):
    fields(value, {'relation','reason'}, 'multi-input relation')
    relation=value['relation']
    if relation is None:
        bounded_text(value['reason']);return deepcopy(value)
    require(value['reason'] is None, 'relation and abstention conflict')
    fields(relation, {'hypothesis','forecast_target','horizon_days','variables','predictions'}, 'relation graph')
    bounded_text(relation['hypothesis'])
    require(relation['forecast_target'] in ('future_yes_price','resolves_yes'), 'invalid forecast target')
    h=relation['horizon_days']
    require((type(h) is int and 1<=h<=14) if relation['forecast_target']=='future_yes_price' else h is None,
            'target/horizon mismatch')
    variables=relation['variables'];require(isinstance(variables,list) and 1<=len(variables)<=8, 'unbounded variables')
    ids=set();measurements=set()
    for var in variables:
        fields(var, {'variable_id','market_id','definition','quantity','unit','lag_days'}, 'explicit variable')
        key=var['variable_id'];mid=var['market_id']
        require(isinstance(key,str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}',key) and key not in ids,
                'invalid/duplicate variable ID')
        require(isinstance(mid,str) and mid in catalogue, 'variable market_id must be an admitted string ID')
        ids.add(key);bounded_text(var['definition'],240);bounded_text(var['quantity'],80);bounded_text(var['unit'],80)
        require(type(var['lag_days']) is int and 0<=var['lag_days']<=30, 'future/unbounded variable lag')
        measurement=(mid,var['quantity'],var['unit'],var['lag_days'])
        require(measurement not in measurements, 'duplicate measurement under another variable ID');measurements.add(measurement)
        if var['quantity']==typed.SOURCE['quantity']:
            require(var['unit']==typed.SOURCE['unit'], 'incorrect Yes share price unit')
    predictions=relation['predictions']
    require(isinstance(predictions,list) and 1<=len(predictions)<=4, 'unbounded prediction targets')
    targets=set();used=set()
    for pred in predictions:
        fields(pred, {'target','input_ids'}, 'joint-input prediction')
        mid=pred['target'];refs=pred['input_ids']
        require(isinstance(mid,str) and mid in catalogue and mid not in targets,
                'unknown/repeated prediction target: combine its inputs into ONE input_ids list')
        targets.add(mid)
        require(isinstance(refs,list) and 1<=len(refs)<=8 and all(isinstance(k,str) for k in refs)
                and len(set(refs))==len(refs) and set(refs)<=ids, 'unknown/duplicate/empty prediction input IDs')
        used.update(refs)
    require(used==ids, 'declared variables must all be used by predictions')
    return deepcopy(value)


def used_markets(relation):
    return {v['market_id'] for v in relation['variables']} | {p['target'] for p in relation['predictions']}


def relation_fingerprint(relation):
    signatures={v['variable_id']:{k:v[k] for k in ('market_id','quantity','unit','lag_days')} for v in relation['variables']}
    body={k:relation[k] for k in ('forecast_target','horizon_days')}
    body['predictions']=sorted([{'target':p['target'],'inputs':sorted([signatures[k] for k in p['input_ids']],key=canonical_hash)}
                                for p in relation['predictions']],key=canonical_hash)
    return canonical_hash(body)


def compile_variables(expression):
    """Lower verified v('id') references into the existing safe arithmetic AST.

    Internal slots are value indexes, never historical lags or market IDs.
    Both the original calls and the entire lowered tree are allowlist-checked.
    """
    bounded_text(expression,400)
    try:root=ast.parse(expression,mode='eval')
    except (SyntaxError,RecursionError) as exc:raise ValidationError('invalid variable expression') from exc
    require(sum(1 for _ in ast.walk(root))<=96, 'expression exceeds node budget')
    refs=set()
    for node in ast.walk(root):
        if not isinstance(node,ast.Call):continue
        require(isinstance(node.func,ast.Name) and not node.keywords and node.func.id in ('v','abs','min','max'),
                'unsupported variable expression function')
        if node.func.id=='v':
            require(len(node.args)==1 and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str)
                    and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}',node.args[0].value), 'v requires one literal variable ID')
            refs.add(node.args[0].value)
    require(0<len(refs)<=8, 'expression needs 1..8 declared variables')
    slots={key:i for i,key in enumerate(sorted(refs))}
    class Lower(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='v':
                return ast.Call(func=ast.Name(id='p',ctx=ast.Load()),
                    args=[ast.Constant(value='target'),ast.Constant(value=slots[node.args[0].value])],keywords=[])
            return self.generic_visit(node)
    lowered=ast.fix_missing_locations(Lower().visit(root))
    compiled,_=compile_expression(ast.unparse(lowered))
    return compiled,slots


def validate_calculation(value,relation):
    fields(value, {'expressions','min_observations','min_improvement','reason'}, 'graph calculation')
    if value['expressions'] is None:
        bounded_text(value['reason'])
        require(value['min_observations'] is None and value['min_improvement'] is None, 'abstention has thresholds')
        return deepcopy(value)
    require(value['reason'] is None, 'calculation and abstention conflict')
    predictions={p['target']:p for p in relation['predictions']};seen=set()
    require(isinstance(value['expressions'],list) and len(value['expressions'])==len(predictions), 'calculation target count differs')
    for item in value['expressions']:
        fields(item, {'target','expression'}, 'target calculation');target=item['target']
        require(isinstance(target,str) and target in predictions and target not in seen, 'unknown/duplicate calculation target')
        seen.add(target);_,refs=compile_variables(item['expression'])
        require(set(refs)==set(predictions[target]['input_ids']), 'formula must use ALL and ONLY assigned variable IDs for '+target)
    minimum=3 if relation['forecast_target']=='resolves_yes' else 8
    require(type(value['min_observations']) is int and minimum<=value['min_observations']<=100, 'invalid minimum observations')
    gain=value['min_improvement']
    require(type(gain) in (int,float) and math.isfinite(gain) and 0<gain<=1, 'invalid minimum improvement')
    return deepcopy(value)


def input_at(relation,prediction,catalogue,feed,at,max_age):
    variables={v['variable_id']:v for v in relation['variables']};values={};evidence=[];errors=[]
    target=prediction['target']
    require(all(variables[k]['quantity']==typed.SOURCE['quantity'] and variables[k]['unit']==typed.SOURCE['unit']
                for k in prediction['input_ids']), 'input quantity/unit is not a registered Yes share price')
    if at<timestamp(catalogue[target]['initialized_at'],'init'):
        return None,[],['baseline:before_initialization']
    baseline,error=fresh_price(feed,target,at,max_age)
    if error:errors.append('baseline:'+error)
    for key in sorted(prediction['input_ids']):
        var=variables[key];mid=var['market_id'];point=at-timedelta(days=var['lag_days'])
        if point<timestamp(catalogue[mid]['initialized_at'],'init'):
            errors.append(key+':before_initialization');continue
        quote,error=fresh_price(feed,mid,point,max_age)
        if error:errors.append(key+':'+error)
        else:
            evidence.append({**quote,'variable_id':key,'lag_days':var['lag_days']});values[key]=quote['price']
    return ({'values':values,'baseline':None if baseline is None else baseline['price'],'baseline_evidence':baseline},evidence,errors)


def grid(prediction,catalogue,protocol,horizon):
    init=timestamp(catalogue[prediction['target']]['initialized_at'],'init')
    start=init.replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=1)
    cutoff=timestamp(protocol['as_of'],'cutoff')
    for day in range(0,protocol['max_days_per_target'],horizon or 1):
        at=start+timedelta(days=day)
        if at+timedelta(days=horizon or 0)>cutoff:break
        yield at
        if horizon is None:break


def check_sources(mapping,relation,registry,catalogue,feed,protocol):
    validate_protocol(protocol);validate_relation({'relation':relation,'reason':None},catalogue)
    mapping=typed.validate_mapping(mapping,relation,registry);variables={v['variable_id']:v for v in relation['variables']}
    missing=[{'variable':variables[b['variable_id']],'reason':b['missing_reason']} for b in mapping['bindings'] if b['source_id'] is None]
    rows=[]
    if not missing:
        for prediction in relation['predictions']:
            for at in grid(prediction,catalogue,protocol,relation['horizon_days']):
                inputs,evidence,errors=input_at(relation,prediction,catalogue,feed,at,protocol['max_quote_age_seconds'])
                rows.append({'target':prediction['target'],'observation_time':iso(at),'evidence':evidence,
                             'baseline_evidence':None if inputs is None else inputs['baseline_evidence'],'errors':errors})
    ready=sum(not r['errors'] for r in rows)
    result={'kind':'multi_input_source_check_v1','status':'unavailable_variables' if missing else 'ready_for_calculation' if ready else 'no_joint_observations',
        'unavailable_variables':missing,'availability_not_attempted':bool(missing),'jointly_available':ready,
        'observations':rows,'failures':dict(Counter(e for r in rows for e in r['errors'])),
        'relation_sha256':canonical_hash(relation),'mapping_sha256':canonical_hash(mapping),'source_registry_sha256':canonical_hash(registry),
        'protocol_sha256':canonical_hash(protocol),'interpretation':'Input-only check, not future target coverage, settlement boundaries or predictive skill.'}
    result['check_sha256']=canonical_hash(result);return result


def validate_plan(proposal,catalogue,registry):
    fields(proposal, {'plan','no_plan_reason'}, 'multi-input plan')
    plan=proposal['plan'];require(plan is not None and proposal['no_plan_reason'] is None,'multi-input evaluator requires a frozen plan')
    fields(plan, {'relation','mapping','calculation'}, 'frozen graph plan')
    validate_relation({'relation':plan['relation'],'reason':None},catalogue)
    mapping=typed.validate_mapping(plan['mapping'],plan['relation'],registry)
    require(all(b['source_id'] is not None for b in mapping['bindings']), 'unbound variable cannot enter scoring')
    validate_calculation(plan['calculation'],plan['relation'])
    require(plan['calculation']['expressions'] is not None,'abstention cannot enter scoring')
    return plan


def forecast_at(plan,prediction,catalogue,feed,at,max_age):
    mapping={b['variable_id']:b for b in plan['mapping']['bindings']}
    require(all(k in mapping and mapping[k]['source_id']==typed.SOURCE['source_id']
                and mapping[k]['missing_reason'] is None for k in prediction['input_ids']),
            'unbound variable cannot enter forecasting')
    expressions={r['target']:r['expression'] for r in plan['calculation']['expressions']}
    root,slots=compile_variables(expressions[prediction['target']])
    inputs,evidence,errors=input_at(plan['relation'],prediction,catalogue,feed,at,max_age)
    if errors:return None,evidence,';'.join(errors)
    try:value=calculate(root,{('target',index):inputs['values'][key] for key,index in slots.items()})
    except ValidationError as exc:return None,evidence,str(exc)
    return {'prediction':value,'baseline':inputs['baseline'],'baseline_evidence':inputs['baseline_evidence']},evidence,None


def execute_plan(proposal,catalogue,registry,feed,labels,protocol):
    validate_protocol(protocol);plan=validate_plan(proposal,catalogue,registry)
    relation=plan['relation'];rows=[];keys=set();feedback=timestamp(protocol['feedback_cutoff'],'feedback')
    for pred in relation['predictions']:
        target=pred['target'];label=labels[target]
        require(type(label['outcome']) is int and label['outcome'] in (0,1),'invalid outcome')
        resolution=timestamp(label['resolution_time'],'resolution');available=timestamp(label['available_at'],'availability')
        require(resolution<=available,'label precedes resolution')
        for at in grid(pred,catalogue,protocol,relation['horizon_days']):
            horizon=relation['horizon_days'];end=at+timedelta(days=horizon) if horizon else None
            if at>=resolution or end is not None and end>=resolution:break
            key=(target,iso(at));require(key not in keys,'duplicate target/time scoring');keys.add(key)
            forecast,evidence,error=forecast_at(plan,pred,catalogue,feed,at,protocol['max_quote_age_seconds'])
            row={'target':target,'event_group_id':catalogue[target]['event_group_id'],'observation_time':iso(at),
                 'forecast':forecast,'feature_evidence':evidence,'error':error,'target_evidence':None,'model_loss':None,'baseline_loss':None}
            if end is not None:
                actual,target_error=fresh_price(feed,target,end,protocol['max_quote_age_seconds'])
                if target_error:row['error']=row['error'] or 'target:'+target_error
                elif timestamp(actual['source_time'],'target')<=at:row['error']=row['error'] or 'target:not_after_observation'
                else:row['target_evidence']=actual
                y=None if actual is None else actual['price']
            else:
                y=label['outcome']
                if available>feedback:row['error']=row['error'] or 'target:feedback_not_available'
                else:row['target_evidence']={'outcome':y,'resolution_time':iso(resolution),'available_at':iso(available)}
            if row['error'] is None:
                row['model_loss']=(forecast['prediction']-y)**2;row['baseline_loss']=(forecast['baseline']-y)**2
            row['observation_sha256']=canonical_hash(row);rows.append(row)
    valid=[r for r in rows if r['error'] is None];groups=defaultdict(list)
    for row in valid:groups[row['event_group_id']].append(row)
    scores={g:{'observations':len(rs),'model_loss':statistics.fmean(r['model_loss'] for r in rs),
               'baseline_loss':statistics.fmean(r['baseline_loss'] for r in rs)} for g,rs in sorted(groups.items())}
    improvement=statistics.fmean(s['baseline_loss']-s['model_loss'] for s in scores.values()) if scores else None
    coverage=len(valid)/len(rows) if rows else 0
    sufficient=(len(valid)>=plan['calculation']['min_observations'] and len(groups)>=protocol['min_event_groups'] and coverage>=protocol['min_coverage'])
    status='insufficient_data' if not sufficient else 'training_screen_passed' if improvement>=plan['calculation']['min_improvement'] else 'training_screen_failed'
    result={'kind':'multi_input_method_result_v1','proposal_sha256':canonical_hash(proposal),'protocol_sha256':canonical_hash(protocol),
        'metric':'brier' if relation['forecast_target']=='resolves_yes' else 'future_yes_price_mse','status':status,
        'observations_attempted':len(rows),'valid_observations':len(valid),'coverage':coverage,'target_event_groups':len(groups),
        'group_scores':scores,'mean_group_improvement':improvement,'failures':dict(Counter(r['error'] for r in rows if r['error'])),
        'eligible_for_independent_validation':status=='training_screen_passed','independent_validation':False,
        'effectiveness_verified':False,'fine_tuning_admitted':False,'net_profit_evaluated':False,'observations':rows}
    result['result_sha256']=canonical_hash(result);return result


def run_workflow(runner,context,catalogue,feed,labels,protocol,freeze,*,named_kind=False,semantic_gate=False,probe_qualified=False,semantic_advisory=False):
    from langgraph.graph import StateGraph,START,END
    context={k:v for k,v in context.items() if k not in ('semantic_probe_cases','semantic_probe_qualified')}
    require(not semantic_advisory or semantic_gate, 'advisory mode requires review node')
    registry=context['variable_sources'];typed.validate_registry(registry)
    def failure(s,stage,exc):return {**s,'status':stage+'_failed','stop_reason':str(exc),'proposal_error':str(exc)}
    def research(s):
        instruction=RELATION_NAMED_KIND if named_kind else RELATION
        if 'data_catalogue' in context:
            from .team_data_catalogue import GUIDANCE
            instruction+=GUIDANCE
        try:
            value=runner.structured('relation_researcher',instruction,typed.research_context(context),{},
                lambda v:(validate_named_relation if named_kind else validate_relation)(v,catalogue))
            return {**s,'relation':value,'status':'relation_proposed' if value['relation'] else 'no_relation','stop_reason':value['reason']}
        except (ValueError,TypeError,KeyError) as exc:return failure(s,'relation',exc)
    def consistency(s):
        if s['stop_reason'] is not None:return s
        if semantic_advisory:
            from .team_review_boundary import observe
            try:return {**s,**observe(runner,s['relation'],catalogue,protocol)}
            except (ValueError,TypeError,KeyError) as exc:return failure(s,'execution_contract',exc)
        from .team_semantic_consistency import check_and_revise
        return {**s,**check_and_revise(runner,s['relation'],typed.research_context(context),catalogue,
            named_kind=named_kind,probe_qualified=probe_qualified)}
    def data(s):
        if s['stop_reason'] is not None:return s
        try:
            value=runner.structured('method_data_member',typed.DATA,{'phase':'multi_input_data_selection','relation':s['relation']['relation'],
                'source_registry':registry,'labels_visible':False,
                **({'execution_contract':s['execution_contract']} if semantic_advisory else {})},{},lambda v:typed.validate_mapping(v,s['relation']['relation'],registry))
            return {**s,'data_request':value}
        except (ValueError,TypeError,KeyError) as exc:return failure(s,'data_selection',exc)
    def availability(s):
        if s['stop_reason'] is not None:return s
        check=check_sources(s['data_request'],s['relation']['relation'],registry,catalogue,feed,protocol)
        return {**s,'data_check':check,'status':check['status'],'stop_reason':None if check['status']=='ready_for_calculation' else check['status']}
    def quant(s):
        if s['stop_reason'] is not None:return s
        try:
            relation=s['relation']['relation']
            value=runner.structured('method_quant_member',QUANT,{'phase':'multi_input_calculation','relation':relation,'source_bindings':s['data_request'],
                **({'execution_contract':s['execution_contract']} if semantic_advisory else {})},
                {'input_check':{k:v for k,v in s['data_check'].items() if k!='observations'}},lambda v:validate_calculation(v,relation))
            if value['expressions'] is None:return {**s,'calculation':value,'status':'no_calculation','stop_reason':value['reason']}
            proposal={'plan':{'relation':deepcopy(relation),'mapping':deepcopy(s['data_request']),'calculation':value},'no_plan_reason':None}
            validate_plan(proposal,catalogue,registry);freeze(proposal)
            return {**s,'calculation':value,'proposal':proposal,'status':'plan_frozen'}
        except (ValueError,TypeError,KeyError) as exc:return failure(s,'calculation',exc)
    def execute(s):
        if s['stop_reason'] is not None:return s
        result=execute_plan(s['proposal'],catalogue,registry,feed,labels,protocol)
        return {**s,'result':result,'status':result['status']}
    def review(s):
        if s['result'] is None:return s
        try:
            value=runner.structured('experiment_auditor',REVIEWER,{'phase':'after_training_tool_feedback'},
                {'result':{k:v for k,v in s['result'].items() if k!='observations'}},lambda v:validate_review(v,s['result']))
            return {**s,'review':value}
        except (ValueError,TypeError,KeyError) as exc:return {**s,'review_error':str(exc)}
    graph=StateGraph(dict);nodes=[('research',research),('data',data),('availability',availability),('quant',quant),('execute',execute),('review',review)]
    if semantic_gate:nodes.insert(1,('consistency',consistency))
    for name,fn in nodes:graph.add_node(name,fn)
    path=[START]+[n for n,_ in nodes]+[END]
    for a,b in zip(path,path[1:]):graph.add_edge(a,b)
    return graph.compile().invoke({'kind':'multi_input_workflow_v1','status':'started','stop_reason':None,'relation':None,'data_request':None,
        'data_check':None,'calculation':None,'proposal':None,'proposal_error':None,'result':None,'review':None,'review_error':None},
        {'max_concurrency':1,'recursion_limit':10})
