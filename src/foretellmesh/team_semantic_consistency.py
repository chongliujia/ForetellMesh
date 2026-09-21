"""Provisional meaning checks, calibrated on fixed mechanism probes.

Same-Base review is fallible, not a semantic oracle. Topic distance is never a
program rule. All probes are diagnostic, with no labels/prices or profit claims.
"""
from copy import deepcopy
from pathlib import Path

from .data import strict_json, sha256_file
from .schema import fields
from .synthetic_sft import canonical_hash
from .team_learning import require
from .team_executable_methods import bounded_text

REVIEW = '''You are a research consistency reviewer. Check the hypothesis against ONE actual prediction target's
historical contract rules and all its declared input variables. This is a meaning check BEFORE data acquisition.
Return ONLY JSON with exactly target_check and variable_checks.
target_check has exactly verdict and reason. verdict is aligned, mismatch, or uncertain; reason is under 400 characters.
variable_checks contains exactly one object per supplied input variable, each exactly variable_id, verdict, reason.
Use the given variable IDs; verdict has the same three values, reason under 400 characters.
ALIGNED means the hypothesis actually describes a test of the chosen target and uses this measured variable with its
declared meaning, quantity and unit. It does NOT mean the relation is predictive, causal, supported by data or profitable.
MISMATCH means a concrete contradiction or substitution: the narrative forecasts a different event, a different
quantity is substituted for the stated measurement, or a variable has no described role in the proposed test.
UNCERTAIN means the declared meaning/link is too ambiguous to assess. Explain the specific ambiguity.
Do not demand empirical proof at this stage. An explicitly stated, testable cross-topic association can be aligned
even if surprising or unproven; unrelated topics ALONE are not a reason for mismatch. Missing data alone is not a
semantic mismatch. Rules do not prove statistical dependence. Do not invent a link the author did not state.
Distinguish Yes contract prices from true event probabilities, revenues, merchandise sales, outcomes and asset prices.
Assess the TARGET that is actually bound, not merely the broad task or an event mentioned elsewhere in the narrative.
Review only supplied historical text. No outcomes, prices, current knowledge, proposed sources or score feedback are provided.'''
REVISION = '''\nA consistency review found a concrete mismatch or ambiguity in this proposal. You have ONE substantive
revision opportunity. Address the supplied findings explicitly in your new relation, or abstain if no faithful,
testable hypothesis can be stated. Do not invent evidence or profitability. Different topics are permitted, but the
hypothesis must actually describe how its declared variables are to be tested against its bound prediction target.
Keep the active task's admitted contract IDs. Correct the declared measurement rather than silently substituting prices.
A clearer statement is not evidence of predictive validity. Return the same relation/reason schema requested above.'''


def validate_review(value, prediction):
    fields(value, {'target_check','variable_checks'}, 'consistency review')
    def check(item, variable=False):
        fields(item, {'verdict','reason'} | ({'variable_id'} if variable else set()), 'consistency finding')
        require(item['verdict'] in ('aligned','mismatch','uncertain'), 'invalid consistency verdict')
        bounded_text(item['reason'],400)
    check(value['target_check'])
    rows=value['variable_checks'];required=set(prediction['input_ids']);seen=set()
    require(isinstance(rows,list) and len(rows)==len(required), 'review must cover every target input')
    for row in rows:
        check(row,True);key=row['variable_id']
        require(isinstance(key,str) and key in required and key not in seen,'unknown/duplicate reviewed variable')
        seen.add(key)
    return deepcopy(value)


def review_relation(runner, relation, catalogue):
    from .team_multi_input import validate_relation
    validate_relation({'relation':relation,'reason':None},catalogue)
    variables={v['variable_id']:v for v in relation['variables']};checks=[]
    for pred in relation['predictions']:
        target=pred['target']
        context={'phase':'pre_data_consistency_review','hypothesis':relation['hypothesis'],
            'prediction_kind':relation['forecast_target'],'horizon_days':relation['horizon_days'],
            'target':{'market_id':target,'historical_rule':catalogue[target]['question']},
            'variables':[{**deepcopy(variables[key]),'historical_rule':catalogue[variables[key]['market_id']]['question']}
                         for key in pred['input_ids']],
            'empirical_effectiveness_not_tested':True}
        value=None;error=None
        try:
            value=runner.structured('semantic_consistency_reviewer',REVIEW,context,{},lambda v:validate_review(v,pred))
        except (ValueError,TypeError,KeyError) as exc:error=str(exc)
        checks.append({'target':target,'context_sha256':canonical_hash(context),'review':value,'error':error})
    complete=all(c['error'] is None for c in checks)
    aligned=complete and all(c['review']['target_check']['verdict']=='aligned'
        and all(v['verdict']=='aligned' for v in c['review']['variable_checks']) for c in checks)
    result={'kind':'provisional_consistency_review_v1','relation_sha256':canonical_hash(relation),'checks':checks,
        'status':'review_failed' if not complete else 'provisionally_consistent' if aligned else 'revision_required',
        'complete':complete,'provisionally_consistent':aligned,'semantic_truth_verified':False,
        'predictive_effectiveness_verified':False,'reviewer_is_same_frozen_base':True}
    result['review_sha256']=canonical_hash(result);return result


def feedback_summary(history):
    result=[]
    for i,item in enumerate(history):
        review=item['review'];findings=[]
        for row in review['checks']:
            if row['error']:
                findings.append({'target':row['target'],'error':row['error'][:300]});continue
            value=row['review']
            if value['target_check']['verdict']!='aligned':
                findings.append({'target':row['target'],**value['target_check']})
            findings.extend({'target':row['target'],**v} for v in value['variable_checks'] if v['verdict']!='aligned')
        result.append({'round':i+1,'status':review['status'],'review_sha256':review['review_sha256'],
                       'finding_count':len(findings),'findings':findings[:8]})
    return result


def check_and_revise(runner, relation_value, context, catalogue, *, named_kind, probe_qualified):
    from .team_multi_input import RELATION,RELATION_NAMED_KIND,validate_relation,validate_named_relation
    history=[];current=deepcopy(relation_value);revision_error=None
    for round_index in range(2):
        review=review_relation(runner,current['relation'],catalogue)
        history.append({'relation':deepcopy(current['relation']),'review':review})
        if review['provisionally_consistent']:
            return {'relation':current,'semantic_reviews':history,'semantic_revision_error':revision_error,
                'status':'semantic_screen_passed' if probe_qualified else 'semantic_gate_unqualified',
                'stop_reason':None if probe_qualified else 'Reviewer did not pass all fixed mechanism probes.'}
        if not review['complete'] or round_index==1:break
        external=deepcopy(current['relation'])
        if named_kind:external['prediction_kind']=external.pop('forecast_target')
        revision_context={**deepcopy(context),'semantic_revision_of':canonical_hash(current['relation'])}
        try:
            current=runner.structured('relation_researcher',(RELATION_NAMED_KIND if named_kind else RELATION)+REVISION,
                revision_context,{'previous_relation':external,'consistency_feedback':feedback_summary([history[-1]])},
                lambda v:(validate_named_relation if named_kind else validate_relation)(v,catalogue))
        except (ValueError,TypeError,KeyError) as exc:
            revision_error=str(exc);break
        if current['relation'] is None:
            return {'relation':current,'semantic_reviews':history,'semantic_revision_error':None,
                    'status':'semantic_abstention','stop_reason':current['reason']}
    return {'relation':current,'semantic_reviews':history,'semantic_revision_error':revision_error,
        'status':'semantic_revision_failed' if revision_error else 'semantic_review_failed' if not history[-1]['review']['complete'] else 'semantic_rejected',
        'stop_reason':revision_error or 'Consistency review remains uncertain or reports an unresolved mismatch.'}


def build_probes(config,catalogue):
    """Known bad archived output and explicit synthetic mechanism controls.

    These controls are never supplied to the researcher or data member, and
    never become strategy candidates, price scores or training supervision.
    """
    from .team_multi_input import validate_relation
    from .team_variable_contracts import SOURCE
    root=Path(config['semantic_probe_source_run']);report=strict_json((root/'report.json').read_text())
    require(sha256_file(root/'report.json')==config['semantic_probe_report_sha256'],'semantic probe source changed')
    for name in ('attempt_01.json','catalogue.json'):
        require(sha256_file(root/name)==report['artifact_hashes'][name],'semantic probe artifact changed')
    attempt=strict_json((root/'attempt_01.json').read_text());archive=strict_json((root/'catalogue.json').read_text())
    bad=attempt['workflow']['relation']['relation'];validate_relation({'relation':bad,'reason':None},catalogue)
    for mid in {v['market_id'] for v in bad['variables']}|{p['target'] for p in bad['predictions']}:
        require(archive[mid]==catalogue[mid],'probe historical rules changed')
    require(len(bad['predictions'])==1,'probe needs one preserved target')
    target=bad['predictions'][0]['target'];other=bad['variables'][0]['market_id']
    require(target!=other,'cross-contract probe needs distinct inputs')
    def variable(key,mid,lag):
        return {'variable_id':key,'market_id':mid,'definition':'Historical Yes share trade price of contract '+mid+'.',
                'quantity':SOURCE['quantity'],'unit':SOURCE['unit'],'lag_days':lag}
    positive={'hypothesis':'Test whether the current Yes trade price of contract '+target+' predicts its own Yes trade price seven days later. This is an unverified persistence reference.',
        'forecast_target':'future_yes_price','horizon_days':7,'variables':[variable('current_quote',target,0)],
        'predictions':[{'target':target,'input_ids':['current_quote']}]}
    cross=deepcopy(positive)
    cross['hypothesis']='Empirically test whether the one-day-lagged Yes trade price of contract '+other+' adds information about the seven-day future Yes trade price of contract '+target+' beyond its current Yes price. No causal or predictive edge is assumed.'
    cross['variables'].append(variable('other_quote',other,1));cross['predictions'][0]['input_ids'].append('other_quote')
    cases=[{'case_id':'preserved_target_measurement_mismatch','origin':'archived_model_output','expected':'revision_required','relation':bad},
           {'case_id':'same_contract_measurement_control','origin':'synthetic_mechanism_control_not_strategy','expected':'provisionally_consistent','relation':positive},
           {'case_id':'cross_topic_testable_association_control','origin':'synthetic_mechanism_control_not_strategy','expected':'provisionally_consistent','relation':cross}]
    for case in cases:validate_relation({'relation':case['relation'],'reason':None},catalogue)
    return cases


def run_probes(runner,cases,catalogue):
    results=[]
    for case in cases:
        # Neither expected verdict nor origin/case name is included in the model request.
        review=review_relation(runner,case['relation'],catalogue)
        results.append({'case_id':case['case_id'],'origin':case['origin'],'expected':case['expected'],
            'relation_sha256':canonical_hash(case['relation']),'review':review,
            'matched_expected':review['complete'] and review['status']==case['expected']})
    return {'kind':'consistency_reviewer_mechanism_probes_v1','cases':results,
        'qualified_for_this_bounded_run':bool(results) and all(r['matched_expected'] for r in results),
        'semantic_reliability_established':False,'new_predictions_scored':0,
        'limitations':'Three exposed training mechanism cases are not general semantic accuracy, independent validation or forecasting evidence.'}
