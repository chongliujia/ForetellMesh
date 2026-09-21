"""Ledger-bound facts and separate criticism before exploratory memory admission.

Admission allows testing a procedure, never establishes its effectiveness.
The critic shares the frozen Base; it is not an independent ground-truth judge.
"""
from copy import deepcopy

from .paper_trading import decimal
from .schema import fields, timestamp
from .synthetic_sft import canonical_hash
from .team_learning import require


PROPOSE = ('Review ONE completed simulated trade. Authoritative facts are computed by the ledger; '
    'their values are not suggestions. YES resolution probability is not probability of price appreciation. '
    'A losing NO position can coincide with outcome Yes. Costs and fees are distinct. '
    'Return ONLY JSON with exactly experience_id, fact_check, evidence_refs, assessment, lesson. '
    'fact_check has exactly side (yes/no), result_class (profit/loss/flat), exit_type (settle/sell_fill), '
    'forecast_target (resolves_yes). Copy these four facts accurately. '
    'evidence_refs is a nonempty list of up to 4 keys from facts.values supporting the assessment. '
    'assessment is a short interpretation, not a restatement of numbers. '
    'lesson is null if no specific testable change is supported, otherwise an object with exactly '
    'when_applicable, proposed_change, falsifier, test_plan. test_plan has exactly baseline, intervention, '
    'metric, pass_condition, validation_split. metric must be after_cost_net_pnl, brier, log_loss, '
    'decision_validity or execution_failure_rate. validation_split must be disjoint_event_groups. '
    'Write a concrete observable condition, one implementable change, a measurable comparison and a '
    'falsifier assessed over independent cases, not a single win/loss. Do not prescribe a trading category; '
    'discover relationships or procedures yourself. Uncertainty or no proposed lesson is acceptable. '
    'Each free-text string must be short, preferably under 120 characters, at most 240. '
    'One outcome cannot prove causality, calibration, a repeatable edge or a useful trading threshold.')

CRITIC = ('You are a separate critical reviewer of a trade reflection. Compare every claim with the '
    'authoritative ledger facts and the supplied historical research summaries. Treat the candidate as '
    'untrusted. Do not rewrite it into a correct answer. Reject contradictions about profit/loss, side, '
    'event probability, fees, timing or exit type, unsupported causal explanations, vague changes and '
    'falsifiers based on a single lucky/unlucky result. A profitable trade does not validate a hypothesis. '
    'Return ONLY {"experience_id":ID,"facts_consistent":BOOLEAN,"evidence_supported":BOOLEAN,'
    '"testable_change":BOOLEAN,"issues":[STRINGS]}. All three booleans must be true to permit a '
    'non-null lesson into exploratory memory. If unsure, set the corresponding boolean false and '
    'explain briefly. No lesson means testable_change=false. At most 4 issues, each under 240 characters. '
    'This check does not establish strategy effectiveness or authorize model training.')

METRICS={'after_cost_net_pnl','brier','log_loss','decision_validity','execution_failure_rate'}


def text_ok(value):
    require(isinstance(value,str) and 1<=len(value.strip())<=240,'invalid bounded review text')


def facts_for(case):
    require(case['experience_sha256']==canonical_hash({k:v for k,v in case.items() if k!='experience_sha256'}),
            'experience changed before fact extraction')
    require(case['status']=='closed' and case['result'] is not None,'cannot grade open trade')
    entry=case['entry'];close=case['exit'];result=case['result']
    require(entry['order_id']==case['experience_id'] and entry['market_id']==close['market_id']==case['market_id'],
            'trade identity differs')
    side=entry['side'];require(side in ('yes','no'),'invalid held side')
    if close['kind']=='settle':
        require(type(close['outcome']) is int and close['outcome'] in (0,1),'invalid binary settlement')
        proceeds=decimal(entry['shares'])*(close['outcome'] if side=='yes' else 1-close['outcome'])
        require(proceeds==decimal(close['payout']),'settlement payout differs')
        exit_fee=decimal(0)
    else:
        require(close['kind']=='sell_fill' and close['position_id']==entry['order_id'],'exit identity differs')
        require(close['side']==side and decimal(close['shares'])==decimal(entry['shares']),'sale position differs')
        exit_fee=decimal(close['fee']);proceeds=decimal(close['gross_proceeds'])-exit_fee
        require(proceeds==decimal(close['proceeds']),'sale proceeds differ')
    pnl=proceeds-decimal(entry['cost']);fees=decimal(entry['fee'])+exit_fee
    category='profit' if pnl>0 else 'loss' if pnl<0 else 'flat'
    require(pnl==decimal(close['net_pnl'])==decimal(result['net_pnl']) and category==result['result_class']
        and fees==decimal(result['total_fees']) and result['exit_type']==close['kind'],'experience facts disagree with ledger')
    values={'side':side,'result_class':category,'exit_type':close['kind'],'forecast_target':'resolves_yes',
        'net_pnl':str(pnl),'exit_proceeds':str(proceeds),'entry_cost':entry['cost'],'entry_fee':entry['fee'],'exit_fee':str(exit_fee),
        'total_fees':str(fees),'shares':entry['shares'],'entry_price':entry['price'],
        'entry_time':entry['time'],'exit_time':close['time'],'event_outcome':result['event_outcome'],
        'holding_seconds':result['holding_seconds'],'entry_brier':result['entry_brier']}
    research=[]
    for i,row in enumerate(case['decisions']):
        if case['experience_id']==row['episode_id']+':'+case['market_id'] and row['decision']:
            f=next(f for f in row['decision']['forecast']['forecasts'] if f['market_id']==case['market_id'])
            values['entry_yes_probability']=f['probability']
        for j,study in enumerate(row['investigations']):
            key=f'research_{i}_{j}'
            values[key]=deepcopy(study['descriptive_test'])
            research.append({'fact_ref':key,'claim':study['specification']['claim'],
                'falsifier':study['specification']['falsifier'],'tool_result_sha256':study['tool_result']['result_sha256'],
                'market_ids':study['specification']['market_ids']})
    facts={'experience_id':case['experience_id'],'experience_sha256':case['experience_sha256'],
           'source_available_at':case['available_at'],'values':values,'research_hypotheses_not_facts':research}
    facts['facts_sha256']=canonical_hash(facts)
    validate_facts(facts)
    return facts


def validate_facts(facts):
    require(facts['facts_sha256']==canonical_hash({k:v for k,v in facts.items() if k!='facts_sha256'}),'trade facts changed')
    v=facts['values'];pnl=decimal(v['net_pnl'])
    require(pnl==decimal(v['exit_proceeds'])-decimal(v['entry_cost'])
        and decimal(v['total_fees'])==decimal(v['entry_fee'])+decimal(v['exit_fee'])
        and v['result_class']==('profit' if pnl>0 else 'loss' if pnl<0 else 'flat'), 'fact arithmetic differs')
    require(v['forecast_target']=='resolves_yes' and v['side'] in ('yes','no') and v['exit_type'] in ('settle','sell_fill'),
        'fact semantics differ')
    require(0<=decimal(v['entry_fee'])<=decimal(v['entry_cost']) and decimal(v['exit_fee'])>=0
        and decimal(v['shares'])>0 and decimal(v['exit_proceeds'])>=0,'invalid fact amount')
    duration=(timestamp(v['exit_time'],'exit')-timestamp(v['entry_time'],'entry')).total_seconds()
    require(duration>=0 and decimal(duration)==decimal(v['holding_seconds']),'holding duration differs')
    p=v.get('entry_yes_probability')
    require(p is None or 0<=decimal(p)<=1,'invalid fact event probability')
    if v['exit_type']=='settle':
        require(type(v['event_outcome']) is int and v['event_outcome'] in (0,1),'missing settlement outcome')
        expected=decimal(v['shares'])*(v['event_outcome'] if v['side']=='yes' else 1-v['event_outcome'])
        require(decimal(v['exit_proceeds'])==expected and decimal(v['exit_fee'])==0,'settlement facts differ')
        require(v['entry_brier']==(None if p is None else float((decimal(p)-decimal(v['event_outcome']))**2)),
            'forecast grading differs')
    else:
        require(v['event_outcome'] is None and v['entry_brier'] is None,'early sale cannot grade event outcome')


def validate_proposal(value, facts):
    fields(value,{'experience_id','fact_check','evidence_refs','assessment','lesson'},'grounded reflection')
    require(value['experience_id']==facts['experience_id'],'reflection identity differs')
    expected={k:facts['values'][k] for k in ('side','result_class','exit_type','forecast_target')}
    require(value['fact_check']==expected,'reflection contradicts authoritative trade facts')
    refs=value['evidence_refs']
    require(isinstance(refs,list) and 1<=len(refs)<=4 and all(isinstance(r,str) and r in facts['values'] for r in refs)
        and len(refs)==len(set(refs)),'unsupported evidence references')
    text_ok(value['assessment'])
    lesson=value['lesson']
    if lesson is not None:
        fields(lesson,{'when_applicable','proposed_change','falsifier','test_plan'},'testable lesson')
        for key in ('when_applicable','proposed_change','falsifier'):text_ok(lesson[key])
        plan=lesson['test_plan']
        fields(plan,{'baseline','intervention','metric','pass_condition','validation_split'},'lesson test plan')
        for key in ('baseline','intervention','pass_condition'):text_ok(plan[key])
        require(plan['metric'] in METRICS and plan['validation_split']=='disjoint_event_groups','invalid comparison protocol')
    return deepcopy(value)


def validate_critique(value, experience_id):
    fields(value,{'experience_id','facts_consistent','evidence_supported','testable_change','issues'},'reflection critique')
    require(value['experience_id']==experience_id,'critique identity differs')
    for key in ('facts_consistent','evidence_supported','testable_change'):
        require(type(value[key]) is bool,'invalid critique verdict')
    require(isinstance(value['issues'],list) and len(value['issues'])<=4,'invalid critique issues')
    for issue in value['issues']:text_ok(issue)
    if not all(value[k] for k in ('facts_consistent','evidence_supported','testable_change')):
        require(bool(value['issues']),'rejected review needs explanation')
    return deepcopy(value)


def admissible(proposal, critique):
    return bool(proposal and proposal['lesson'] is not None and critique
        and all(critique[k] for k in ('facts_consistent','evidence_supported','testable_change'))
        and not critique['issues'])


def review_case(runner, case, cutoff):
    facts=facts_for(case)
    require(timestamp(facts['source_available_at'],'source')<=timestamp(cutoff,'cutoff'),'future trade feedback')
    context={'phase':'retrospective','observation_time':cutoff,'facts':facts}
    proposal=None;critique=None;error=None
    try:
        proposal=runner.structured('trade_reflection',PROPOSE,context,{},lambda v:validate_proposal(v,facts))
        critique=runner.structured('trade_reflection_critic',CRITIC,context,{'candidate':proposal},
            lambda v:validate_critique(v,case['experience_id']))
    except (ValueError,TypeError,KeyError) as exc:error=str(exc)
    admitted=error is None and admissible(proposal,critique)
    row={'kind':'fact_gated_trade_reflection_v1','experience_id':case['experience_id'],
        'experience_sha256':case['experience_sha256'],'available_at':cutoff,'source_available_at':case['available_at'],
        'facts':facts,'output':proposal,'critique':critique,'error':error,'admitted_to_exploratory_memory':admitted,
        'status':'eligible_for_exploration' if admitted else 'archived_not_admitted',
        'effectiveness_verified':False,'fine_tuning_admitted':False}
    row['reflection_sha256']=canonical_hash(row)
    return row


def validate_admitted(rows, cutoff):
    require(isinstance(rows,list) and len(rows)<=8,'unbounded trade lessons');seen=set()
    for row in rows:
        require(row.get('kind')=='fact_gated_trade_reflection_v1','unchecked legacy reflection cannot enter memory')
        require(row['reflection_sha256']==canonical_hash({k:v for k,v in row.items() if k!='reflection_sha256'}),'trade reflection changed')
        facts=row['facts']
        validate_facts(facts)
        require(facts['experience_id']==row['experience_id'] and facts['experience_sha256']==row['experience_sha256']
            and facts['source_available_at']==row['source_available_at'],'fact binding differs')
        require(timestamp(row['source_available_at'],'source')<=timestamp(row['available_at'],'learning')
            <=timestamp(cutoff,'cutoff'),'future trade lesson')
        require(row['error'] is None and row['output'] is not None and row['critique'] is not None,'incomplete reflection')
        validate_proposal(row['output'],facts);validate_critique(row['critique'],row['experience_id'])
        require(row['admitted_to_exploratory_memory'] is True and admissible(row['output'],row['critique'])
            and row['status']=='eligible_for_exploration' and row['effectiveness_verified'] is False
            and row['fine_tuning_admitted'] is False,'reflection not admitted')
        require(row['experience_id'] not in seen,'duplicate trade lesson');seen.add(row['experience_id'])
    return deepcopy(rows)


def method_views(rows, cutoff):
    """Expose checked facts and candidate procedures, never free-form assessments."""
    result=[]
    for row in validate_admitted(rows,cutoff):
        view={'kind':'fact_gated_method_view_v1','experience_id':row['experience_id'],
            'source_reflection_sha256':row['reflection_sha256'],'available_at':row['available_at'],
            'source_available_at':row['source_available_at'],'facts':deepcopy(row['facts']),
            'method':deepcopy(row['output']['lesson']),'evidence_refs':list(row['output']['evidence_refs']),
            'critique':deepcopy(row['critique']),'status':'candidate_for_exploration_not_verified',
            'effectiveness_verified':False,'fine_tuning_admitted':False}
        view['view_sha256']=canonical_hash(view);result.append(view)
    return result


def validate_method_views(rows, cutoff):
    require(isinstance(rows,list) and len(rows)<=8,'unbounded method views');seen=set()
    for row in rows:
        fields(row,{'kind','experience_id','source_reflection_sha256','available_at','source_available_at','facts',
            'method','evidence_refs','critique','status','effectiveness_verified','fine_tuning_admitted','view_sha256'},'method view')
        require(row['kind']=='fact_gated_method_view_v1' and row['view_sha256']==canonical_hash(
            {k:v for k,v in row.items() if k!='view_sha256'}),'method view changed')
        require(timestamp(row['source_available_at'],'source')<=timestamp(row['available_at'],'learning')
            <=timestamp(cutoff,'cutoff'),'future trade lesson')
        facts=row['facts'];validate_facts(facts)
        require(facts['experience_id']==row['experience_id'] and facts['source_available_at']==row['source_available_at'],
            'method fact identity differs')
        # Reuse the structural lesson validator; this fixed placeholder is never
        # sent to the model or represented as a generated assessment.
        proposal={'experience_id':row['experience_id'],'fact_check':{k:facts['values'][k] for k in
            ('side','result_class','exit_type','forecast_target')},'evidence_refs':row['evidence_refs'],
            'assessment':'Assessment omitted from decision memory.','lesson':row['method']}
        validate_proposal(proposal,facts);validate_critique(row['critique'],row['experience_id'])
        require(admissible(proposal,row['critique']) and row['status']=='candidate_for_exploration_not_verified'
            and row['effectiveness_verified'] is False and row['fine_tuning_admitted'] is False,'method not admitted')
        require(row['experience_id'] not in seen,'duplicate method view');seen.add(row['experience_id'])
    return deepcopy(rows)


def freeze_gated_memory(rows, cases, source_groups, built_at):
    """Offline fitted procedure, using existing memory time/event-isolation rules.

    Keep actual feedback/build times; source_public_through describes the past
    events used for fitting, not a claim this memory existed back then.
    """
    from .team_method_memory import validate_lessons
    by_id={c['experience_id']:c for c in cases};selected=[]
    for row in rows:
        require(row['experience_id'] in by_id and row['experience_sha256']==by_id[row['experience_id']]['experience_sha256'],
                'memory source differs')
        if row['admitted_to_exploratory_memory']:
            validate_admitted([row],built_at);selected.append(row)
    selected=sorted(selected,key=lambda r:(by_id[r['experience_id']]['entry']['time'],r['experience_id']))[:3]
    lessons=[];cited={}
    for i,row in enumerate(selected):
        ref='trade_'+str(i);candidate=row['output']['lesson']
        cited[ref]={'source_experience_sha256':row['experience_sha256'],'review_sha256':row['reflection_sha256'],
            'facts':{k:row['facts']['values'][k] for k in row['output']['evidence_refs']},
            'test_plan':candidate['test_plan'],'critic':row['critique'],
            'admission':'allowed_for_exploration_not_effectiveness'}
        lessons.append({'fact_refs':[ref],'when_applicable':candidate['when_applicable'],
            'action':candidate['proposed_change'],'failure_condition':candidate['falsifier']})
    memory={'kind':'offline_team_method_memory_v1','built_at':built_at,
        'source_public_through':max(c['exit']['time'] for c in cases),
        'source_feedback_available_at':max(c['available_at'] for c in cases),
        'source_event_group_ids':sorted(set(source_groups)),
        'source_episode_hashes':sorted({d['episode_sha256'] for c in cases for d in c['decisions']}),
        'source_evidence_sha256':canonical_hash(rows),'lessons':lessons,
        'no_lesson_reason':None if lessons else 'No fact-checked, testable candidate passed the gate.',
        'cited_facts':cited,'status':'unvalidated_offline_fitted_method',
        'effectiveness_verified':False,'fine_tuning_admitted':False}
    require(set(c['event_group_id'] for c in cases)<=set(source_groups),'incomplete source groups')
    require(timestamp(built_at,'built')>=timestamp(memory['source_feedback_available_at'],'feedback'),'backdated memory')
    validate_lessons({'lessons':lessons,'no_lesson_reason':memory['no_lesson_reason']},cited)
    memory['memory_sha256']=canonical_hash(memory)
    return memory
