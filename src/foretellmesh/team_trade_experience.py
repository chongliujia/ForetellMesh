"""Every filled position is experience, regardless of profit; lessons are hypotheses."""
from copy import deepcopy
from datetime import timedelta

from .paper_trading import decimal
from .schema import fields, iso, timestamp
from .synthetic_sft import canonical_hash
from .team_learning import require


REFLECT = ('Review this completed simulated trade as one team learning experience. Winning and losing trades '
    'are equally eligible. Separate forecast evidence, relationship hypothesis, timing, execution costs and '
    'luck; a win does not establish a good method and a loss does not refute a probability. '
    'Only use supplied facts. Do not invent a causal diagnosis or alternative profitable fill. '
    'Return exactly {"experience_id":ID,"assessment":STRING,"lesson":null or '
    '{"when_applicable":STRING,"proposed_change":STRING,"falsifier":STRING,"next_test":STRING}}. '
    'Be concise: each text string should be at most 160 characters. Return ONLY the object; never echo input. '
    'A null lesson is valid when this case supports no specific change. '
    'Any lesson is an unverified procedural hypothesis, never an instruction to fine-tune or a verified strategy.')


def trade_experiences(episodes, account, labels, policy):
    """Build one auditable case per entry fill, including open and zero-PnL positions.

    May archive future terminal results, but available_at must gate any learning.
    Unsold positions retain outcome/result=None, not an invented loss or win.
    """
    entries={r['order_id']:r for r in account['ledger'] if r['kind']=='fill'}
    closes={(r['position_id'] if r['kind']=='sell_fill' else r['order_id']):r
            for r in account['ledger'] if r['kind'] in ('sell_fill','settle')}
    cases=[]
    for oid,entry in entries.items():
        close=closes.get(oid);mid=entry['market_id'];decisions=[];group=None;entry_forecast=None
        for ep in episodes:
            context=ep['context'];decision=ep['decision']
            ready=timestamp(context['observation_time'],'observation')+timedelta(seconds=float(ep['decision_seconds']))
            is_entry=oid==context['episode_id']+':'+mid
            if not is_entry and (ready<timestamp(entry['time'],'entry') or
                                close and ready>timestamp(close['time'],'close')):continue
            if not any(m['market_id']==mid for m in context['markets']):continue
            if is_entry:
                group=next(m['event_group_id'] for m in context['markets'] if m['market_id']==mid)
                if decision:
                    entry_forecast=next(f['probability'] for f in decision['forecast']['forecasts'] if f['market_id']==mid)
            decisions.append({'episode_id':context['episode_id'],'episode_sha256':canonical_hash(ep),
                'observation_time':context['observation_time'],'ready_at':iso(ready),
                'decision':deepcopy(decision),'decision_error':ep['decision_error'],
                'investigations':deepcopy(ep.get('discovery',{}).get('investigations',[]))})
        require(group is not None,'entry experience missing source decision')
        available=None;result=None
        if close:
            available=timestamp(close['time'],'close')
            if close['kind']=='settle':available=max(available,timestamp(labels[mid]['available_at'],'label availability'))
            pnl=decimal(close['net_pnl'])
            result={'net_pnl':str(pnl),'result_class':'profit' if pnl>0 else 'loss' if pnl<0 else 'flat',
                'total_fees':str(decimal(entry['fee'])+decimal(close.get('fee','0'))),
                'holding_seconds':(timestamp(close['time'],'close')-timestamp(entry['time'],'entry')).total_seconds(),
                'exit_type':close['kind'],
                'event_outcome':close['outcome'] if close['kind']=='settle' else None,
                'entry_brier':None if close['kind']!='settle' or entry_forecast is None else
                    float((decimal(entry_forecast)-decimal(close['outcome']))**2)}
        case={'kind':'team_trade_experience_v1','experience_id':oid,'market_id':mid,'event_group_id':group,
            'status':'closed' if close else 'open','available_at':iso(available) if available else None,
            'entry':deepcopy(entry),'exit':deepcopy(close),'result':result,'decisions':decisions,
            'execution_policy':deepcopy(policy),'effectiveness_verified':False,'fine_tuning_admitted':False}
        case['experience_sha256']=canonical_hash(case);cases.append(case)
    require(len(cases)==len(entries),'trade experience coverage differs')
    return cases


def validate_reflection(value, case):
    fields(value,{'experience_id','assessment','lesson'},'trade reflection')
    require(value['experience_id']==case['experience_id'],'reflection experience differs')
    values=[value['assessment']]
    if value['lesson'] is not None:
        fields(value['lesson'],{'when_applicable','proposed_change','falsifier','next_test'},'trade lesson')
        values.extend(value['lesson'].values())
    require(all(isinstance(v,str) and 1<=len(v.strip())<=600 for v in values),'invalid trade reflection text')
    return deepcopy(value)


def reflection_context(case):
    """Compact deterministic evidence summaries; full cases remain in the archive."""
    from .team_discovery import compact_tool
    value={k:deepcopy(v) for k,v in case.items() if k!='decisions'}
    value['decision_summaries']=[]
    for row in case['decisions']:
        decision=row['decision'];summary={k:v for k,v in row.items() if k not in ('decision','investigations')}
        summary['forecast']=None if not decision else next((f for f in decision['forecast']['forecasts']
            if f['market_id']==case['market_id']),None)
        summary['action']=None if not decision else next((a for a in decision.get('lifecycle',{}).get('actions',[])
            if a['market_id']==case['market_id']),None)
        summary['risk_veto']=bool(decision and case['market_id'] in decision.get('risk',{}).get('veto_markets',[]))
        summary['investigation_refs']=[{'investigation_id':r['investigation_id'],
            'specification_sha256':r['specification_sha256'],'result_sha256':r['tool_result']['result_sha256'],
            'descriptive_test':r['descriptive_test']} for r in row['investigations']]
        value['decision_summaries'].append(summary)
    # Detailed evidence at entry and at the last review; retain the entire
    # probability/action/test trajectory without repeating long tool payloads.
    selected=sorted({0,len(case['decisions'])-1}) if case['decisions'] else []
    value['detailed_review_indices']=selected
    value['evidence_summaries']=[{'review_index':i,'research':(case['decisions'][i]['decision'] or {}).get('research'),
        'risk':(case['decisions'][i]['decision'] or {}).get('risk'),
        'investigations':[{'specification':r['specification'],'descriptive_test':r['descriptive_test'],
            'tool_result':compact_tool(r['tool_result'])} for r in case['decisions'][i]['investigations']]}
        for i in selected]
    value['full_decisions_archived']=True
    value['summary_scope']='All decisions retain probabilities, actions and test summaries; detailed evidence is shown for entry and last review only. Other full evidence remains archived, not silently presented as read.'
    return value


def learn_available(runner, cases, cutoff, *, enabled):
    """Visit every newly available closed case; no PnL-based selection or promotion."""
    if not enabled:return
    at=timestamp(cutoff,'learning cutoff')
    for case in cases:
        if case['available_at'] is None or timestamp(case['available_at'],'availability')>at:continue
        if case['experience_id'] in runner.trade_reflections:continue
        require(case['experience_sha256']==canonical_hash({k:v for k,v in case.items() if k!='experience_sha256'}),
                'trade experience changed')
        from .team_experience_gate import review_case
        runner.trade_reflections[case['experience_id']]=review_case(runner,case,cutoff)


def validate_online_lessons(rows, cutoff):
    from .team_experience_gate import validate_admitted, validate_method_views
    if rows and rows[0].get('kind')=='fact_gated_method_view_v1':
        return validate_method_views(rows,cutoff)
    return validate_admitted(rows,cutoff)
