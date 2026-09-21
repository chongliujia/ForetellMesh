"""Reconcile an audited simulated profit, preserving the original run unchanged.

Constant probabilities are post-hoc mechanical controls, not new strategies,
new independent evaluation or evidence of deployable forecasting skill.
"""
import argparse
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.team_discovery_experiment import prepare
from foretellmesh.team_learning import require
from foretellmesh.team_learning_experiment import read_rows
from foretellmesh.team_portfolio import replay_portfolio
from foretellmesh.synthetic_sft import canonical_hash

D=Decimal


def audited(root):
    report=strict_json((root/'report.json').read_text());audit=strict_json((root/'audit.json').read_text())
    require(report['status']=='completed' and audit['status']=='passed'
        and audit['report_sha256']==sha256_file(root/'report.json'),'run not audited')
    for name,digest in report['artifact_hashes'].items():
        p=(root/name).resolve();require(p.is_relative_to(root.resolve()) and sha256_file(p)==digest,'changed run artifact')
    plan=strict_json((root/'plan.json').read_text())
    require(sha256_file(root/'plan.json')==report['plan_sha256'],'changed source plan')
    for name,digest in plan['source_hashes'].items():
        p=(root/name).resolve();require(p.is_relative_to(root.resolve()) and sha256_file(p)==digest,'changed source snapshot')
    return report


def diagnose(root,output):
    require(not output.exists(),'diagnosis already exists')
    report=audited(root);config=strict_json((root/'config.json').read_text())
    jobs,_,feed=prepare(config);episodes=read_rows(root/'episodes.jsonl')
    require(jobs==read_rows(root/'jobs.jsonl'),'source population changed')
    account=replay_portfolio(jobs,episodes,feed,config['execution_policy'])
    require(account==strict_json((root/'portfolio.json').read_text()),'original account does not reproduce')
    output.mkdir(parents=True)
    controls=[.1,.2,.3,.5]
    plan={'source_run':str(root),'report_sha256':sha256_file(root/'report.json'),
        'script_sha256':sha256_file(Path(__file__)),'source_outcomes_already_seen':True,
        'constant_probability_controls':controls,'preserved':'original decision schedule, measured latency, risk vetoes, execution costs and limits',
        'interpretation':'Post-hoc accounting/mechanism diagnosis only. No controls selected by their PnL and no default promotion.'}
    (output/'plan.json').write_text(json_text(plan))
    indexed={e['episode_id']:(j,e) for j,e in zip(jobs,episodes)}
    fills={r['order_id']:r for r in account['ledger'] if r['kind']=='fill'}
    orders={r['order_id']:r for r in account['ledger'] if r['kind']=='order'}
    rows=[]
    for settlement in account['ledger']:
        if settlement['kind']!='settle':continue
        oid=settlement['order_id'];fill=fills[oid];order=orders[oid];eid,mid=oid.rsplit(':',1);job,ep=indexed[eid]
        market=next(m for m in ep['context']['markets'] if m['market_id']==mid)
        prediction=next(f for f in ep['decision']['forecast']['forecasts'] if f['market_id']==mid)
        require(D(settlement['payout'])-D(fill['cost'])==D(settlement['net_pnl']),'settlement identity differs')
        require(D(fill['cost'])>=D(fill['fee']),'invalid fee accounting')
        label=job['labels'][mid];require(label['outcome']==settlement['outcome'] and label['resolution_time']==settlement['time'],'settlement label differs')
        title=market['input']['question'].split('description:',1)[0].removeprefix('q: title:').rstrip(', ').strip()
        studies=[r for r in ep['discovery']['investigations'] if mid in r['specification']['market_ids']]
        rows.append({'market_id':mid,'title':title,'order_id':oid,'episode_sha256':canonical_hash(ep),
            'decision_at':order['time'],'fill_at':fill['time'],'settled_at':settlement['time'],
            'market_yes_probability_at_observation':market['input']['market']['probability'],
            'model_yes_probability':prediction['probability'],'side':fill['side'],'estimated_edge':order['estimated_edge'],
            'actual_fill_side_price':fill['price'],'actual_reference_yes_price':fill['reference_yes_price'],
            'shares':fill['shares'],'cost_including_fee':fill['cost'],'fee_already_in_cost':fill['fee'],
            'payout':settlement['payout'],'net_pnl':settlement['net_pnl'],'risk':ep['decision']['risk'],
            'research_about_this_contract':[{'specification':s['specification'],'descriptive_test':s['descriptive_test'],
                'tool_result_sha256':s['tool_result']['result_sha256']} for s in studies],
            'research_is_placeholder':bool(studies) and all(s['specification']['claim']=='tentative relationship' for s in studies)})
    total_cost=sum((D(r['cost_including_fee']) for r in rows),D(0));payout=sum((D(r['payout']) for r in rows),D(0))
    pnl=sum((D(r['net_pnl']) for r in rows),D(0));fees=sum((D(r['fee_already_in_cost']) for r in rows),D(0))
    require(pnl==D(account['net_pnl'])==payout-total_cost and D(account['final_cash'])==D(account['initial_cash'])+pnl,'terminal identity differs')
    require(fees==D(account['fees_paid']) and total_cost==D(account['entry_turnover'])+fees,'fee/turnover identity differs')
    flat=[r for r in account['equity_curve'] if r['open_positions']==0 and D(r['reserved'])==0]
    peak=max(flat,key=lambda r:D(r['cash']))
    best=max(rows,key=lambda r:D(r['net_pnl']))
    probability_counts=Counter(str(f['probability']) for e in episodes if e['decision'] for f in e['decision']['forecast']['forecasts'])
    scenarios=[]
    for value in controls:
        altered=deepcopy(episodes)
        for e in altered:
            if e['decision']:
                for forecast in e['decision']['forecast']['forecasts']:forecast['probability']=value
        result=replay_portfolio(jobs,altered,feed,config['execution_policy'])
        name='constant_'+str(value).replace('.','_')+'.portfolio.json'
        (output/name).write_text(json_text(result))
        control_fills={r['market_id']:r for r in result['ledger'] if r['kind']=='fill'}
        original_fills={r['market_id']:r for r in fills.values()}
        exact_shared=[mid for mid in sorted(set(control_fills)&set(original_fills)) if control_fills[mid]==original_fills[mid]]
        scenarios.append({'probability':value,'final_cash':result['final_cash'],'net_pnl':result['net_pnl'],
            'identical_fill_record_count':len(exact_shared),'identical_fill_market_ids':exact_shared,
            'new_fill_market_ids':sorted(set(control_fills)-set(original_fills)),
            'omitted_fill_market_ids':sorted(set(original_fills)-set(control_fills)),
            'fills':result['filled_trades'],'fees':result['fees_paid'],
            'largest_winner_market_filled':best['market_id'] in {r['market_id'] for r in result['ledger'] if r['kind']=='fill'},
            'artifact_sha256':sha256_file(output/name)})
    result={'kind':'audited_profit_attribution_v1','plan_sha256':sha256_file(output/'plan.json'),'source_run':str(root),
        'source_report_sha256':sha256_file(root/'report.json'),'original_account_reproduced':True,'simulation_only':True,
        'account_id':str(root)+':portfolio','initial_cash':account['initial_cash'],'final_cash':account['final_cash'],
        'net_pnl':str(pnl),'cash_identity':{'initial':account['initial_cash'],'total_fill_cost_including_fees':str(total_cost),
            'total_settlement_payout':str(payout),'final':account['final_cash']},
        'fees_already_deducted':str(fees),'fees_excluded_pnl_at_unchanged_fills':str(pnl+fees),
        'winning_trades':sum(D(r['net_pnl'])>0 for r in rows),'losing_trades':sum(D(r['net_pnl'])<0 for r in rows),
        'winner_net_pnl':str(sum((D(r['net_pnl']) for r in rows if D(r['net_pnl'])>0),D(0))),
        'loser_net_pnl':str(sum((D(r['net_pnl']) for r in rows if D(r['net_pnl'])<0),D(0))),
        'largest_winner':best,'other_trades_net_pnl':str(pnl-D(best['net_pnl'])),
        'highest_flat_cash':peak,'drawdown_after_highest_flat_to_terminal':str(D(peak['cash'])-D(account['final_cash'])),
        'trades_in_settlement_order':rows,'forecast_probability_counts':dict(sorted(probability_counts.items(),key=lambda r:float(r[0]))),
        'posthoc_fixed_probability_controls':scenarios,
        'control_interpretation':'Accounting sensitivity of these known historical cases; not independent forecasts, profitable strategy discovery or a test of semantic research causality.',
        'new_model_calls':0,'fine_tuning_performed':False,'default_promoted':False,'final_test_opened':False}
    (output/'report.json').write_text(json_text(result))
    reference={'kind':'preserved_historical_reference_v1','source_run':str(root),'source_report_sha256':sha256_file(root/'report.json'),
        'source_config_sha256':sha256_file(root/'config.json'),'source_plan_sha256':sha256_file(root/'plan.json'),
        'source_model_manifest_sha256':sha256_file(root/'model_manifest.json'),
        'status':'profitable_learning_partition_candidate_not_validated_default','model_weights_frozen':True,
        'original_result':'100 -> 106.523346','independent_effectiveness_verified':False,'fine_tuning_admitted':False,
        'future_comparison_requires':'same fresh events, starting capital, evidence, costs and risk; original workflow and candidate workflow both frozen before evaluation'}
    (output/'preserved_reference.json').write_text(json_text(reference))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=diagnose(a.run,a.output)
    print(json_text({k:r[k] for k in ('initial_cash','final_cash','net_pnl','cash_identity','fees_already_deducted',
        'winning_trades','losing_trades','winner_net_pnl','loser_net_pnl','other_trades_net_pnl','highest_flat_cash','posthoc_fixed_probability_controls')}))
