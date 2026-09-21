"""Compare frozen discovery runs without treating exploration gains as validation."""
import argparse
from collections import Counter
from decimal import Decimal
from pathlib import Path
from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.metrics import score_predictions
from foretellmesh.team_learning import require
from foretellmesh.team_discovery import study_quality_issues
from foretellmesh.team_learning_experiment import read_rows


def summarize(root):
    report = strict_json((root/'report.json').read_text())
    audit = strict_json((root/'audit.json').read_text())
    require(report['status'] == 'completed' and audit['status'] == 'passed'
            and audit['report_sha256'] == sha256_file(root/'report.json'), 'unaudited discovery run')
    episodes = read_rows(root/'episodes.jsonl')
    ps=[]; qs=[]; ys=[]; studies=[]; errors=Counter(); latency=[]; retained=[]; invalid_specs=[]
    for ep in episodes:
        latency.append(ep['decision_seconds'])
        if ep['decision_error']: errors[ep['decision_error']] += 1
        forecasts = {} if ep['decision'] is None else {r['market_id']:r['probability'] for r in ep['decision']['forecast']['forecasts']}
        for m in ep['context']['markets']:
            mid = m['market_id']
            if mid in forecasts:
                ps.append(forecasts[mid]); qs.append(m['input']['market']['probability']); ys.append(ep['feedback']['outcomes'][mid])
        reviews={r['investigation_id']:r for r in (ep['discovery'].get('review') or {}).get('reviews',[])}
        for r in ep['discovery'].get('investigations',[]):
            issues=study_quality_issues(r['specification'])
            if issues: invalid_specs.append({'investigation_id':r['investigation_id'],'issues':issues})
            if (reviews.get(r['investigation_id'],{}).get('verdict') == 'retain_for_testing'
                    and r.get('descriptive_test',{}).get('measure') is not None and not issues):
                retained.append({'investigation_id':r['investigation_id'],'specification_sha256':r['specification_sha256'],
                    'kind':r['descriptive_test']['kind'],'requires_independent_test':True})
            if len(r['specification']['market_ids']) == 2:
                studies.append({'episode_id':ep['episode_id'],'specification':r['specification'],
                    'paired_daily_changes':r['tool_result']['paired_daily_changes'],
                    'pearson_change_correlation':r['tool_result']['pearson_change_correlation'],
                    'tool_result_sha256':r['tool_result']['result_sha256']})
    portfolio=strict_json((root/'portfolio.json').read_text())
    settlements=[r for r in portfolio['ledger'] if r['kind']=='settle']
    best=max(settlements,key=lambda r:Decimal(r['net_pnl']),default=None)
    concentration={'settled_trades':len(settlements),'fees_paid':portfolio['fees_paid'],
        'largest_trade_net_pnl':best['net_pnl'] if best else None,
        'largest_trade_market_id':best['market_id'] if best else None,
        'other_trades_net_pnl':str(sum((Decimal(r['net_pnl']) for r in settlements if r is not best),Decimal(0)))}
    return {'run':str(root),'report_sha256':sha256_file(root/'report.json'),
        'catalogue_contracts':len(read_rows(root/'catalogue.jsonl')),'pnl_concentration':concentration,
        'statistics':report['statistics'],'goal':report['goal'],
        'matched_forecast_scores':score_predictions(ps,ys),'matched_market_scores':score_predictions(qs,ys),
        'matched_constant_half_scores':score_predictions([.5]*len(ys),ys),
        'decision_failures':dict(errors),'cross_contract_studies':studies,
        'retained_studies_with_numeric_measure':retained,'studies_failing_concrete_hypothesis_gate':invalid_specs,
        'cross_studies_with_nonzero_paired_changes':sum(s['paired_daily_changes']>0 for s in studies),
        'cross_studies_with_computable_correlation':sum(s['pearson_change_correlation'] is not None for s in studies),
        'mean_decision_seconds':sum(latency)/len(latency),
        'cash_baseline_terminal_cash':'100','independent_effectiveness_verified':False,
        'fine_tuning_admitted':False}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'comparison output exists')
    result={'kind':'frozen_team_discovery_comparison_v1','analysis_source_sha256':sha256_file(Path(__file__)),'runs':[summarize(r) for r in a.runs],
        'scope':'Training exploration only. Different cohorts or observation counts are not a paired profitability comparison.',
        'independent_validation_performed':False,'fine_tuning_performed':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json_text(result))
    print(json_text({k:v for k,v in result.items() if k != 'runs'}))
