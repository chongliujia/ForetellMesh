"""Summarize an audited paired memory pilot without selecting a new candidate."""
import argparse
from collections import Counter
from pathlib import Path
import statistics
from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.team_learning import require
from foretellmesh.team_learning_experiment import read_rows
from foretellmesh.metrics import score_predictions
from foretellmesh.synthetic_sft import canonical_hash


def summarize(root):
    report=strict_json((root/'report.json').read_text());audit=strict_json((root/'audit.json').read_text())
    require(audit['status']=='passed' and audit['report_sha256']==sha256_file(root/'report.json'),'unaudited result')
    memory=strict_json((root/'memory.json').read_text());details={}
    for name in ('learning','control','memory'):
        eps=read_rows(root/(name+'.episodes.jsonl'));portfolio=strict_json((root/(name+'.portfolio.json')).read_text())
        calls=[c for e in eps for c in e['calls']];ps=[];ys=[];qs=[];eligible=0
        for e in eps:
            eligible+=len(e['context']['markets'])
            # Forecast diagnostic independent of downstream risk validity. Do not count failed forecast requests.
            valid={canonical_hash(c['request']) for c in e['calls'] if c['request']['agent']=='forecast' and c['error'] is None}
            indexed={r['market_id']:r for r in e['readouts'] if r['request_sha256'] in valid}
            for m in e['context']['markets']:
                mid=m['market_id'];ps.append(indexed[mid]['probability'] if mid in indexed else None)
                ys.append(e['feedback']['outcomes'][mid]);qs.append(m['input']['market']['probability'] if mid in indexed else None)
        details[name]={'episodes':len(eps),'completed_decisions':sum(e['decision'] is not None for e in eps),
            'final_cash':portfolio['final_cash'],'filled_trades':sum(r['kind']=='fill' for r in portfolio['ledger']),
            'fees':str(sum(__import__('decimal').Decimal(r['fee']) for r in portfolio['ledger'] if r['kind']=='fill')),
            'drawdown_usd':portfolio['sampled_equity_proxy_max_drawdown_usd'],
            'decision_seconds_mean':statistics.mean(e['decision_seconds'] for e in eps),
            'model_calls':len(calls),'invalid_calls':sum(c['error'] is not None for c in calls),
            'decision_errors':[{'episode_id':e['episode_id'],'error':e['decision_error']} for e in eps if e['decision_error']],
            'errors_by_role':dict(Counter(c['request']['agent'] for c in calls if c['error'] is not None)),
            'tools':dict(Counter(r['specification']['tool'] for e in eps for r in e['discovery']['investigations'])),
            'forecast_before_risk_scores':score_predictions(ps,ys),'same_coverage_market_scores':score_predictions(qs,ys),
            'hypothesis_review_verdicts':dict(Counter(r['verdict'] for e in eps for r in (e['discovery']['review'] or {}).get('reviews',[])))}
    return {'report_sha256':audit['report_sha256'],'script_sha256':sha256_file(Path(__file__)),
        'memory_sha256':memory['memory_sha256'],'lessons':memory['lessons'],'no_lesson_reason':memory['no_lesson_reason'],
        'details':details,'paired_comparison':report['comparison'],'memory_injected_calls':audit['memory_injected_decision_calls'],
        'peak_allocated_gib':report['peak_allocated_bytes']/2**30,'peak_reserved_gib':report['peak_reserved_bytes']/2**30,
        'fine_tuning_admitted':False,'effectiveness_verified':False,'final_test_opened':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=summarize(a.run);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json_text(result))
    print(json_text({k:v for k,v in result.items() if k!='paired_comparison'}))
