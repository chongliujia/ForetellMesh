"""Post-hoc fixed-duration order diagnostic; original evaluation is immutable."""
import argparse
from copy import deepcopy
from pathlib import Path
from collections import Counter
from foretellmesh.data import sha256_file,strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.team_learning import require
from foretellmesh.team_learning_experiment import read_rows
from foretellmesh.team_portfolio import replay_portfolio
from foretellmesh.team_reference_recheck import prepare,controls


def diagnose(root,output):
    require(not output.exists(),'duration diagnostic exists')
    report=strict_json((root/'report.json').read_text());audit=strict_json((root/'audit.json').read_text())
    require(report['status']=='completed' and audit['status']=='passed' and audit['report_sha256']==sha256_file(root/'report.json'),'unaudited source')
    for name,digest in report['artifact_hashes'].items():require(sha256_file(root/name)==digest,'changed artifact')
    config=strict_json((root/'config.json').read_text());jobs,_,feed,_,_=prepare(config)
    episodes={n:read_rows(root/(n+'.episodes.jsonl')) for n in ('reference','candidate')}
    hours=(1,4,24);output.mkdir(parents=True)
    plan={'source_report_sha256':sha256_file(root/'report.json'),'script_sha256':sha256_file(Path(__file__)),
        'order_lifetimes_hours':list(hours),'posthoc_after_evaluation':True,'new_model_calls':0,
        'fixed':'original inputs, measured model latency, forecast probabilities, risk vetoes, premiums, fees and capital limits',
        'only_change':'pending-order lifetime; new prices must still meet the original after-cost edge at fill',
        'limitation':'Frozen forecast is not refreshed during the longer order lifetime; this is a sensitivity check, not a validated deployment policy.',
        'promotion_allowed':False}
    (output/'plan.json').write_text(json_text(plan));rows=[]
    for h in hours:
        scenario=deepcopy(config);scenario['execution_policy']['fill_window_seconds']=h*3600
        accounts={n:replay_portfolio(jobs,eps,feed,scenario['execution_policy']) for n,eps in episodes.items()}
        accounts.update(controls(jobs,feed,scenario))
        for name,account in accounts.items():
            if h==1:require(account==strict_json((root/(name+'.portfolio.json')).read_text()),'primary result not preserved')
            file=f'{h}h.{name}.portfolio.json';(output/file).write_text(json_text(account))
            rows.append({'hours':h,'arm':name,'orders':account['orders'],'filled_trades':account['filled_trades'],
                'final_cash':account['final_cash'],'net_pnl':account['net_pnl'],'fees':account['fees_paid'],
                'drawdown_usd':account['sampled_equity_proxy_max_drawdown_usd'],
                'cancel_reasons':dict(Counter(r['reason'] for r in account['ledger'] if r['kind']=='cancel')),
                'artifact':file,'artifact_sha256':sha256_file(output/file)})
    result={'kind':'fixed_order_duration_diagnostic_v1','plan_sha256':sha256_file(output/'plan.json'),
        'rows':rows,'original_primary_reproduced':True,'independent_validation_claim':False,'fine_tuning_admitted':False}
    (output/'report.json').write_text(json_text(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=diagnose(a.run,a.output);print(json_text(r))
