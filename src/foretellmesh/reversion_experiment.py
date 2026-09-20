"""Frozen, CPU-only mean-reversion/weekly-activity development replay."""
import argparse
from collections import defaultdict
from datetime import timedelta,datetime,timezone
from pathlib import Path
import shutil
import sys
import time

from .data import sha256_file,strict_json
from .evaluation import code_provenance,json_text
from .market_development import read_inputs,rows
from .mean_reversion import MarketWindow,HistoricalBars,validate_config,features,simulate
from .paper_trading import Settlement
from .schema import ValidationError,fields,timestamp,iso,binary_outcome
from .sft_data import jsonl
from .synthetic_sft import canonical_hash

ARMS={'mean_reversion_only':(True,False),'weekly_liquidity_control':(False,True),'mean_reversion_weekly':(True,True)}


def config(path):
    c=strict_json(path.read_text())
    fixed={'schema_version':'1','partition':'validation','calendar':'min_to_max_approved_validation_observation',
           'universe':'admitted_validation_contracts_initialized_to_last_approved_observation',
           'purpose':'short_horizon_price_reversion_with_rolling_seven_day_fill_requirement',
           'arms':list(ARMS),'final_test_scored':False,'training':False,'model_calls':0,'real_orders_sent':0}
    fields(c,set(fixed)|{'run_name','dataset_report_sha256','policy','scenarios'},'reversion experiment')
    if any(type(c[k]) is not type(v) or c[k]!=v for k,v in fixed.items()):raise ValidationError('unsupported replay scope')
    validate_config(c['policy'])
    if set(c['scenarios'])!={'frictionless_reference','cost_assumption','cost_stress'}:raise ValidationError('missing cost controls')
    for costs in c['scenarios'].values():
        fields(costs,{'entry_price_premium','fee_fraction'},'hypothetical costs')
    return c


def prepare(dataset,trade_store,c):
    report,inputs,membership=read_inputs(dataset,c)
    grouped=defaultdict(list)
    for r in inputs:
        m=membership[r['sample_id']]
        if not m['event_id'].startswith('polymarket:'):raise ValidationError('non-Polymarket market')
        mid=m['event_id'].split(':',1)[1]
        if not mid.isdigit():raise ValidationError('invalid market identity')
        grouped[mid].append(r)
    proof={};bindings={'dataset_report_sha256':sha256_file(dataset/'report.json')}
    for relative,digest in report['artifact_hashes'].items():
        if relative.startswith('audit/') and relative.endswith('/contracts.jsonl'):
            path=dataset/relative
            if sha256_file(path)!=digest:raise ValidationError('contract audit changed')
            bindings[relative]=digest
            for r in rows(path):
                if r['market_id'] not in grouped:continue
                if not r.get('proof') or r['proof_errors'] or r['proof']['creator_update_count']!=0:raise ValidationError('invalid admitted contract proof')
                # Explicit projection: outcome and resolution metadata are not
                # passed to feature generation or decision policy.
                pr={k:r['proof'][k] for k in ('initialized_at','rule_valid_through','historical_question')}
                pr['event_group_id']=r['event_group_id']
                if r['market_id'] in proof and proof[r['market_id']]!=pr:raise ValidationError('conflicting rule audits')
                proof[r['market_id']]=pr
    if set(proof)!=set(grouped):raise ValidationError('missing admitted contract initialization')
    windows=[];catalog=[]
    for mid,rr in sorted(grouped.items()):
        pr=proof[mid];retire=max(timestamp(r['input']['observation_time'],'approved observation') for r in rr)
        initial=timestamp(pr['initialized_at'],'rule initialization');group=pr['event_group_id']
        if initial>=retire or timestamp(pr['rule_valid_through'],'rule coverage')<retire:raise ValidationError('invalid rule coverage')
        if any(r['input']['question']!=pr['historical_question'] or membership[r['sample_id']]['event_group_id']!=group for r in rr):
            raise ValidationError('contract rules or event grouping differ')
        windows.append(MarketWindow(mid,group,initial,retire))
        catalog.append({'market_id':mid,'event_group_id':group,'initialized_at':iso(initial),'retire_at':iso(retire),
                        'question_sha256':canonical_hash(pr['historical_question'])})
    start=min(timestamp(r['input']['observation_time'],'calendar start') for r in inputs)
    end=max(timestamp(r['input']['observation_time'],'calendar end') for r in inputs)
    tr=strict_json((trade_store/'report.json').read_text());counts=tr.get('counts',{})
    digest=sha256_file(trade_store/'prices.sqlite')
    if (tr.get('kind')!='pma_historical_trade_store' or not counts.get('unique_valid_selected_trades')
            or counts['unique_valid_selected_trades']!=counts['trades_with_block_time']
            or counts['required_distinct_blocks']!=counts['matched_block_times']
            or digest!=tr['artifact_hashes']['prices.sqlite']):raise ValidationError('invalid trade-store provenance')
    bindings.update(trade_report_sha256=sha256_file(trade_store/'report.json'),trade_store_sha256=digest)
    return windows,catalog,start,end,bindings,inputs,membership


def build_states(feed,windows,start,end,policy):
    ticks=[];t=start
    while t<=end:ticks.append(t);t+=timedelta(seconds=policy['step_seconds'])
    if ticks[-1]!=end:ticks.append(end)
    states=[]
    for t in ticks:
        for m in windows:
            if m.initialized_at<=t and t+timedelta(seconds=policy['mandatory_holding_seconds'])<=min(end,m.retire_at):
                states.append({'market_id':m.market_id,'observation_time':iso(t),'features':features(feed,m.market_id,t,policy)})
    return states


def load_settlements(dataset,inputs,membership,windows):
    labels=rows(dataset/'partitions/validation.labels.jsonl')
    by_sample={r['sample_id']:r['label'] for r in labels}
    if len(by_sample)!=len(labels) or set(by_sample)!={r['sample_id'] for r in inputs}:raise ValidationError('validation label identity mismatch')
    values={}
    for r in inputs:
        mid=membership[r['sample_id']]['event_id'].split(':',1)[1];label=by_sample[r['sample_id']]
        s=Settlement(mid,timestamp(label['resolution_time'],'audited CTF settlement'),binary_outcome(label['outcome']))
        if mid in values and values[mid]!=s:raise ValidationError('conflicting contract labels')
        values[mid]=s
    if set(values)!={m.market_id for m in windows}:raise ValidationError('settlement universe differs')
    return list(values.values())


def materialize(dataset,trade_store,c):
    windows,catalog,start,end,bindings,inputs,membership=prepare(dataset,trade_store,c)
    feed=HistoricalBars.from_store(trade_store/'prices.sqlite',{m.market_id for m in windows},
        start-timedelta(hours=c['policy']['lookback_hours']+3),end+timedelta(seconds=c['policy']['fill_window_seconds']))
    states=build_states(feed,windows,start,end,c['policy'])
    return windows,catalog,start,end,bindings,inputs,membership,feed,states


def calculate(dataset,c,windows,start,end,inputs,membership,feed,states):
    settlements=load_settlements(dataset,inputs,membership,windows)
    lookup={(s['market_id'],s['observation_time']):s['features'] for s in states}
    return {scenario:{arm:simulate(windows,settlements,feed,start,end,c['policy'],costs,natural=natural,weekly=weekly,states=lookup)
        for arm,(natural,weekly) in ARMS.items()} for scenario,costs in c['scenarios'].items()}


def run(dataset,trade_store,config_path,output):
    if output.exists():raise ValidationError('output already exists')
    c=config(config_path);started=time.perf_counter()
    windows,catalog,start,end,bindings,inputs,membership,feed,states=materialize(dataset,trade_store,c)
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(config_path,output/'config.json')
    (output/'catalog.jsonl').write_text(jsonl(catalog));(output/'features.jsonl').write_text(jsonl(states))
    plan={'frozen_at':datetime.now(timezone.utc).isoformat(),'bindings':bindings,'config':c,
        'source_paths':{k:str(v.resolve()) for k,v in {'dataset':dataset,'trade_store':trade_store}.items()},
        'start':iso(start),'end':iso(end),'code':code_provenance(),
        'python_version':sys.version,
        'artifact_hashes':{str(p.relative_to(output)):sha256_file(p) for p in output.rglob('*') if p.is_file()}}
    (output/'plan.json').write_text(json_text(plan))
    print(f'Frozen {len(windows)} markets, {len(states)} as-of feature states; parsing labels only for the settlement environment.',flush=True)
    rr=calculate(dataset,c,windows,start,end,inputs,membership,feed,states)
    summary={'status':'completed','plan_sha256':sha256_file(output/'plan.json'),'start':iso(start),'end':iso(end),
        'market_count':len(windows),'event_groups':len({m.event_group_id for m in windows}),'feature_states':len(states),
        'native_rows_loaded':sum(n for bars in feed.data.values() for _,_,n in bars),'scenarios':{},'artifact_hashes':{},
        'primary_strategy':'mean_reversion_weekly','training':False,'model_calls':0,'final_test_scored':False,'real_orders_sent':0,
        'limitations':['Development universe of already audited/resolved contracts, not a point-in-time complete market universe.',
            'Expanded earlier observations use verified initialized immutable rules; retire time is the predeclared last approved observation, not future outcome.',
            'Mean reversion targets a future price, not a calibrated resolution probability; regime shifts may prevent reversion.',
            'Historical next-block full fills, synthetic No complements and hypothetical entry/exit fees/slippage; no book depth or queue.',
            'Cadence counts actual simulated buy/sell fills only. Missing prints, capital or eligibility can violate the requirement and are reported.',
            'Policy v2 allows bounded $1 weekly participation at tail prices, choosing the higher-price side whose all-in cost is below $1 per share; this can still have negative expected value.',
            'End-of-period liquidation proxy and terminal cash after any outstanding settlements are distinct.',
            'No LLM is called in this deterministic quant-policy test; it does not establish multi-agent or fine-tuning improvement.']}
    for scenario,arms in rr.items():
        summary['scenarios'][scenario]={};(output/scenario).mkdir()
        for arm,value in arms.items():
            for field in ('ledger','equity_curve','decisions'):
                path=output/scenario/f'{arm}.{field}.jsonl';path.write_text(jsonl(value[field]))
                summary['artifact_hashes'][str(path.relative_to(output))]=sha256_file(path)
            summary['scenarios'][scenario][arm]={k:v for k,v in value.items() if k not in ('ledger','equity_curve','decisions')}
    summary['elapsed_seconds']=time.perf_counter()-started
    (output/'report.json').write_text(json_text(summary))
    return summary


def audit(root):
    c=config(root/'config.json');p=strict_json((root/'plan.json').read_text());r=strict_json((root/'report.json').read_text())
    if r['status']!='completed' or r['plan_sha256']!=sha256_file(root/'plan.json') or c!=p['config']:raise ValidationError('run plan changed')
    for name,digest in {**p['artifact_hashes'],**r['artifact_hashes']}.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root/name)!=digest:raise ValidationError('reversion artifact changed')
    paths={k:Path(v) for k,v in p['source_paths'].items()}
    windows,catalog,start,end,bindings,inputs,membership,feed,states=materialize(**paths,c=c)
    if (catalog!=rows(root/'catalog.jsonl') or states!=rows(root/'features.jsonl') or bindings!=p['bindings']
            or iso(start)!=p['start'] or iso(end)!=p['end']):raise ValidationError('as-of features/universe do not reproduce')
    rr=calculate(paths['dataset'],c,windows,start,end,inputs,membership,feed,states)
    if set(rr)!=set(r['scenarios']):raise ValidationError('scenario mismatch')
    for scenario,arms in rr.items():
        if set(arms)!=set(r['scenarios'][scenario]):raise ValidationError('strategy mismatch')
        for arm,value in arms.items():
            if {k:v for k,v in value.items() if k not in ('ledger','equity_curve','decisions')}!=r['scenarios'][scenario][arm]:
                raise ValidationError('metrics fail exact reproduction')
            for field in ('ledger','equity_curve','decisions'):
                if rows(root/scenario/f'{arm}.{field}.jsonl')!=value[field]:raise ValidationError('ledger/decisions fail exact reproduction')
    return {'status':'passed','report_sha256':sha256_file(root/'report.json'),'features_exactly_reproduced':True,
        'ledgers_exactly_reproduced':True,'cadence_exactly_reproduced':True,'training':False,'real_orders_sent':0,'final_test_scored':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    a=subs.add_parser('run')
    for name in ('dataset','trade-store','config','output'):a.add_argument('--'+name,type=Path,required=True)
    subs.add_parser('audit').add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if args.command=='audit':result=audit(args.run)
    else:
        kw=vars(args);kw.pop('command');kw['config_path']=kw.pop('config');result=run(**kw)
    print(json_text(result))


if __name__=='__main__':main()
