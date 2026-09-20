"""Frozen Base-only market expert ablation and USD 100 historical simulation."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import math
from pathlib import Path
import shutil
import time

from .agent_baseline_data import input_context
from .agent_runtime import agent_instruction
from .capabilities import load_capabilities
from .data import sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .lora_probe import verify_model_manifest
from .market_development import read_inputs, rows
from .market_experts import build_features, enrich
from .metrics import score_predictions
from .offline_base import audit as audit_base, make_runner, replay
from .paper_backtest import load_config as load_paper
from .paper_trading import Signal, Settlement, TradePrintFeed, decimal, simulate
from .peft_runtime import SharedPeftExecutor, PeftTextBackend, render_agent_prompt
from .schema import ValidationError, fields, parse_record, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash

ARMS = {'features_only':'market_control', 'quant':'market_quant_forecast',
        'game':'market_game_forecast', 'quant_game':'market_quant_game_forecast'}


def load_config(path):
    c = strict_json(path.read_text())
    fixed = {'schema_version':'1', 'model':'Qwen/Qwen3-8B-Base',
        'model_revision':'49e3418fbbbca6ecbdf9608b4d22e5a407081db4', 'partition':'validation',
        'selection':'earliest_observation_then_closest_to_half_per_event_group',
        'arms':ARMS, 'output_protocol':'grounded_json_v3', 'response_transport':'single_json_fence',
        'max_context_tokens':8192, 'max_new_tokens':768, 'max_repairs':1, 'mode':'base',
        'engine':'langgraph', 'do_sample':False, 'base_dtype':'bfloat16', 'attention':'sdpa',
        'training':False, 'load_adapters':False, 'default_promotion':False}
    if 'stop_on_json_object' in c:
        fixed['stop_on_json_object'] = True
    fields(c, set(fixed)|{'run_name','seed','dataset_report_sha256'}, 'market expert experiment')
    if (any(type(c[k]) is not type(v) or c[k]!=v for k,v in fixed.items())
            or type(c['seed']) is not int or not 0<=c['seed']<2**32):
        raise ValidationError('unsupported expert experiment scope')
    return c


def select(inputs, membership):
    grouped = {}
    for row in inputs:
        p=row['input'];sid=row['sample_id'];m=membership[sid]
        if m['split']!='validation' or p['market'] is None:raise ValidationError('invalid selection population')
        key=(timestamp(p['observation_time'],'selection time'), abs(p['market']['probability']-.5), sid)
        group=m['event_group_id']
        if group not in grouped or key<grouped[group][0]:grouped[group]=(key,row)
    return [r for _,r in sorted(grouped.values(), key=lambda pair:pair[0])]


def sources(dataset, trade_store, baseline, c):
    manifest,inputs,membership=read_inputs(dataset,c)
    audited=audit_base(baseline)
    baseline_inputs={r['sample_id']:r for r in rows(baseline/'inputs.jsonl')}
    if baseline_inputs!={r['sample_id']:r for r in inputs}:raise ValidationError('cached Base population differs')
    original_config=strict_json((baseline/'config.json').read_text())
    if original_config['dataset_report_sha256']!=c['dataset_report_sha256']:raise ValidationError('cached Base dataset differs')
    tr=strict_json((trade_store/'report.json').read_text());n=tr.get('counts',{})
    db_hash=sha256_file(trade_store/'prices.sqlite')
    if (tr.get('kind')!='pma_historical_trade_store' or n.get('unique_valid_selected_trades',0)<=0
            or n.get('unique_valid_selected_trades')!=n.get('trades_with_block_time')
            or n.get('required_distinct_blocks')!=n.get('matched_block_times')
            or db_hash!=tr['artifact_hashes']['prices.sqlite']):raise ValidationError('invalid trade time provenance')
    binding={'dataset_report_sha256':sha256_file(dataset/'report.json'), 'trade_store_sha256':db_hash,
             'trade_report_sha256':sha256_file(trade_store/'report.json'), 'baseline_report_sha256':audited['report_sha256']}
    return select(inputs,membership), membership, binding


def prepare(dataset, trade_store, baseline, c):
    selected,membership,binding=sources(dataset,trade_store,baseline,c)
    feed=TradePrintFeed(trade_store/'prices.sqlite');enriched=[];audits=[]
    try:
        for row in selected:
            started=time.perf_counter();sid=row['sample_id'];p=row['input'];event=membership[sid]['event_id']
            if not event.startswith('polymarket:') or not event.split(':',1)[1].isdigit():raise ValidationError('unsupported market identity')
            market=event.split(':',1)[1];t=timestamp(p['observation_time'],'feature cutoff')
            q=feed.latest(market,t)
            if q is None or abs(q[0]-decimal(p['market']['probability']))>decimal('1e-12') or q[1]!=timestamp(p['market']['observed_at'],'reference time'):
                raise ValidationError('feature reference differs from dataset market price')
            feature,proof=build_features(feed,market,t);item=enrich(row,feature);input_context(item['input'])
            proof.update(sample_id=sid,seconds=time.perf_counter()-started)
            enriched.append(item);audits.append(proof)
    finally:feed.close()
    return selected,enriched,audits,binding


def jobs(inputs):
    result=[];names=list(ARMS)
    for i,row in enumerate(inputs):
        order=names[i%len(names):]+names[:i%len(names)]
        result.extend({'sample_id':row['sample_id'],'arm':arm} for arm in order)
    return result


def raw_replay(root, c):
    agents,_=load_capabilities(root/'agent_config.json');inputs=rows(root/'inputs.jsonl');results=rows(root/'results.jsonl')
    if [{'sample_id':r['sample_id'],'arm':r['arm']} for r in results]!=jobs(inputs):raise ValidationError('expert job population changed')
    indexed={r['sample_id']:r for r in inputs}
    for r in results:
        rc={**c,'workflow':ARMS[r['arm']]}
        replay([indexed[r['sample_id']]],[r],agents,rc)
    return results


def run(dataset,trade_store,baseline,config_path,agent_config,model_manifest,paper_config,output):
    if output.exists():raise ValidationError('expert output already exists')
    c=load_config(config_path);agents,_=load_capabilities(agent_config)
    raw,inputs,feature_audits,binding=prepare(dataset,trade_store,baseline,c)
    load_paper(paper_config)
    if sha256_file(paper_config)!=sha256_file(baseline/'paper_config.json'):raise ValidationError('trading policy must match original Base experiment')
    model_path,model_hash=verify_model_manifest(model_manifest,c)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValidationError('BF16 CUDA required')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    for row in inputs:
        for role in ('research','market_quant','game_theory','forecast'):
            p=render_agent_prompt({'instruction':agent_instruction(role,c['output_protocol']),'input':row['input'],'upstream':{}})
            if len(tokenizer.encode(p,add_special_tokens=False))+4*c['max_new_tokens']>c['max_context_tokens']:
                raise ValidationError('expert preflight context budget exceeded')
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    for path,name in [(config_path,'config.json'),(agent_config,'agent_config.json'),(paper_config,'paper_config.json')]:shutil.copyfile(path,output/name)
    for name,value in [('inputs.jsonl',inputs),('original_inputs.jsonl',raw),('feature_audits.jsonl',feature_audits)]:
        (output/name).write_text(jsonl(value))
    plan={'frozen_at':datetime.now(timezone.utc).isoformat(), 'bindings':binding,'model_manifest_sha256':model_hash,
          'source_paths':{k:str(v.resolve()) for k,v in {'dataset':dataset,'trade_store':trade_store,'baseline':baseline}.items()},
          'config':c,'jobs':jobs(inputs),'code':code_provenance(),
          'artifact_hashes':{str(p.relative_to(output)):sha256_file(p) for p in output.rglob('*') if p.is_file()}}
    (output/'plan.json').write_text(json_text(plan))
    report={'status':'running','plan_sha256':sha256_file(output/'plan.json'),'started_at':datetime.now(timezone.utc).isoformat(),
        'planned_jobs':len(plan['jobs']),'completed_jobs':0,'training':False,'loaded_adapters':[],
        'packages':{p:version(p) for p in ('torch','transformers','langgraph')},'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda}
    def save():
        p=output/'generation_report.tmp';p.write_text(json_text(report));p.replace(output/'generation_report.json')
    save();by_input={r['sample_id']:r for r in inputs}
    try:
        set_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        model.gradient_checkpointing_disable();model.config.use_cache=True;executor=SharedPeftExecutor(model)
        if executor.available_adapters:raise ValidationError('expert comparison must use unmodified Base')
        class Recording(PeftTextBackend):
            def generate(self,request):
                call={'request':deepcopy(request),'request_sha256':canonical_hash(request),'output':None,'error':None,'error_type':None,'usage':None}
                try:call['output']=super().generate(request);return call['output']
                except Exception as exc:call.update(error=str(exc),error_type=type(exc).__name__);raise
                finally:call['usage']=self.last_usage;self.calls.append(call)
        backend=Recording(executor,tokenizer,max_context_tokens=c['max_context_tokens'],max_new_tokens=c['max_new_tokens'],
                          stop_on_json_object=c.get('stop_on_json_object',False))
        runner=make_runner(agents,backend,c)
        report.update(base_model_loads=1,generation_started_at=datetime.now(timezone.utc).isoformat());save()
        with (output/'results.jsonl').open('x') as log:
            for job in plan['jobs']:
                row=by_input[job['sample_id']];backend.calls=[]
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                value=runner.run(input_context(row['input']),workflow=ARMS[job['arm']],mode='base')
                torch.cuda.synchronize()
                r={**job,'input_sha256':canonical_hash(row['input']),'result':value,'calls':backend.calls,
                   'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                   'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                log.write(jsonl([r]));log.flush();report['completed_jobs']+=1;save()
                print(f"{report['completed_jobs']}/{report['planned_jobs']} {job['arm']} {value['status']}",flush=True)
                if any(x['error_type']=='OutOfMemoryError' for x in backend.calls):raise RuntimeError('CUDA OOM')
        raw_replay(output,c)
        report.update(status='completed',results_sha256=sha256_file(output/'results.jsonl'),
                      all_parameters_frozen=all(not p.requires_grad for p in model.parameters()))
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:report['finished_at']=datetime.now(timezone.utc).isoformat();save()
    return report


def verify(root):
    c=load_config(root/'config.json');plan=strict_json((root/'plan.json').read_text());r=strict_json((root/'generation_report.json').read_text())
    if (r['status']!='completed' or r['plan_sha256']!=sha256_file(root/'plan.json')
            or r['results_sha256']!=sha256_file(root/'results.jsonl') or r['loaded_adapters']
            or not r['all_parameters_frozen'] or r['base_model_loads']!=1 or c!=plan['config']
            or r['planned_jobs']!=len(plan['jobs']) or r['completed_jobs']!=len(plan['jobs'])
            or not plan['frozen_at']<r['started_at']<r['generation_started_at']<r['finished_at']):
        raise ValidationError('expert generation invariants failed')
    for name,digest in plan['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root/name)!=digest:raise ValidationError('frozen expert artifact changed')
    paths={k:Path(v) for k,v in plan['source_paths'].items()}
    raw,enriched,proofs,binding=prepare(**paths,c=c)
    frozen_proofs=rows(root/'feature_audits.jsonl')
    if (binding!=plan['bindings'] or raw!=rows(root/'original_inputs.jsonl') or enriched!=rows(root/'inputs.jsonl')
            or [{k:v for k,v in p.items() if k!='seconds'} for p in proofs]!=[{k:v for k,v in p.items() if k!='seconds'} for p in frozen_proofs]
            or jobs(enriched)!=plan['jobs']):raise ValidationError('feature provenance does not reproduce')
    results=raw_replay(root,c)
    return c,paths,raw,results,frozen_proofs


def calculate(root):
    c,paths,inputs,results,proofs=verify(root)
    _,policies=load_paper(root/'paper_config.json')
    _,_,membership=read_inputs(paths['dataset'],c)
    labels=rows(paths['dataset']/'partitions/validation.labels.jsonl')
    by_label={r['sample_id']:r['label'] for r in labels}
    if len(by_label)!=len(labels):raise ValidationError('duplicate validation labels')
    cache={r['sample_id']:r for r in rows(paths['baseline']/'results.jsonl')}
    fresh={(r['sample_id'],r['arm']):r for r in results};times={p['sample_id']:p['seconds'] for p in proofs}
    arms=[*ARMS,'original_base','market_implied','cash'];signals={k:[] for k in arms};probs={k:[] for k in arms if k!='cash'}
    outcomes=[];settlements={};latencies={k:[] for k in arms}
    for row in inputs:
        sid=row['sample_id'];p=row['input'];m=membership[sid];market=m['event_id'].split(':',1)[1]
        record=parse_record({**p,'sample_id':sid,'dataset_source':'market_expert_pilot','dataset_version':m['dataset_version'],
                'event_id':m['event_id'],'event_group_id':m['event_group_id'],'label':by_label[sid]})
        s=Settlement(market,record.label.resolution_time,record.label.outcome)
        if market in settlements and settlements[market]!=s:raise ValidationError('inconsistent settlement')
        settlements[market]=s;outcomes.append(record.label.outcome)
        for arm in arms:
            rr=fresh[(sid,arm)] if arm in ARMS else cache[sid]
            seconds=rr['seconds']+(times[sid] if arm in ARMS else 0)
            if type(seconds) not in (float,int) or not math.isfinite(seconds) or seconds<0:raise ValidationError('invalid decision latency')
            forecast=rr['result']['prediction']['probability'] if rr['result']['status']=='completed' else None
            if arm=='market_implied':forecast=p['market']['probability']
            if arm=='cash':forecast=None
            t=record.forecast_input.observation_time
            signals[arm].append(Signal(sid,market,m['event_group_id'],t,t+timedelta(seconds=math.ceil(seconds)),
                timestamp(p['market']['observed_at'],'market timestamp'),decimal(p['market']['probability']),
                None if forecast is None else decimal(forecast)))
            latencies[arm].append(seconds)
            if arm!='cash':probs[arm].append(forecast)
    feed=TradePrintFeed(paths['trade_store']/'prices.sqlite')
    try:simulation={name:{arm:simulate(ss,list(settlements.values()),feed,policy) for arm,ss in signals.items()} for name,policy in policies.items()}
    finally:feed.close()
    common=[i for i in range(len(inputs)) if all(probs[a][i] is not None for a in [*ARMS,'original_base'])]
    summary={'status':'completed','observations':len(inputs),'event_groups':len({membership[r['sample_id']]['event_group_id'] for r in inputs}),
        'forecast_scores':{a:score_predictions(p,outcomes) for a,p in probs.items()},
        'common_valid_count':len(common),
        'common_valid_scores':{a:score_predictions([p[i] for i in common],[outcomes[i] for i in common]) for a,p in probs.items()} if common else {},
        'mean_total_decision_seconds':{a:sum(v)/len(v) for a,v in latencies.items()},
        'probability_departures_from_market':{a:sum(p is not None and abs(p-q)>1e-6 for p,q in zip(v,probs['market_implied'])) for a,v in probs.items()},
        'resources':{},'scenarios':{},'training':False,'final_test_scored':False,'real_orders_sent':0,'default_promotion':False,
        'generation_report_sha256':sha256_file(root/'generation_report.json'),
        'limitations':['13-group development pilot selected without labels; not all contracts or an untouched profitability test.',
            'All new arms have identical derived inputs. Cached original Base lacks features and used zero repairs; new arms allow one.',
            'Trade prints and hypothetical fees/full fills, not executable depth; no minimum-order, queue, gas or market-impact reconstruction.',
            'Individual arm latency affects fills; sampled equity drawdown can understate intraperiod risk.',
            'Game scenarios are unverified hypotheses; descriptive price statistics do not establish forecasting edge.',
            'Historical outcomes may be in Base pretraining. Block-time availability is reconstructed, not an archived live feed.']}
    for arm in ARMS:
        rr=[r for r in results if r['arm']==arm];calls=[call for r in rr for call in r['calls']]
        summary['resources'][arm]={'model_calls':len(calls),'role_calls':dict(Counter(x['request']['agent'] for x in calls)),
            'valid_forecasts':sum(r['result']['status']=='completed' for r in rr),'workflow_seconds':sum(r['seconds'] for r in rr),
            'max_allocated_GiB':max(r['peak_allocated_bytes'] for r in rr)/2**30,'max_reserved_GiB':max(r['peak_reserved_bytes'] for r in rr)/2**30,
            'input_tokens':sum((x['usage'] or {}).get('input_tokens',0) for x in calls),
            'output_tokens':sum((x['usage'] or {}).get('output_tokens',0) for x in calls),
            'output_limit_calls':sum((x['usage'] or {}).get('output_reached_token_limit',False) for x in calls),
            'failed_stages':dict(Counter(r['result'].get('stage') for r in rr if r['result']['status']!='completed'))}
    for scenario,aa in simulation.items():summary['scenarios'][scenario]={a:{k:v for k,v in result.items() if k not in ('ledger','equity_curve')} for a,result in aa.items()}
    return summary,simulation


def evaluate(root):
    dest=root/'evaluation'
    if dest.exists():raise ValidationError('evaluation already exists')
    summary,simulation=calculate(root);dest.mkdir();artifacts={}
    for scenario,arms in simulation.items():
        (dest/scenario).mkdir()
        for arm,value in arms.items():
            for field in ('ledger','equity_curve'):
                p=dest/scenario/f'{arm}.{field}.jsonl';p.write_text(jsonl(value[field]));artifacts[str(p.relative_to(dest))]=sha256_file(p)
    summary['artifact_hashes']=artifacts;(dest/'report.json').write_text(json_text(summary))
    return summary


def audit(root):
    summary,simulation=calculate(root);dest=root/'evaluation';report=strict_json((dest/'report.json').read_text())
    artifacts=report.pop('artifact_hashes')
    if report!=summary:raise ValidationError('expert metrics do not reproduce')
    expected=set()
    for scenario,arms in simulation.items():
        for arm,value in arms.items():
            for field in ('ledger','equity_curve'):
                name=f'{scenario}/{arm}.{field}.jsonl';expected.add(name)
                if rows(dest/name)!=value[field] or sha256_file(dest/name)!=artifacts.get(name):raise ValidationError('expert ledger does not reproduce')
    if set(artifacts)!=expected:raise ValidationError('expert artifact inventory changed')
    return {'status':'passed','report_sha256':sha256_file(dest/'report.json'),'raw_outputs_replayed':True,
            'features_exactly_reproduced':True,'ledgers_exactly_reproduced':True,'training':False,'final_test_scored':False,'real_orders_sent':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    run_parser=subs.add_parser('run')
    for name in ('dataset','trade-store','baseline','config','agent-config','model-manifest','paper-config','output'):
        run_parser.add_argument('--'+name,type=Path,required=True)
    for name in ('evaluate','audit'):
        subs.add_parser(name).add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if args.command=='run':
        kw=vars(args);kw.pop('command');kw['config_path']=kw.pop('config');result=run(**kw)
    elif args.command=='evaluate':result=evaluate(args.run)
    else:result=audit(args.run)
    print(json_text(result))


if __name__=='__main__':main()
