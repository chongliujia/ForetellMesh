"""Fresh supplementary cases for the preserved profitable workflow and a fixed candidate.

Original cohorts remain immutable. Boundary-quarantined inputs are re-admitted
only under an explicit new chronological protocol, never an outcome/PnL filter.
"""
import argparse
from copy import deepcopy
from datetime import timedelta
from importlib import import_module, util
from importlib.metadata import version
from pathlib import Path
import shutil
import sqlite3
import sys

from .data import sha256_file, strict_json
from .evaluation import json_text
from .lora_probe import verify_model_manifest
from .mean_reversion import HistoricalBars
from .metrics import score_predictions
from .peft_runtime import SharedPeftExecutor, PeftTextBackend
from .schema import timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .team_discovery import HistoricalCatalogue, explore_episode
from .team_discovery_experiment import current_account, statistics
from .team_learning import require
from .team_learning_experiment import read_rows
from .team_memory_loop import MemoryRunner
from .team_memory_experiment import Replay
from .team_outcome_learning import OutcomeBackend, binary_prompt, binary_probability
from .team_portfolio import replay_portfolio, target_progress, continuous_feedback
from .team_rsi_data import make_jobs
from .team_rsi_experiment import now


def bound_report(path,digest):
    require(sha256_file(path)==digest,'source report changed: '+str(path));return strict_json(path.read_text())


def reference_modules(root):
    name='foretellmesh_preserved_v5';directory=(root/'source_snapshot/foretellmesh').resolve()
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).parent==directory,'reference namespace already bound elsewhere')
    else:
        spec=util.spec_from_file_location(name,directory/'__init__.py',submodule_search_locations=[str(directory)])
        module=util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return {n:import_module(name+'.'+n) for n in ('team_discovery','team_outcome_learning','peft_runtime')}


def select_fresh(markets,source_groups,used,related,config):
    start=timestamp(config['initialized_from'],'new start');end=timestamp(config['resolution_before'],'new end')
    admitted=[];excluded=[]
    for m in markets:
        group=m['event_group_id'];proof=m['proof'];reason=None
        if source_groups[group]!='purged_boundary':reason='not_previously_unassigned_boundary_candidate'
        elif group in used:reason='previously_exposed_event_group'
        elif group in related:reason='related_previously_evaluated_family'
        elif timestamp(proof['initialized_at'],'init')<start:reason='initialization_before_new_boundary'
        elif timestamp(proof['resolution_time'],'resolution')>=end:reason='outside_new_resolution_window'
        if reason:excluded.append({'market_id':m['market_id'],'event_group_id':group,'reason':reason})
        else:admitted.append(deepcopy(m))
    return admitted,excluded


def prepare(config):
    require(config['kind']=='preserved_team_fresh_recheck_v1' and config['automatic_fine_tuning'] is False
        and config['final_test_opened'] is False and config['sampling'] is None,'invalid frozen recheck protocol')
    ref=Path(config['reference_run']);r=bound_report(ref/'report.json',config['reference_report_sha256'])
    require(strict_json((ref/'audit.json').read_text())['report_sha256']==config['reference_report_sha256'],'reference audit differs')
    plan=strict_json((ref/'plan.json').read_text());require(r['plan_sha256']==sha256_file(ref/'plan.json'),'reference plan changed')
    for name,digest in {**r['artifact_hashes'],**plan['source_hashes']}.items():
        p=(ref/name).resolve();require(p.is_relative_to(ref.resolve()) and sha256_file(p)==digest,'reference artifact changed')
    ref_config=strict_json((ref/'config.json').read_text())
    for key in ('model','model_revision','execution_policy','max_context_tokens','max_new_tokens','max_scoring_tokens','seed','objective'):
        require(config[key]==ref_config[key],'reference model/runtime/cost budget changed: '+key)
    cohort=Path(config['cohort']);cr=bound_report(cohort/'report.json',config['cohort_report_sha256'])
    require(sha256_file(cohort/'markets.jsonl')==cr['artifact_hashes']['markets.jsonl'],'cohort markets changed')
    old=bound_report(Path(config['old_development_report']),config['old_development_report_sha256'])
    consumed=bound_report(Path(config['consumed_registry']),config['consumed_registry_sha256'])
    used={g for g,p in cr['groups'].items() if p=='train'}|{g for g,p in old['groups'].items() if p=='development'}
    used.update(consumed['used_as_catalogue_or_target_event_groups'])
    markets,excluded=select_fresh(read_rows(cohort/'markets.jsonl'),cr['groups'],used,
        set(config['related_group_exclusions']),config)
    require(markets and not {m['event_group_id'] for m in markets}&used,'no fresh isolated markets')
    ids={m['market_id'] for m in markets};groups={m['event_group_id'] for m in markets}
    # Scan prior model-input artifacts, including searchable catalogues; labels are not used.
    exposure_files=[]
    for pattern in ('*/catalogue.jsonl','*/*.jobs.jsonl','*/jobs.jsonl'):
        for path in sorted(Path('runs').glob(pattern)):
            if path.parent.name==config['run_name']:continue
            rows=read_rows(path);exposure_files.append({'path':str(path),'sha256':sha256_file(path)})
            for row in rows:
                candidates=row.get('context',{}).get('markets',[]) if 'context' in row else [row]
                require(not any(m.get('market_id') in ids or m.get('event_group_id') in groups for m in candidates),
                    'fresh candidate already present in model input: '+str(path))
    store=Path(config['price_store']);pr=bound_report(store/'report.json',config['price_store_report_sha256'])
    require(sha256_file(store/'prices.sqlite')==pr['artifact_hashes']['prices.sqlite'],'prices changed')
    job_config={'observation_age_hours':[config['observation_age_hours']],'observation_trigger':'first_trade_after_age',
        'max_quote_age_seconds':config['execution_policy']['max_quote_age_seconds'],'max_markets_per_episode':3}
    with sqlite3.connect((store/'prices.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        partitions,missing=make_jobs(markets,{m['event_group_id']:'development' for m in markets},db,job_config)
    jobs=partitions['development'];require(jobs,'no fresh observations')
    source_end=max(timestamp(l['resolution_time'],'source settlement') for j in read_rows(ref/'jobs.jsonl') for l in j['labels'].values())
    require(source_end<min(timestamp(j['context']['observation_time'],'fresh cutoff') for j in jobs),'source facts later than evaluation')
    cat=HistoricalCatalogue(jobs,partition='development')
    feed=HistoricalBars.from_store(store/'prices.sqlite',set(cat.rows),
        min(timestamp(m['proof']['initialized_at'],'init') for m in markets)-timedelta(days=30),timestamp(config['resolution_before'],'end'))
    cat.feed=feed;cat.max_age_seconds=config['execution_policy']['max_quote_age_seconds']
    metadata={'admitted_market_ids':sorted(ids),'admitted_event_groups':sorted(groups),
        'source_groups_unchanged':{m['market_id']:cr['groups'][m['event_group_id']] for m in markets},
        'excluded':excluded,'missing_observations':missing,'prior_input_artifacts':exposure_files,
        'source_public_through':source_end.isoformat(),
        'protocol_change':'New supplemental cohort: initialize from Sep 1, resolve before May 1, observe after 168 hours. Prior purged cohorts unchanged.',
        'selection_uses_outcome_value_price_or_return':False}
    return jobs,cat,feed,metadata,reference_modules(ref)


def collect(job,prior,episodes,cat,feed,config,backend,arm,modules,latency=None):
    account=current_account(prior,episodes,feed,config,job['context']['observation_time'])
    transform=lambda decision,elapsed,feedback:continuous_feedback(prior,episodes,job,decision,elapsed,feedback,feed,config['execution_policy'])
    if arm=='reference':
        runner=modules['team_discovery'].DiscoveryRunner(backend,cat);explore=modules['team_discovery'].explore_episode
    else:runner=MemoryRunner(backend,cat);explore=explore_episode
    return explore(job['context'],job['labels'],feed,config['execution_policy'],runner,account,
        feedback_transform=transform,recorded_latency=latency)


def controls(jobs,feed,config):
    result={}
    for value in config['constant_probabilities']:
        eps=[{'context':j['context'],'decision_seconds':config['constant_latency_seconds'],
            'decision':{'forecast':{'forecasts':[{'market_id':m['market_id'],'probability':value,'consider_trade':True}
                for m in j['context']['markets']]},'risk':{'veto_markets':[]}}} for j in jobs]
        result['constant_'+str(value)]=replay_portfolio(jobs,eps,feed,config['execution_policy'])
    result['cash']=replay_portfolio([],[],feed,config['execution_policy']);return result


def summarize(jobs,arms,accounts,config):
    ps={n:[] for n in arms};qs=[];ys=[];low={n:[] for n in arms};low_q=[];low_y=[];candidate_side_count=0
    for i,j in enumerate(jobs):
        fs={n:{} if eps[i]['decision'] is None else {f['market_id']:f['probability'] for f in eps[i]['decision']['forecast']['forecasts']} for n,eps in arms.items()}
        for m in j['context']['markets']:
            mid=m['market_id'];q=m['input']['market']['probability']
            is_low=min(q,1-q)<=config['low_price_threshold'];candidate_side_count+=is_low
            if all(mid in fs[n] for n in arms):
                qs.append(q);ys.append(j['labels'][mid]['outcome'])
                for n in arms:ps[n].append(fs[n][mid])
                if is_low:
                    low_q.append(q);low_y.append(j['labels'][mid]['outcome'])
                    for n in arms:low[n].append(fs[n][mid])
    from decimal import Decimal
    armstats={n:statistics(e) for n,e in arms.items()}
    scores={n:score_predictions(p,ys) for n,p in {**ps,'market':qs,**{'constant_'+str(v):[v]*len(ys) for v in config['constant_probabilities']}}.items()}
    return {'arm_statistics':armstats,'common_scores':scores,
        'low_price_observations':candidate_side_count,'low_price_common_scores':{n:score_predictions(p,low_y) for n,p in {**low,'market':low_q}.items()},
        'account_results':{n:{k:a[k] for k in ('initial_cash','final_cash','net_pnl','fees_paid','filled_trades','sampled_equity_proxy_max_drawdown_usd')} for n,a in accounts.items()},
        'reference_profitable_on_this_pilot':Decimal(accounts['reference']['final_cash'])>Decimal('100'),
        'candidate_net_cash_above_reference_on_this_pilot':Decimal(accounts['candidate']['final_cash'])>Decimal(accounts['reference']['final_cash']),
        'effectiveness_verified':False,'fine_tuning_admitted':False,
        'interpretation':'Small fresh-event diagnostic only; it cannot establish a durable edge or separate luck from skill.'}


def run(config_path,output):
    require(not output.exists(),'recheck output exists');config=strict_json(config_path.read_text())
    require(output.name==config['run_name'],'output must match frozen run name')
    jobs,cat,feed,meta,modules=prepare(config)
    print('Prepared',len(jobs),'observations;',len(cat.rows),'fresh contracts; verifying frozen Base',flush=True)
    ref=Path(config['reference_run']);manifest=ref/'model_manifest.json';model_path,model_hash=verify_model_manifest(manifest,config)
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'BF16 GPU unavailable')
    output.mkdir(parents=True);shutil.copyfile(config_path,output/'config.json');shutil.copyfile(manifest,output/'model_manifest.json')
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    (output/'jobs.jsonl').write_text(jsonl(jobs));(output/'catalogue.jsonl').write_text(jsonl(list(cat.rows.values())))
    (output/'selection.json').write_text(json_text(meta))
    plan={'created_at':now(),'config_sha256':sha256_file(config_path),'model_manifest_sha256':model_hash,
        'jobs_sha256':sha256_file(output/'jobs.jsonl'),'selection_sha256':sha256_file(output/'selection.json'),
        'catalogue_sha256':sha256_file(output/'catalogue.jsonl'),'reference_report_sha256':config['reference_report_sha256'],
        'source_hashes':{str(p.relative_to(output)):sha256_file(p) for p in sorted((output/'source_snapshot').rglob('*.py'))},
        'frozen_before_model_calls':True,'arm_order':'alternate reference/candidate; reset fixed seed per arm',
        'primary':'after-cost terminal cash and drawdown vs cash and frozen reference; report coverage and common forecast scores',
        'no_automatic_promotion':True}
    (output/'plan.json').write_text(json_text(plan))
    report={'kind':config['kind'],'status':'running','started_at':now(),'plan_sha256':sha256_file(output/'plan.json'),
        'completed_pairs':0,'fine_tuning_triggered':False,'final_test_opened':False,'real_orders_sent':0,'default_promoted':False,
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'packages':{n:version(n) for n in ('torch','transformers','peft','langgraph')}}
    def save():
        p=output/'report.tmp';p.write_text(json_text(report));p.replace(output/'report.json')
    save();arms={'reference':[],'candidate':[]}
    try:
        set_seed(config['seed']);torch.backends.cuda.matmul.allow_tf32=False
        tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        executor=SharedPeftExecutor(model);report['base_model_loads']=1
        backends={}
        for n in arms:
            text_cls=modules['peft_runtime'].PeftTextBackend if n=='reference' else PeftTextBackend
            out_cls=modules['team_outcome_learning'].OutcomeBackend if n=='reference' else OutcomeBackend
            backends[n]=out_cls(text_cls(executor,tok,max_context_tokens=config['max_context_tokens'],
                max_new_tokens=config['max_new_tokens'],sampling=None,stop_on_json_object=True),max_scoring_tokens=config['max_scoring_tokens'])
        for i,j in enumerate(jobs):
            for n in (('reference','candidate') if i%2==0 else ('candidate','reference')):
                set_seed(config['seed']);b=backends[n];b.readouts=[]
                ep=collect(j,jobs[:i],arms[n],cat,feed,config,b,n,modules);ep['readouts']=deepcopy(b.readouts);arms[n].append(ep)
                with (output/(n+'.episodes.jsonl')).open('a') as f:f.write(jsonl([ep]))
                print(n,i+1,'decision',ep['decision'] is not None,'error',ep['decision_error'],flush=True)
            report['completed_pairs']=i+1;save()
        accounts={n:replay_portfolio(jobs,e,feed,config['execution_policy']) for n,e in arms.items()};accounts.update(controls(jobs,feed,config))
        for n,a in accounts.items():(output/(n+'.portfolio.json')).write_text(json_text(a))
        require(all(not p.requires_grad for p in model.parameters()),'weights unfrozen')
        report.update(status='completed',finished_at=now(),summary=summarize(jobs,arms,accounts,config),
            goals={n:target_progress(accounts[n],config['objective']) for n in arms},
            all_parameters_frozen=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            limitations=['Only three new historical event groups; not representative or sufficient for fine tuning admission.',
                'Original v5 code and readout preserved. Candidate uses previously developed concrete-study guard and selectable rules tool; this is an interface comparison, not newly learned RSI memory.',
                'No external contemporaneous news; historical pretraining contamination is not ruled out.',
                'Boundary quarantine is re-reviewed under an explicit new protocol; no old cohort or final-test assignment is changed.',
                'Trade-print fills and settlement exits are simulations; a long-dated contract may remain open for months.',
                'Fixed controls use predeclared 60-second delay; model arms use measured latency. No control is selected or promoted by PnL.'])
        report['artifact_hashes']={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name not in ('report.json','report.tmp','audit.json')};save()
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=now());save();raise
    return report


def audit(output):
    report=strict_json((output/'report.json').read_text());config=strict_json((output/'config.json').read_text());plan=strict_json((output/'plan.json').read_text())
    require(report['status']=='completed' and report['plan_sha256']==sha256_file(output/'plan.json'),'unfinished/changed recheck')
    for name,digest in {**plan['source_hashes'],**report['artifact_hashes']}.items():
        p=(output/name).resolve();require(p.is_relative_to(output.resolve()) and sha256_file(p)==digest,'artifact changed')
    for name,key in [('config.json','config_sha256'),('model_manifest.json','model_manifest_sha256'),('jobs.jsonl','jobs_sha256'),
                     ('selection.json','selection_sha256'),('catalogue.jsonl','catalogue_sha256')]:require(sha256_file(output/name)==plan[key],'plan binding differs')
    jobs,cat,feed,meta,modules=prepare(config)
    # New unrelated run artifacts may have appeared since the frozen exposure scan.
    recorded=strict_json((output/'selection.json').read_text());old_scan=recorded['prior_input_artifacts'];now_scan=meta.pop('prior_input_artifacts')
    require(all(r in now_scan for r in old_scan),'old exposure artifacts changed');recorded_without_scan={k:v for k,v in recorded.items() if k!='prior_input_artifacts'}
    require(meta==recorded_without_scan and jobs==read_rows(output/'jobs.jsonl') and list(cat.rows.values())==read_rows(output/'catalogue.jsonl'),'fresh source changed')
    arms={n:read_rows(output/(n+'.episodes.jsonl')) for n in ('reference','candidate')};accounts={};total=0
    for n,eps in arms.items():
        require(len(eps)==len(jobs),'missing episodes')
        prompt=modules['team_outcome_learning'].binary_prompt if n=='reference' else binary_prompt
        for i,(j,ep) in enumerate(zip(jobs,eps)):
            replay=Replay(ep['calls']);rebuilt=collect(j,jobs[:i],eps[:i],cat,feed,config,replay,n,modules,ep['decision_seconds'])
            for key in ('context','account_at_observation','decision','decision_error','decision_call_count','discovery','feedback','reflection','reflection_error'):
                require(rebuilt[key]==ep[key],'recheck replay differs: '+key)
            require(replay.i==len(ep['calls']),'unused model calls');total+=1
            calls={canonical_hash(c['request']):c for c in ep['calls']}
            for r in ep['readouts']:require(r['prompt']==prompt(calls[r['request_sha256']]['request'],r['market_id']) and
                r['probability']==binary_probability(r['logits_no_yes']),'readout changed')
        accounts[n]=replay_portfolio(jobs,eps,feed,config['execution_policy'])
    accounts.update(controls(jobs,feed,config))
    for n,a in accounts.items():require(a==strict_json((output/(n+'.portfolio.json')).read_text()),'account differs')
    require(summarize(jobs,arms,accounts,config)==report['summary'],'summary differs')
    result={'status':'passed','report_sha256':sha256_file(output/'report.json'),'replayed_model_episodes':total,
        'reproduced_accounts':len(accounts),'reference_uses_preserved_source':True,'new_model_calls':0,'final_test_opened':False,'fine_tuning_performed':False}
    (output/'audit.json').write_text(json_text(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--config',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    sub.add_parser('audit').add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json_text(audit(a.output) if a.command=='audit' else run(a.config,a.output)))
