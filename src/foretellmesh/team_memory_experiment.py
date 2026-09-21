"""Offline method learning followed by a frozen, fresh-event paired paper comparison."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
from .data import sha256_file, strict_json
from .evaluation import json_text
from .lora_probe import verify_model_manifest
from .mean_reversion import HistoricalBars
from .metrics import score_predictions
from .peft_runtime import SharedPeftExecutor, PeftTextBackend
from .schema import timestamp, ValidationError
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .team_discovery import HistoricalCatalogue, explore_episode
from .team_discovery_experiment import current_account, statistics
from .team_learning import require
from .team_learning_experiment import read_rows
from .team_memory_loop import MemoryRunner
from .team_method_memory import EDITOR, learning_evidence, validate_lessons, freeze_memory, validate_memory, validate_evidence
from .team_outcome_learning import OutcomeBackend, binary_prompt, binary_probability
from .team_portfolio import replay_portfolio, target_progress, continuous_feedback
from .team_rsi_experiment import load_partition, now


def spaced(rows,n):
    require(rows and type(n) is int and 1<=n<=64,'empty/unbounded selection')
    n=min(n,len(rows));return rows[:1] if n==1 else [rows[i*(len(rows)-1)//(n-1)] for i in range(n)]


def prepare(config):
    require(config['kind']=='team_memory_comparison_v1' and config['automatic_fine_tuning'] is False
        and config['final_test_opened'] is False and config['sampling'] is None and 'training' not in config,'invalid frozen protocol')
    train=load_partition(config,'train')
    start=timestamp(config['learning_start'],'start');end=timestamp(config['learning_end'],'end')
    pool=[j for j in train if start<=timestamp(j['context']['observation_time'],'at')<end]
    def mature(j):
        m=next(m for m in j['context']['markets'] if m['event_group_id']==j['anchor_group_id'])
        return (timestamp(j['context']['observation_time'],'at')-timestamp(j['provenance'][m['market_id']]['initialized_at'],'init')).total_seconds()>=config['minimum_anchor_age_hours']*3600
    learning=spaced([j for j in pool if mature(j)],config['learning_episodes'])
    old_path=Path(config['previous_development_report'])
    require(sha256_file(old_path)==config['previous_development_report_sha256'],'changed previous evaluation exclusion')
    old=strict_json(old_path.read_text());excluded={g for g,s in old['groups'].items() if s=='development'}
    excluded.update(config['additional_related_group_exclusions'])
    # Inputs only; selection never reads a return, forecast or binary outcome.
    development=[]; lineage=[]
    for original in load_partition(config,'development'):
        if original['anchor_group_id'] in excluded:continue
        job=deepcopy(original);job['context']['markets']=[m for m in job['context']['markets'] if m['event_group_id'] not in excluded]
        ids={m['market_id'] for m in job['context']['markets']}
        job['labels']={k:v for k,v in job['labels'].items() if k in ids}
        job['provenance']={k:v for k,v in job['provenance'].items() if k in ids}
        lineage.append({'episode_id':job['context']['episode_id'],'parent_job_sha256':canonical_hash(original),
            'derived_job_sha256':canonical_hash(job),'removed_peer_market_ids':sorted(set(original['labels'])-ids)})
        development.append(job)
    evaluation=spaced(development,config['evaluation_episodes'])
    learn_cat=HistoricalCatalogue(pool);eval_cat=HistoricalCatalogue(development,partition='development')
    learn_groups={r['event_group_id'] for r in learn_cat.rows.values()};eval_groups={r['event_group_id'] for r in eval_cat.rows.values()}
    require(not learn_groups&eval_groups and not eval_groups&excluded,'overlapping memory/evaluation/used groups')
    require(max(timestamp(l['resolution_time'],'source') for j in learning for l in j['labels'].values())
        <min(timestamp(j['context']['observation_time'],'evaluation') for j in evaluation),'source outcomes later than evaluation')
    store=Path(config['price_store']);report=strict_json((store/'report.json').read_text())
    require(sha256_file(store/'report.json')==config['price_store_report_sha256'] and
        sha256_file(store/'prices.sqlite')==report['artifact_hashes']['prices.sqlite'],'changed price source')
    cats=[learn_cat,eval_cat];rows=[r for c in cats for r in c.rows.values()]
    feed=HistoricalBars.from_store(store/'prices.sqlite',{r['market_id'] for r in rows},
        min(timestamp(r['initialized_at'],'init') for r in rows)-timedelta(days=30),timestamp(config['price_read_before'],'end'))
    for cat in cats:cat.feed=feed;cat.max_age_seconds=config['execution_policy']['max_quote_age_seconds']
    metadata={'excluded_groups':sorted(excluded),'learning_pool_episodes':len(pool),'learning_catalogue_contracts':len(learn_cat.rows),
        'fresh_development_pool_events':len(development),'evaluation_catalogue_contracts':len(eval_cat.rows),'derived_job_lineage':lineage,
        'selection':'fixed chronology and minimum learning observation age, never observed profits'}
    target_progress(replay_portfolio([],[],feed,config['execution_policy']),config['objective'])
    return learning,evaluation,learn_cat,eval_cat,feed,metadata


def comparison(jobs,arms,portfolios):
    ys=[];market=[];ps={'control':[],'memory':[]};changes=[]
    for i,j in enumerate(jobs):
        forecasts={name:{} if eps[i]['decision'] is None else {r['market_id']:r['probability'] for r in eps[i]['decision']['forecast']['forecasts']}
                   for name,eps in arms.items()}
        for m in j['context']['markets']:
            mid=m['market_id']
            if all(mid in forecasts[n] for n in ps):
                ys.append(j['labels'][mid]['outcome']);market.append(m['input']['market']['probability'])
                for name in ps:ps[name].append(forecasts[name][mid])
        changes.append({'episode_id':j['context']['episode_id'],
            'study_changed':arms['control'][i]['discovery']['study']!=arms['memory'][i]['discovery']['study'],
            'decision_changed':arms['control'][i]['decision']!=arms['memory'][i]['decision']})
    scores={n:score_predictions(p,ys) for n,p in {**ps,'market':market,'constant_0_5':[.5]*len(ys)}.items()}
    stats={n:statistics(e) for n,e in arms.items()}
    tools={n:dict(Counter(r['specification']['tool'] for e in eps for r in e['discovery']['investigations'])) for n,eps in arms.items()}
    from decimal import Decimal
    directional=bool(ys) and scores['memory']['brier']<scores['control']['brier'] and scores['memory']['log_loss']<=scores['control']['log_loss']
    directional=directional and stats['memory']['completed_decisions']>=stats['control']['completed_decisions']
    directional=directional and stats['memory']['forecast_scores']['prediction_count']>=stats['control']['forecast_scores']['prediction_count']
    directional=directional and Decimal(portfolios['memory']['final_cash'])>=Decimal(portfolios['control']['final_cash'])
    directional=directional and Decimal(portfolios['memory']['sampled_equity_proxy_max_drawdown_usd'])<=Decimal(portfolios['control']['sampled_equity_proxy_max_drawdown_usd'])
    return {'arm_statistics':stats,'common_covered_forecast_scores':scores,'behavior_changes':changes,'tool_choices':tools,
        'predeclared_directional_screen_passed':directional,'effectiveness_verified':False,'fine_tuning_admitted':False,
        'interpretation':'Small developmental paired diagnostic, not statistical confirmation. Evaluation feedback never edits memory.'}


def collect(job,prior_jobs,episodes,feed,config,backend,catalogue,memory=None,recorded=None):
    account=current_account(prior_jobs,episodes,feed,config,job['context']['observation_time'])
    transform=lambda decision,elapsed,feedback:continuous_feedback(prior_jobs,episodes,job,decision,elapsed,feedback,feed,config['execution_policy'])
    return explore_episode(job['context'],job['labels'],feed,config['execution_policy'],MemoryRunner(backend,catalogue,memory),
        account,recorded_latency=recorded,feedback_transform=transform)


def edit_memory(backend,catalogue,jobs,episodes,*,archived_evidence=None):
    evidence=learning_evidence(jobs,episodes) if archived_evidence is None else validate_evidence(jobs,episodes,archived_evidence)
    runner=MemoryRunner(backend,catalogue)
    error=None;edited=None
    try:edited=runner.structured('memory_editor',EDITOR,evidence,{},lambda v:validate_lessons(v,evidence['facts']))
    except (ValueError,TypeError,KeyError) as exc:error=str(exc)
    return evidence,edited,error,runner.calls


def run(config_path,manifest,output):
    require(not output.exists(),'output exists');config=strict_json(config_path.read_text())
    learning,evaluation,learn_cat,eval_cat,feed,metadata=prepare(config)
    print('Prepared learning',len(learning),'fresh evaluation',len(evaluation),'catalogues',len(learn_cat.rows),len(eval_cat.rows),flush=True)
    model_path,model_hash=verify_model_manifest(manifest,config)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'BF16 GPU unavailable')
    output.mkdir(parents=True);shutil.copyfile(config_path,output/'config.json');shutil.copyfile(manifest,output/'model_manifest.json')
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    for name,value in [('learning',learning),('evaluation',evaluation)]: (output/(name+'.jobs.jsonl')).write_text(jsonl(value))
    (output/'selection.json').write_text(json_text(metadata))
    plan={'created_at':now(),'config_sha256':sha256_file(config_path),'model_manifest_sha256':model_hash,
        'learning_jobs_sha256':sha256_file(output/'learning.jobs.jsonl'),'evaluation_jobs_sha256':sha256_file(output/'evaluation.jobs.jsonl'),
        'selection_sha256':sha256_file(output/'selection.json'),
        'source_hashes':{str(p.relative_to(output)):sha256_file(p) for p in sorted((output/'source_snapshot').rglob('*.py'))},
        'arm_order':'alternate control/memory by episode; each resets seed; one independent continuous $100 account per arm',
        'directional_screen':'lower common-covered Brier, no worse common log loss, decision/forecast coverage, terminal cash or drawdown',
        'promotion':'none; this small pilot cannot admit fine tuning or establish effectiveness'}
    (output/'plan.json').write_text(json_text(plan))
    report={'kind':'team_memory_comparison_v1','status':'running','stage':'learning','started_at':now(),
        'plan_sha256':sha256_file(output/'plan.json'),'fine_tuning_triggered':False,'final_test_opened':False,
        'effectiveness_verified':False,'default_promoted':False,'real_orders_sent':0,'completed_learning':0,'completed_pairs':0,
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda}
    def save():
        p=output/'report.tmp';p.write_text(json_text(report));p.replace(output/'report.json')
    save();episodes=[];arms={'control':[],'memory':[]}
    try:
        set_seed(config['seed']);torch.backends.cuda.matmul.allow_tf32=False
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        executor=SharedPeftExecutor(model);report['base_model_loads']=1
        backend=OutcomeBackend(PeftTextBackend(executor,tokenizer,max_context_tokens=config['max_context_tokens'],
            max_new_tokens=config['max_new_tokens'],sampling=None,stop_on_json_object=True),max_scoring_tokens=config['max_scoring_tokens'])
        def capture(job,prior,eps,cat,memory,path):
            set_seed(config['seed']);backend.readouts=[]
            ep=collect(job,prior,eps,feed,config,backend,cat,memory);ep['readouts']=deepcopy(backend.readouts)
            eps.append(ep)
            with path.open('a') as f:f.write(jsonl([ep]))
            print(path.stem,len(eps),'decision',ep['decision'] is not None,'studies',len(ep['discovery']['investigations']),flush=True)
        for i,j in enumerate(learning):
            capture(j,learning[:i],episodes,learn_cat,None,output/'learning.episodes.jsonl');report['completed_learning']=len(episodes);save()
        report['stage']='memory_compilation';save()
        evidence,edited,error,calls=edit_memory(backend,learn_cat,learning,episodes)
        (output/'memory_editor.json').write_text(json_text({'evidence':evidence,'edited':edited,'error':error,'calls':calls}))
        require(edited is not None,'memory editor failed; do not manufacture a lesson or open evaluation')
        memory=freeze_memory(learning,episodes,evidence,edited,now(),learn_cat)
        for j in evaluation:validate_memory(memory,j['context'],{r['event_group_id'] for r in eval_cat.rows.values()})
        (output/'memory.json').write_text(json_text(memory))
        seal={'frozen_at':now(),'memory_file_sha256':sha256_file(output/'memory.json'),
              'learning_episodes_sha256':sha256_file(output/'learning.episodes.jsonl'),'memory_editor_sha256':sha256_file(output/'memory_editor.json'),
              'evaluation_jobs_sha256':sha256_file(output/'evaluation.jobs.jsonl'),'evaluation_calls_so_far':0}
        (output/'memory_freeze.json').write_text(json_text(seal));report['stage']='paired_evaluation';report['memory_freeze_sha256']=sha256_file(output/'memory_freeze.json');save()
        print('Memory frozen:',len(memory['lessons']),'lessons; starting fixed evaluation',flush=True)
        for i,j in enumerate(evaluation):
            require(sha256_file(output/'memory.json')==seal['memory_file_sha256'],'memory mutated')
            for name in (('control','memory') if i%2==0 else ('memory','control')):
                capture(j,evaluation[:i],arms[name],eval_cat,memory if name=='memory' else None,output/(name+'.episodes.jsonl'))
            report['completed_pairs']=i+1;save()
        portfolios={name:replay_portfolio(evaluation,eps,feed,config['execution_policy']) for name,eps in arms.items()}
        learning_account=replay_portfolio(learning,episodes,feed,config['execution_policy'])
        for name,account in {**portfolios,'learning':learning_account}.items():(output/(name+'.portfolio.json')).write_text(json_text(account))
        report.update(status='completed',stage='completed',finished_at=now(),learning_statistics=statistics(episodes),
            comparison=comparison(evaluation,arms,portfolios),goals={n:target_progress(a,config['objective']) for n,a in portfolios.items()},
            all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            limitations=['Offline fitting uses earlier publicly resolved events but actual evidence capture and memory creation occur in 2026; not a historical live-deployable result.',
                'Pretraining contamination is not ruled out. Small, nonpolitical admitted pilot is not all prediction markets.',
                'Trade-print simulated fills, explicit fee/premium assumptions, settlement exits; no order-book execution guarantee.',
                'Related same-season contracts may remain correlated; event counts do not imply independent statistical samples.',
                'No automatic promotion or fine tuning. Evaluation feedback is quarantined from the frozen method.'])
        require(report['all_parameters_frozen'],'unfrozen weights')
        report['artifact_hashes']={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name not in ('report.json','report.tmp','audit.json')};save()
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=now());save();raise
    return report


class Replay:
    def __init__(self,calls):self.calls=calls;self.i=0
    def generate(self,request):
        call=self.calls[self.i];self.i+=1;require(call['request']==request,'model request differs')
        if call['output'] is None:raise ValidationError(call['error'])
        return call['output']


def audit(output):
    config=strict_json((output/'config.json').read_text());report=strict_json((output/'report.json').read_text());plan=strict_json((output/'plan.json').read_text())
    require(report['status']=='completed' and report['plan_sha256']==sha256_file(output/'plan.json'),'unfinished/changed run')
    for name,digest in {**plan['source_hashes'],**report['artifact_hashes']}.items():
        p=(output/name).resolve();require(p.is_relative_to(output.resolve()) and sha256_file(p)==digest,'changed artifact')
    require(plan['config_sha256']==sha256_file(output/'config.json') and plan['model_manifest_sha256']==sha256_file(output/'model_manifest.json')
        and plan['learning_jobs_sha256']==sha256_file(output/'learning.jobs.jsonl')
        and plan['evaluation_jobs_sha256']==sha256_file(output/'evaluation.jobs.jsonl')
        and plan['selection_sha256']==sha256_file(output/'selection.json'),'plan binding differs')
    learning,evaluation,learn_cat,eval_cat,feed,metadata=prepare(config)
    require(learning==read_rows(output/'learning.jobs.jsonl') and evaluation==read_rows(output/'evaluation.jobs.jsonl')
        and metadata==strict_json((output/'selection.json').read_text()),'selection changed')
    memory=strict_json((output/'memory.json').read_text());seal=strict_json((output/'memory_freeze.json').read_text())
    require(report['memory_freeze_sha256']==sha256_file(output/'memory_freeze.json') and seal['evaluation_calls_so_far']==0,'seal differs')
    for k,p in [('memory_file_sha256','memory.json'),('learning_episodes_sha256','learning.episodes.jsonl'),
                ('memory_editor_sha256','memory_editor.json'),('evaluation_jobs_sha256','evaluation.jobs.jsonl')]:
        require(seal[k]==sha256_file(output/p),'freeze inputs changed')
    episodes=read_rows(output/'learning.episodes.jsonl');editor=strict_json((output/'memory_editor.json').read_text())
    replay=Replay(editor['calls']);evidence,edited,error,_=edit_memory(replay,learn_cat,learning,episodes,archived_evidence=editor['evidence'])
    require(evidence==editor['evidence'] and edited==editor['edited'] and error==editor['error'] and replay.i==len(editor['calls']),'editor replay differs')
    require(freeze_memory(learning,episodes,evidence,edited,memory['built_at'],learn_cat)==memory,'memory does not reproduce')
    arms={n:read_rows(output/(n+'.episodes.jsonl')) for n in ('control','memory')}
    portfolios={};total=0;memory_calls=0
    for name,eps in {'learning':episodes,**arms}.items():
        jobs=learning if name=='learning' else evaluation;cat=learn_cat if name=='learning' else eval_cat
        require(len(eps)==len(jobs),'missing episodes')
        for i,(j,ep) in enumerate(zip(jobs,eps)):
            replay=Replay(ep['calls']);rebuilt=collect(j,jobs[:i],eps[:i],feed,config,replay,cat,memory if name=='memory' else None,ep['decision_seconds'])
            for key in ('account_at_observation','context','decision','decision_error','decision_call_count','discovery','feedback','reflection','reflection_error'):
                require(rebuilt[key]==ep[key],'episode replay differs: '+key)
            require(replay.i==len(ep['calls']),'unused calls');total+=1
            for call in ep['calls']:
                carries='offline_team_memory' in call['request']['upstream'];role=call['request']['agent']
                require(carries==(name=='memory' and role!='discovery_reflection'),'memory missing or crossed arm/feedback boundary')
                if carries:memory_calls+=1
            calls={canonical_hash(c['request']):c for c in ep['calls']}
            for r in ep['readouts']:
                require(r['prompt']==binary_prompt(calls[r['request_sha256']]['request'],r['market_id'])
                    and r['probability']==binary_probability(r['logits_no_yes']),'readout differs')
        account=replay_portfolio(jobs,eps,feed,config['execution_policy'])
        require(account==strict_json((output/(name+'.portfolio.json')).read_text()),'portfolio differs');portfolios[name]=account
    require(comparison(evaluation,arms,portfolios)==report['comparison'] and statistics(episodes)==report['learning_statistics'],'metrics differ')
    require({n:target_progress(portfolios[n],config['objective']) for n in arms}==report['goals'],'goal differs')
    result={'status':'passed','report_sha256':sha256_file(output/'report.json'),'replayed_episodes':total,
        'memory_injected_decision_calls':memory_calls,'new_model_calls':0,
        'auditor_sha256':sha256_file(Path(__file__)),'memory_validator_sha256':sha256_file(Path(__file__).with_name('team_method_memory.py')),
        'legacy_fact_reference_order':'exact archived order after set, uniqueness, source-hash and value checks','final_test_opened':False,'fine_tuning_performed':False}
    (output/'audit.json').write_text(json_text(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run')
    for k in ('config','model-manifest','output'):r.add_argument('--'+k,type=Path,required=True)
    sub.add_parser('audit').add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.output) if a.command=='audit' else run(a.config,a.model_manifest,a.output)
    print(json_text(result))
