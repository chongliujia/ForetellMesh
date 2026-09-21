"""Run and audit bounded catalogue exploration with one frozen Base and one paper account."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
import shutil

from .data import sha256_file, strict_json
from .evaluation import json_text
from .lora_probe import verify_model_manifest
from .mean_reversion import HistoricalBars
from .peft_runtime import SharedPeftExecutor, PeftTextBackend
from .schema import timestamp, ValidationError
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .team_discovery import HistoricalCatalogue, DiscoveryRunner, explore_episode
from .team_learning import require
from .team_learning_experiment import read_rows
from .team_outcome_learning import OutcomeBackend, binary_prompt, binary_probability
from .team_portfolio import replay_portfolio, target_progress, continuous_feedback
from .team_rsi_experiment import load_partition, now
from .metrics import score_predictions


def prepare(config):
    require(config['kind'] == 'team_discovery_v1' and config['automatic_fine_tuning'] is False
        and config['effectiveness_evidence_required'] is True and config['final_test_opened'] is False
        and config['sampling'] is None and config['catalogue_search_rounds'] == 2
        and type(config['max_episodes']) is int and 1 <= config['max_episodes'] <= 64 and 'training' not in config, 'invalid frozen discovery protocol')
    jobs=load_partition(config,'train'); catalogue=HistoricalCatalogue(jobs)
    count=min(len(jobs),config['max_episodes'])
    selection=config.get('episode_selection','chronological_prefix')
    require(selection in ('chronological_prefix','chronological_even_spacing'), 'invalid episode selection')
    selected=jobs[:count] if selection == 'chronological_prefix' or count <= 1 else [
        jobs[i*(len(jobs)-1)//(count-1)] for i in range(count)]
    store=Path(config['price_store']); report=strict_json((store/'report.json').read_text())
    require(sha256_file(store/'report.json') == config['price_store_report_sha256']
        and sha256_file(store/'prices.sqlite') == report['artifact_hashes']['prices.sqlite'], 'changed discovery prices')
    start=min(timestamp(r['initialized_at'],'initialization') for r in catalogue.rows.values())-timedelta(days=30)
    feed=HistoricalBars.from_store(store/'prices.sqlite',set(catalogue.rows),start,timestamp(config['price_read_before'],'price end'))
    catalogue.feed=feed; catalogue.max_age_seconds=config['execution_policy']['max_quote_age_seconds']
    # Validate the objective on an empty, terminal cash baseline before any GPU use.
    cash=replay_portfolio([],[],feed,config['execution_policy'])
    target_progress(cash,config['objective'])
    return selected,catalogue,feed


def current_account(jobs, episodes, feed, config, cutoff):
    state=replay_portfolio(jobs,episodes,feed,config['execution_policy'],through=timestamp(cutoff,'account cutoff'))['snapshot']
    state['objective']=deepcopy(config['objective']); state['simulation_only']=True
    return state


def statistics(episodes):
    records=[r for e in episodes for r in e['discovery'].get('investigations',[])]
    searches=[r for e in episodes for r in e['discovery'].get('searches',[])]
    reviews=[r for e in episodes for r in (e['discovery'].get('review') or {}).get('reviews',[])]
    ps=[]; qs=[]; ys=[]
    for e in episodes:
        forecasts={} if e['decision'] is None else {r['market_id']:r['probability'] for r in e['decision']['forecast']['forecasts']}
        for m in e['context']['markets']:
            ps.append(forecasts.get(m['market_id'])); qs.append(m['input']['market']['probability'])
            ys.append(e['feedback']['outcomes'][m['market_id']])
    return {'episodes':len(episodes),'completed_decisions':sum(e['decision'] is not None for e in episodes),
        'forecast_scores':score_predictions(ps,ys),'market_scores':score_predictions(qs,ys),
        'model_calls':sum(len(e['calls']) for e in episodes),
        'invalid_calls':sum(c['error'] is not None for e in episodes for c in e['calls']),
        'searches':len(searches),'empty_searches':sum(not s['records'] for s in searches),
        'distinct_retrieved_contracts':len({r['market_id'] for s in searches for r in s['records']}),
        'investigations':len(records),'cross_contract_investigations':sum(len(r['specification']['market_ids']) == 2 for r in records),
        'review_verdicts':dict(Counter(r['verdict'] for r in reviews)),
        'valid_reflections':sum(e['reflection'] is not None for e in episodes),
        'nonempty_reflections':sum(bool(e['reflection'] and e['reflection']['observations']) for e in episodes),
        'explicit_no_lesson':sum(bool(e['reflection'] and not e['reflection']['observations']) for e in episodes),
        'reflection_failures':sum(e['reflection_error'] is not None for e in episodes)}


def run(config_path, model_manifest, output):
    require(not output.exists(),'discovery output exists'); config=strict_json(config_path.read_text())
    jobs,catalogue,feed=prepare(config)
    print('Prepared',len(jobs),'episodes and',len(catalogue.rows),'admitted catalogue contracts; verifying Base',flush=True)
    model_path,model_hash=verify_model_manifest(model_manifest,config)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'BF16 GPU unavailable')
    output.mkdir(parents=True); shutil.copyfile(config_path,output/'config.json'); shutil.copyfile(model_manifest,output/'model_manifest.json')
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    (output/'jobs.jsonl').write_text(jsonl(jobs)); (output/'catalogue.jsonl').write_text(jsonl(list(catalogue.rows.values())))
    plan={'created_at':now(),'config_sha256':sha256_file(config_path),'model_manifest_sha256':model_hash,
        'jobs_sha256':sha256_file(output/'jobs.jsonl'),'catalogue_sha256':sha256_file(output/'catalogue.jsonl'),
        'source_hashes':{str(p.relative_to(output)):sha256_file(p) for p in sorted((output/'source_snapshot').rglob('*.py'))}}
    (output/'plan.json').write_text(json_text(plan))
    report={'kind':'frozen_team_discovery_v1','status':'running','started_at':now(),'plan_sha256':sha256_file(output/'plan.json'),
        'fine_tuning_triggered':False,'effectiveness_verified':False,'final_test_opened':False,'development_opened':False,
        'default_promoted':False,'real_orders_sent':0,'completed_episodes':0,'base_model_loads':0,
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
        'packages':{p:version(p) for p in ('torch','transformers','peft','langgraph')}}
    def save():
        p=output/'report.tmp'; p.write_text(json_text(report)); p.replace(output/'report.json')
    save(); episodes=[]
    try:
        set_seed(config['seed']); torch.backends.cuda.matmul.allow_tf32=False
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        executor=SharedPeftExecutor(model); report['base_model_loads']=1
        backend=OutcomeBackend(PeftTextBackend(executor,tokenizer,max_context_tokens=config['max_context_tokens'],
            max_new_tokens=config['max_new_tokens'],sampling=None,stop_on_json_object=True),max_scoring_tokens=config['max_scoring_tokens'])
        with (output/'episodes.jsonl').open('x') as stream:
            for i,job in enumerate(jobs):
                set_seed(config['seed']); backend.readouts=[]
                account=current_account(jobs[:i],episodes,feed,config,job['context']['observation_time'])
                transform=lambda decision,elapsed,feedback:continuous_feedback(jobs[:i],episodes,job,decision,elapsed,feedback,feed,config['execution_policy'])
                ep=explore_episode(job['context'],job['labels'],feed,config['execution_policy'],DiscoveryRunner(backend,catalogue),account,
                                   feedback_transform=transform)
                ep['readouts']=deepcopy(backend.readouts); episodes.append(ep); stream.write(jsonl([ep])); stream.flush()
                report['completed_episodes']=len(episodes); save()
                print('Exploration',len(episodes),'/',len(jobs),'decision',ep['decision'] is not None,
                    'studies',len(ep['discovery'].get('investigations',[])),'reflection',ep['reflection'] is not None,flush=True)
        account=replay_portfolio(jobs,episodes,feed,config['execution_policy'])
        (output/'portfolio.json').write_text(json_text(account))
        records=[{'episode_id':e['episode_id'],'original_cutoff':e['context']['observation_time'],
            'investigations':e['discovery'].get('investigations',[]),'review':e['discovery'].get('review'),
            'reflection':e['reflection'],'reflection_error':e['reflection_error'],
            'feedback_available_at':e['feedback']['available_at'],'episode_sha256':canonical_hash(e),
            'status':'unvalidated_candidate','fine_tuning_admitted':False} for e in episodes]
        (output/'research_records.jsonl').write_text(jsonl(records))
        require(all(not p.requires_grad for p in model.parameters()),'discovery unfroze weights')
        report.update(status='completed',finished_at=now(),statistics=statistics(episodes),
            goal=target_progress(account,config['objective']),
            all_parameters_frozen=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            limitations=['Learning partition exploration, not independent validation or a demonstrated trading edge.',
                'Catalogue is limited to admitted learning contracts, not all raw archive markets.',
                'Two catalogue searches and at most two preregistered investigations per episode.',
                'Lexical retrieval supports broad browsing; no learned semantic retriever yet.',
                'Daily trade-print sampling may be sparse; retained hypotheses are not validated.',
                'Retrospectives are archived after feedback, not backfilled as online historical memory.',
                'Continuous account uses hypothetical print fills and settlement exits, not live orders or autonomous early exits.',
                'The 10% milestone is cumulative net settled cash, not a promised or annualized return.'])
        report['artifact_hashes']={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name not in ('report.json','report.tmp','audit.json')}
        save()
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=now()); save(); raise
    return report


def audit(output):
    config=strict_json((output/'config.json').read_text()); report=strict_json((output/'report.json').read_text())
    plan=strict_json((output/'plan.json').read_text())
    require(report['status'] == 'completed' and report['plan_sha256'] == sha256_file(output/'plan.json'),'unfinished/changed run')
    for name,digest in {**plan['source_hashes'],**report['artifact_hashes']}.items():
        path=(output/name).resolve(); require(path.is_relative_to(output.resolve()) and sha256_file(path) == digest,'artifact changed')
    jobs,catalogue,feed=prepare(config); episodes=read_rows(output/'episodes.jsonl')
    require(jobs == read_rows(output/'jobs.jsonl') and list(catalogue.rows.values()) == read_rows(output/'catalogue.jsonl')
            and len(jobs) == len(episodes),'discovery source differs')
    for i,(job,ep) in enumerate(zip(jobs,episodes)):
        account=current_account(jobs[:i],episodes[:i],feed,config,job['context']['observation_time'])
        require(account == ep['account_at_observation'],'account state does not reproduce')
        class Replay:
            def __init__(self): self.i=0
            def generate(self,request):
                call=ep['calls'][self.i]; self.i+=1
                require(call['request'] == request,'model request differs')
                if call['output'] is None: raise ValidationError(call['error'])
                return call['output']
        transform=lambda decision,elapsed,feedback:continuous_feedback(jobs[:i],episodes[:i],job,decision,elapsed,feedback,feed,config['execution_policy'])
        replay=Replay(); rebuilt=explore_episode(job['context'],job['labels'],feed,config['execution_policy'],
            DiscoveryRunner(replay,catalogue),account,recorded_latency=ep['decision_seconds'],feedback_transform=transform)
        for key in ('context','decision','decision_error','decision_call_count','discovery','feedback','reflection','reflection_error'):
            require(rebuilt[key] == ep[key],'discovery replay differs: '+key)
        require(replay.i == len(ep['calls']),'unused calls')
        calls={canonical_hash(c['request']):c for c in ep['calls']}
        for row in ep['readouts']:
            require(row['prompt'] == binary_prompt(calls[row['request_sha256']]['request'],row['market_id'])
                and row['probability'] == binary_probability(row['logits_no_yes']),'readout changed')
    account=replay_portfolio(jobs,episodes,feed,config['execution_policy'])
    require(account == strict_json((output/'portfolio.json').read_text())
        and target_progress(account,config['objective']) == report['goal']
        and statistics(episodes) == report['statistics'],'ledger/metrics do not reproduce')
    result={'status':'passed','report_sha256':sha256_file(output/'report.json'),'replayed_episodes':len(episodes),
        'searches_studies_feedback_and_single_account_reproduced':True,'new_model_calls':0,
        'fine_tuning_performed':False,'effectiveness_verified':False,'final_test_opened':False}
    (output/'audit.json').write_text(json_text(result)); return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run')
    for k in ('config','model-manifest','output'): r.add_argument('--'+k,type=Path,required=True)
    sub.add_parser('audit').add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=audit(a.output) if a.command == 'audit' else run(a.config,a.model_manifest,a.output)
    print(json_text({k:v for k,v in result.items() if k not in ('artifact_hashes','episodic_diagnostics_not_portfolio')}))
