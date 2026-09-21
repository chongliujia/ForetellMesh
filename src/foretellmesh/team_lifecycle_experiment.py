"""Bounded frozen-8B lifecycle diagnostic and all-trade retrospective, with CPU audit."""
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
from .peft_runtime import PeftTextBackend, SharedPeftExecutor
from .schema import timestamp, ValidationError
from .synthetic_sft import canonical_hash
from .team_discovery_experiment import prepare as prepare_training
from .team_learning import require
from .team_learning_experiment import read_rows
from .team_lifecycle import LifecycleRunner, run_window, validate_timing
from .team_outcome_learning import OutcomeBackend, binary_prompt, binary_probability
from .team_rsi_experiment import now
from .team_trade_experience import trade_experiences, learn_available


def prepare(config):
    require(config['kind']=='team_lifecycle_smoke_v1' and config['selection']=='first_chronological_training_job_no_return_filter'
        and config['retrospective_selection']=='all_filled_trades_from_preserved_reference_no_pnl_filter'
        and config['online_learning'] is True and config['automatic_fine_tuning'] is False
        and config['final_test_opened'] is False and config['effectiveness_verified'] is False,'invalid smoke protocol')
    validate_timing(config['timing'])
    require(type(config['max_reviews']) is int and 1<=config['max_reviews']<=3
        and type(config['review_window_days']) is int and 1<=config['review_window_days']<=14,'unbounded smoke')
    source=Path(config['source_config']);require(sha256_file(source)==config['source_config_sha256'],'source config changed')
    base=strict_json(source.read_text());jobs,cat,feed=prepare_training(base)
    ref=Path(config['reference_run']);require(sha256_file(ref/'report.json')==config['reference_report_sha256'],'reference changed')
    report=strict_json((ref/'report.json').read_text())
    for name in ('jobs.jsonl','episodes.jsonl','portfolio.json','config.json','model_manifest.json'):
        require(sha256_file(ref/name)==report['artifact_hashes'][name],'reference artifact changed')
    oldjobs=read_rows(ref/'jobs.jsonl');episodes=read_rows(ref/'episodes.jsonl');labels={}
    for j in oldjobs:
        require(j['partition']=='train','reference experience must be training only')
        for mid,label in j['labels'].items():
            require(mid not in labels or labels[mid]==label,'inconsistent reference label');labels[mid]=label
    account=strict_json((ref/'portfolio.json').read_text())
    oldpolicy=strict_json((ref/'config.json').read_text())['execution_policy']
    cases=trade_experiences(episodes,account,labels,oldpolicy)
    return base,jobs[0],cat,feed,cases


def run(config_path, output):
    require(not output.exists(),'diagnostic output exists');config=strict_json(config_path.read_text())
    require(output.name==config['run_name'],'run name differs')
    base,job,cat,feed,cases=prepare(config);manifest=Path(config['reference_run'])/'model_manifest.json'
    model_path,_=verify_model_manifest(manifest,base)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'BF16 GPU unavailable')
    output.mkdir(parents=True);shutil.copyfile(config_path,output/'config.json');shutil.copyfile(manifest,output/'model_manifest.json')
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    (output/'job.json').write_text(json_text(job));(output/'reference_trade_experiences.json').write_text(json_text(cases))
    plan={'created_at':now(),'selection_uses_return':False,'config_sha256':sha256_file(config_path),
        'source_hashes':{str(p.relative_to(output)):sha256_file(p) for p in (output/'source_snapshot').rglob('*.py')},
        'purpose':'Interface and all-trade learning diagnostic on already used training events, not an effectiveness test.'}
    (output/'plan.json').write_text(json_text(plan))
    report={'kind':config['kind'],'status':'running','started_at':now(),'plan_sha256':sha256_file(output/'plan.json'),
        'base_model':base['model'],'model_revision':base['model_revision'],'seed':base['seed'],
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
        'packages':{p:version(p) for p in ('torch','transformers','peft','langgraph')},
        'fine_tuning_triggered':False,'effectiveness_verified':False,'final_test_opened':False,'real_orders_sent':0}
    def save(): (output/'report.json').write_text(json_text(report))
    save()
    try:
        set_seed(base['seed']);torch.backends.cuda.matmul.allow_tf32=False
        tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        executor=SharedPeftExecutor(model)
        backend=OutcomeBackend(PeftTextBackend(executor,tok,max_context_tokens=base['max_context_tokens'],
            max_new_tokens=config['max_new_tokens'],sampling=None,stop_on_json_object=True),max_scoring_tokens=base['max_scoring_tokens'])
        runner=LifecycleRunner(backend,cat,config['timing'])
        end=timestamp(job['context']['observation_time'],'start')+timedelta(days=config['review_window_days'])
        window=run_window(job,feed,base['execution_policy'],runner,end=end,max_reviews=config['max_reviews'],learning=True)
        (output/'window.json').write_text(json_text(window));(output/'readouts.json').write_text(json_text(backend.readouts))
        print('Lifecycle window:',window['review_count'],'reviews;',window['observation_account']['entry_count'],'entries;',
              window['observation_account']['early_exit_count'],'early exits',flush=True)
        # Separate retrospective session; never inject these newly generated
        # reference lessons into the earlier historical window or call it held out.
        retrospective=LifecycleRunner(backend,cat,config['timing']);cutoff=now()
        report['retrospective_cutoff']=cutoff;save()
        for case in cases:
            learn_available(retrospective,[case],cutoff,enabled=True)
            (output/'reference_reflections.json').write_text(json_text(list(retrospective.trade_reflections.values())))
            (output/'reference_reflection_calls.json').write_text(json_text(retrospective.calls))
            print('Trade reflection',case['market_id'],case['result']['result_class'],
                  retrospective.trade_reflections.get(case['experience_id'],{}).get('error'),flush=True)
        require(all(not p.requires_grad for p in model.parameters()),'weights unfrozen')
        reflections=list(retrospective.trade_reflections.values())
        allcalls=[c for e in window['episodes'] for c in e['calls']]+window['closing_reflection_calls']+retrospective.calls
        report.update(status='completed',finished_at=now(),base_model_loads=1,all_parameters_frozen=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            reference_trade_count=len(cases),reference_result_classes=dict(Counter(c['result']['result_class'] for c in cases)),
            reflected_trade_count=len(reflections),valid_trade_reflections=sum(r['output'] is not None for r in reflections),
            model_calls=len(allcalls),invalid_calls=sum(c['error'] is not None for c in allcalls),
            model_call_seconds=sum(c['seconds'] for c in allcalls),
            window_summary={'reviews':window['review_count'],'valid_decisions':sum(e['decision'] is not None for e in window['episodes']),
                'entries':window['observation_account']['entry_count'],'early_exits':window['observation_account']['early_exit_count'],
                'settlements':window['observation_account']['settlement_count']},
            limitations=['Already used training events, not new validation.',
                'Synthetic tests exercise winning/losing early exits; the Base may choose no trades in this real-data window.',
                'Trade-print simulations have no order-book depth guarantee.',
                'Model lessons are unverified candidates; no promotion, independent efficacy claim or SFT admission.'])
        report['artifact_hashes']={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!='report.json'}
        save()
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=now());save();raise
    return report


class RecordedBackend:
    def __init__(self,calls): self.calls=calls;self.index=0
    def generate(self,request):
        require(self.index<len(self.calls),'extra replay call');row=self.calls[self.index];self.index+=1
        require(request==row['request'],'model request changed during replay')
        if row['output'] is None:raise ValidationError(row['error'])
        return row['output']


class AuditRunner(LifecycleRunner):
    def restored(self,fn,*args,**kwargs):
        start=len(self.calls);index=self.backend.index
        try:return fn(*args,**kwargs)
        finally:
            for row,expected in zip(self.calls[start:],self.backend.calls[index:self.backend.index]):
                require(all(row[k]==expected[k] for k in ('request','output','error')),'call validation changed')
                row['seconds']=expected['seconds']
    def structured(self,*args,**kwargs):return self.restored(super().structured,*args,**kwargs)
    def call(self,*args,**kwargs):return self.restored(super().call,*args,**kwargs)


def audit(output):
    report=strict_json((output/'report.json').read_text());config=strict_json((output/'config.json').read_text())
    plan=strict_json((output/'plan.json').read_text());require(report['status']=='completed','unfinished diagnostic')
    require(sha256_file(output/'plan.json')==report['plan_sha256'],'plan changed')
    for name,digest in {**plan['source_hashes'],**report['artifact_hashes']}.items():
        p=(output/name).resolve();require(p.is_relative_to(output.resolve()) and sha256_file(p)==digest,'artifact changed')
    base,job,cat,feed,cases=prepare(config)
    require(job==strict_json((output/'job.json').read_text()) and cases==strict_json((output/'reference_trade_experiences.json').read_text()),'inputs changed')
    window=strict_json((output/'window.json').read_text())
    calls=[c for e in window['episodes'] for c in e['calls']]+window['closing_reflection_calls']
    backend=RecordedBackend(calls);runner=AuditRunner(backend,cat,config['timing'])
    rebuilt=run_window(job,feed,base['execution_policy'],runner,
        end=timestamp(job['context']['observation_time'],'start')+timedelta(days=config['review_window_days']),
        max_reviews=config['max_reviews'],recorded_latencies=[e['decision_seconds'] for e in window['episodes']],learning=True)
    require(rebuilt==window and backend.index==len(calls),'window replay differs')
    for row in strict_json((output/'readouts.json').read_text()):
        request=next(c['request'] for c in calls if canonical_hash(c['request'])==row['request_sha256'])
        require(binary_prompt(request,row['market_id'])==row['prompt'] and
            abs(binary_probability(row['logits_no_yes'])-row['probability'])<1e-12,'forecast readout changed')
    reflection_calls=strict_json((output/'reference_reflection_calls.json').read_text());backend=RecordedBackend(reflection_calls)
    runner=AuditRunner(backend,cat,config['timing']);learn_available(runner,cases,report['retrospective_cutoff'],enabled=True)
    require(list(runner.trade_reflections.values())==strict_json((output/'reference_reflections.json').read_text())
        and backend.index==len(reflection_calls),'trade reflections replay differs')
    result={'status':'passed','report_sha256':sha256_file(output/'report.json'),'replayed_model_calls':len(calls)+len(reflection_calls),
        'trade_experience_coverage':len(cases),'new_model_calls':0,'gpu_required':False}
    (output/'audit.json').write_text(json_text(result));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','audit'));p.add_argument('--config',type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    print(json_text(run(args.config,args.output) if args.action=='run' else audit(args.output)))


if __name__=='__main__':main()
