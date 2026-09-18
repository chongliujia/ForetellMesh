"""Freeze and run a two-checkpoint, three-workflow Quant ablation; replay on CPU."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import math
from pathlib import Path
import shutil
import sys
import time

from .agent_baseline_data import input_context
from .agent_check import ScriptedBackend
from .agent_runtime import AgentRunner
from .capabilities import load_capabilities
from .capability_evaluation import verify_training_run
from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .lora_probe import verify_model_manifest
from .paired_evaluation import decode
from .peft_runtime import PeftTextBackend, SharedPeftExecutor, render_agent_prompt
from .schema import ValidationError, fields
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .tool_assisted_diagnostic import read, VARIANTS
from .uncertainty_diagnostic import judge_uncertainty

ADAPTERS={'base':None,'candidate':'research_tool_lora'}


def load_config(path):
    fixed={'schema_version':'1','partition':'validation','arms':list(ADAPTERS),'variants':list(VARIANTS),
           'output_protocol':'grounded_json_v3','response_transport':'single_json_fence',
           'max_context_tokens':2048,'max_new_tokens':384,'do_sample':False,'attention':'sdpa',
           'base_dtype':'bfloat16','max_repairs':1,'training':False,'default_promotion':False}
    c=fields(strict_json(path.read_text()),set(fixed)|{'run_name','seed'},'tool evaluation config')
    if any(type(c[k]) is not type(v) or c[k]!=v for k,v in fixed.items()) or type(c['seed']) is not int or not 0<=c['seed']<2**32:
        raise ValidationError('unsupported tool evaluation config')
    if not isinstance(c['run_name'],str) or not c['run_name'].strip():raise ValidationError('missing run name')
    return c


def run_job(job,agents,backend,c):
    return AgentRunner(agents,backend,output_protocol=c['output_protocol'],response_transport=c['response_transport']).run(
        input_context(job['input']),workflow=job['workflow'],mode='base' if job['arm']=='base' else 'capability',
        capability_scope=None if job['arm']=='base' else {'research_tool_lora'})


def make_jobs(inputs,c,agents):
    if agents['limits']['max_repairs']!=c['max_repairs']:raise ValidationError('repair policy mismatch')
    jobs=[]; cells=[(a,v) for a in c['arms'] for v in c['variants']]
    for i,row in enumerate(inputs):
        offset=i%len(cells)
        for arm,variant in cells[offset:]+cells[:offset]:
            role=row['role'];workflow=('quant_' if variant=='tool' else '')+role
            job={'sample_id':row['sample_id'],'arm':arm,'variant':variant,'role':role,'workflow':workflow,'input':deepcopy(row['variants'][variant])}
            # Capture a real runtime request; dummy outputs are schema-only and never frozen as answers.
            dummy={'unknowns':[],'observation_time':job['input']['observation_time']}
            if role=='research':dummy.update(evidence_ids=[],counter_evidence_ids=[])
            else:dummy['risks']=[]
            b=ScriptedBackend({role:[dummy]},['research_tool_lora']); result=run_job(job,agents,b,c)
            if result['status']!='completed' or len(b.calls)!=1:raise ValidationError('request planning failed')
            job.update(request=b.calls[0],request_sha256=canonical_hash(b.calls[0]));jobs.append(job)
    return jobs


def stable_result(result):
    r=deepcopy(result)
    for key in ('trace','tool_trace'):
        for step in r.get(key,[]):step.pop('seconds',None)
    return r


class RecordingBackend:
    def __init__(self,backend):self.backend=backend;self.available_adapters=backend.available_adapters;self.calls=[]
    def generate(self,request):
        raw=self.backend.generate(request)
        self.calls.append({'request':deepcopy(request),'raw_output':raw,'usage':deepcopy(self.backend.last_usage)})
        return raw


def score_record(record,judge,c,agents):
    calls=record['calls']
    if not calls:raise ValidationError('missing model call')
    first=decode(calls[0]['raw_output'],calls[0]['request'],c,agents['limits']['max_output_chars'])['output']
    final=record['result']['stages'].get(record['role'])
    row={'sample_id':record['sample_id'],'role':record['role'],'input':record['input']}
    return {stage:judge_uncertainty(row,judge,value) for stage,value in [('first',first),('final',final)]}


def summarize(inputs,judges,records,c,agents):
    labels={j['sample_id']:j for j in judges}; ids={r['sample_id'] for r in inputs}
    keys=[(r['sample_id'],r['arm'],r['variant']) for r in records]
    expected={(sid,a,v) for sid in ids for a in ADAPTERS for v in VARIANTS}
    if len(keys)!=len(expected) or set(keys)!=expected or len(labels)!=len(inputs):raise ValidationError('incomplete or duplicate tool evaluation grid')
    scores={(r['sample_id'],r['arm'],r['variant']):score_record(r,labels[r['sample_id']],c,agents) for r in records}
    def aggregate(rows,stage):
        selected=[scores[(r['sample_id'],r['arm'],r['variant'])][stage] for r in rows]
        return {'tasks':len(selected),**{key:sum(bool(s.get(key)) for s in selected) for key in
            ('correct','schema_valid','vocabulary_valid','known_as_unknown','missing_unknowns','out_of_vocabulary')}}
    cells={}
    for arm in ADAPTERS:
        for variant in VARIANTS:
            rows=[r for r in records if r['arm']==arm and r['variant']==variant]
            calls=[call for r in rows for call in r['calls']];seconds=math.fsum(x['usage']['seconds'] for x in calls)
            cells[arm+':'+variant]={'first':aggregate(rows,'first'),'final':aggregate(rows,'final'),
                'by_condition':{cond:{stage:aggregate([r for r in rows if labels[r['sample_id']]['condition']==cond],stage) for stage in ('first','final')}
                                for cond in sorted({j['condition'] for j in judges})},
                'by_group':{group:{stage:aggregate([r for r in rows if labels[r['sample_id']]['event_group_id']==group],stage) for stage in ('first','final')}
                            for group in sorted({j['event_group_id'] for j in judges})},
                'resources':{'model_calls':len(calls),'repair_calls':len(calls)-len(rows),
                    'generation_seconds':seconds,'mean_wall_seconds':math.fsum(r['seconds'] for r in rows)/len(rows),
                    'output_tokens_per_second':sum(x['usage']['output_tokens'] for x in calls)/seconds if seconds else None,
                    'input_tokens':sum(x['usage']['input_tokens'] for x in calls),
                    'output_tokens':sum(x['usage']['output_tokens'] for x in calls),
                    'token_limit_calls':sum(x['usage']['output_reached_token_limit'] for x in calls),
                    'peak_allocated_bytes':max(r['memory']['peak_allocated_bytes'] for r in rows),
                    'mean_tool_seconds':math.fsum(t['seconds'] for r in rows for t in r['result'].get('tool_trace',[]))/len(rows)}}
    contrasts={}
    for arm in ADAPTERS:
        for left,right in [('prose','structured'),('structured','tool'),('prose','tool')]:
            stages={}
            for stage in ('first','final'):
                pairs=[(scores[sid,arm,left][stage]['correct'],scores[sid,arm,right][stage]['correct']) for sid in sorted(ids)]
                stages[stage]={'pairs':len(pairs),'improved':sum(not a and b for a,b in pairs),'regressed':sum(a and not b for a,b in pairs),
                               'both_correct':sum(a and b for a,b in pairs),'both_wrong':sum(not a and not b for a,b in pairs)}
            contrasts[arm+':'+left+'->'+right]=stages
    return {'primary':'first_response_exact_unknown_set_and_valid_schema','secondary':'same_score_after_bounded_schema_repair',
            'event_groups':len({j['event_group_id'] for j in judges}),'cells':cells,'matched_contrasts':contrasts}


def evaluate(cohort,config_path,agent_config,training_run,bundle,model_manifest,output):
    if output.exists():raise ValidationError('evaluation output exists')
    c=load_config(config_path);manifest,inputs,judges=read(cohort);agents,agent_hash=load_capabilities(agent_config)
    trained,tc=verify_training_run(training_run,bundle,agents)
    exclusions=strict_json((cohort/'exclusions.json').read_text())
    if exclusions['sources']['candidate_bundle_manifest_sha256']!=sha256_file(bundle/'manifest.json'):raise ValidationError('wrong excluded training bundle')
    model_path,model_hash=verify_model_manifest(model_manifest,tc)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValidationError('CUDA BF16 required')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    jobs=make_jobs(inputs,c,agents);preflight=[]
    for job in jobs:
        prompt=render_agent_prompt(job['request']);n=len(tokenizer.encode(prompt,add_special_tokens=False))
        if n+c['max_new_tokens']>c['max_context_tokens']:raise ValidationError('prompt budget exceeded')
        preflight.append({'sample_id':job['sample_id'],'arm':job['arm'],'variant':job['variant'],'input_tokens':n,'prompt_sha256':sha256_bytes(prompt.encode())})
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    (output/'config.json').write_bytes(config_path.read_bytes());(output/'agent_config.json').write_bytes(agent_config.read_bytes())
    (output/'jobs.jsonl').write_text(jsonl(jobs));(output/'token_preflight.json').write_text(json_text(preflight))
    plan={'frozen_at':datetime.now(timezone.utc).isoformat(),'planned_jobs':len(jobs),'config_sha256':sha256_file(config_path),
          'agent_config_sha256':agent_hash,'cohort_path':str(cohort.resolve()),'cohort_manifest_sha256':sha256_file(cohort/'manifest.json'),
          'jobs_sha256':sha256_file(output/'jobs.jsonl'),'token_preflight_sha256':sha256_file(output/'token_preflight.json'),
          'model_manifest_path':str(model_manifest.resolve()),'model_manifest_sha256':model_hash,
          'base_model':tc['model'],'base_revision':tc['model_revision'],'code':code_provenance(),
          'training_run':str(training_run.resolve()),'bundle':str(bundle.resolve()),
          'training_report_sha256':sha256_file(training_run/'report.json'),'adapter_hashes':trained['adapter_hashes'],
          'adapter_names':ADAPTERS,'primary':'first_response_exact_unknown_set_and_valid_schema',
          'secondary':'same_score_after_bounded_schema_repair','training':False,'default_promoted':False,'rl_started':False}
    (output/'plan.json').write_text(json_text(plan))
    report={'schema_version':'1','kind':'tool_assisted_evaluation','status':'running','plan_sha256':sha256_file(output/'plan.json'),
            'started_at':datetime.now(timezone.utc).isoformat(),'planned_jobs':len(jobs),'completed_jobs':0,'base_model_loads':0,
            'python_executable':sys.executable,'packages':{p:version(p) for p in ('torch','transformers','peft')},
            'gpu':torch.cuda.get_device_name(0),'cuda_version':torch.version.cuda,'limitations':manifest['limitations'],
            'default_promoted':False,'rl_started':False}
    def save():
        temp=output/'report.tmp';temp.write_text(json_text(report));temp.replace(output/'report.json')
    results=[];save()
    try:
        torch.cuda.set_device(0);set_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        report['base_model_loads']=1
        model=PeftModel.from_pretrained(base,training_run/'adapter',adapter_name=ADAPTERS['candidate'],is_trainable=False,local_files_only=True)
        model.gradient_checkpointing_disable();model.config.use_cache=True
        report['generation_config']=model.generation_config.to_dict()
        backend=PeftTextBackend(SharedPeftExecutor(model),tokenizer,max_context_tokens=c['max_context_tokens'],max_new_tokens=c['max_new_tokens'])
        report['generation_started_at']=datetime.now(timezone.utc).isoformat();save()
        with (output/'results.jsonl').open('x') as log:
            for i,job in enumerate(jobs):
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                recorder=RecordingBackend(backend);result=run_job(job,agents,recorder,c);torch.cuda.synchronize()
                if not recorder.calls or any(t['status']=='backend_error' for t in result['trace']):raise ValidationError('model backend failed')
                if recorder.calls[0]['request']!=job['request']:raise ValidationError('runtime request changed after freeze')
                record={**job,'result':result,'calls':recorder.calls,'seconds':time.perf_counter()-started,
                        'memory':{'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}}
                log.write(jsonl([record]));log.flush();results.append(record);report['completed_jobs']=len(results);save()
                print(json_text({'progress':f'{i+1}/{len(jobs)}','arm':job['arm'],'variant':job['variant'],'status':result['status']}).strip(),flush=True)
        report['metrics']=summarize(inputs,judges,results,c,agents);report['results_sha256']=sha256_file(output/'results.jsonl')
        report['all_parameters_frozen']=all(not p.requires_grad for p in model.parameters());report['status']='completed'
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:report['finished_at']=datetime.now(timezone.utc).isoformat();save()
    return report


def audit(run):
    report=strict_json((run/'report.json').read_text());plan=strict_json((run/'plan.json').read_text())
    if report['status']!='completed' or sha256_file(run/'plan.json')!=report['plan_sha256']:raise ValidationError('incomplete or changed plan')
    c=load_config(run/'config.json');agents,agent_hash=load_capabilities(run/'agent_config.json')
    cohort=Path(plan['cohort_path']);manifest,inputs,judges=read(cohort);jobs=make_jobs(inputs,c,agents)
    for name,key in [('config.json','config_sha256'),('jobs.jsonl','jobs_sha256'),('token_preflight.json','token_preflight_sha256')]:
        if sha256_file(run/name)!=plan[key]:raise ValidationError('frozen artifact changed')
    if agent_hash!=plan['agent_config_sha256'] or sha256_file(cohort/'manifest.json')!=plan['cohort_manifest_sha256']:raise ValidationError('cohort or agents changed')
    chunks=[]
    for file in sorted((run/'source_snapshot/foretellmesh').glob('*.py')):chunks.extend((file.name.encode(),b'\0',file.read_bytes(),b'\0'))
    if sha256_bytes(b''.join(chunks))!=plan['code']['source_sha256'] or code_provenance()['source_sha256']!=plan['code']['source_sha256']:
        raise ValidationError('source mismatch; use archived source for replay')
    trained,_=verify_training_run(Path(plan['training_run']),Path(plan['bundle']),agents)
    if trained['adapter_hashes']!=plan['adapter_hashes'] or sha256_file(Path(plan['training_run'])/'report.json')!=plan['training_report_sha256']:
        raise ValidationError('checkpoint changed')
    if sha256_file(Path(plan['model_manifest_path']))!=plan['model_manifest_sha256']:raise ValidationError('model manifest changed')
    excluded=strict_json((cohort/'exclusions.json').read_text())
    if excluded['sources']['candidate_bundle_manifest_sha256']!=sha256_file(Path(plan['bundle'])/'manifest.json'):raise ValidationError('exclusion bundle changed')
    if [strict_json(s) for s in (run/'jobs.jsonl').read_text().splitlines()]!=jobs:raise ValidationError('jobs changed')
    records=[strict_json(s) for s in (run/'results.jsonl').read_text().splitlines()];tokens=strict_json((run/'token_preflight.json').read_text())
    if len(records)!=len(jobs) or len(tokens)!=len(jobs) or report['completed_jobs']!=len(jobs) or plan['planned_jobs']!=len(jobs) or sha256_file(run/'results.jsonl')!=report['results_sha256']:
        raise ValidationError('results incomplete or changed')
    for job,record,token in zip(jobs,records,tokens):
        if {k:record[k] for k in job}!=job:raise ValidationError('result identity changed')
        b=ScriptedBackend({job['role']:[c['raw_output'] for c in record['calls']]},['research_tool_lora'])
        replay=run_job(job,agents,b,c)
        if stable_result(replay)!=stable_result(record['result']) or b.calls!=[c['request'] for c in record['calls']]:raise ValidationError('runtime replay changed')
        if b.calls[0]!=job['request']:raise ValidationError('first request changed')
        expected={k:job[k] for k in ('sample_id','arm','variant')}
        expected.update(input_tokens=record['calls'][0]['usage']['input_tokens'],prompt_sha256=sha256_bytes(render_agent_prompt(job['request']).encode()))
        if token!=expected:raise ValidationError('preflight mismatch')
        if any(x['usage']['input_tokens']+c['max_new_tokens']>c['max_context_tokens'] for x in record['calls']):raise ValidationError('token budget changed')
    if summarize(inputs,judges,records,c,agents)!=report['metrics']:raise ValidationError('metrics changed')
    if not(manifest['frozen_at']<plan['frozen_at']<report['started_at']<report['generation_started_at']<report['finished_at']):raise ValidationError('freeze ordering changed')
    if report['base_model_loads']!=1 or not report['all_parameters_frozen'] or plan['adapter_names']!=ADAPTERS:raise ValidationError('shared frozen base invariant failed')
    return {'status':'passed','jobs':len(records),'model_calls':sum(len(r['calls']) for r in records),'report_sha256':sha256_file(run/'report.json'),
            'runtime_requests_outputs_and_tool_results_replayed':True,'metrics_exactly_reproduced':True,'default_promoted':False,'rl_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);run=sub.add_parser('run')
    for name in ('cohort','config','agent-config','training-run','bundle','model-manifest','output'):run.add_argument('--'+name,type=Path,required=True)
    check=sub.add_parser('audit');check.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if a.command=='audit':print(json_text(audit(a.run)))
    else:
        args=vars(a);args.pop('command');args['config_path']=args.pop('config');r=evaluate(**args)
        print(json_text({'status':r['status'],'jobs':r['completed_jobs']}))

if __name__=='__main__':main()
