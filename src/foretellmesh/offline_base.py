"""Offline single-resident Base deployment for historical paper-trading signals."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import shutil
import time

from .agent_baseline_data import input_context
from .capabilities import load_capabilities
from .data import sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .langgraph_runtime import LangGraphRunner
from .lora_probe import verify_model_manifest
from .market_development import read_inputs, rows, without_timing
from .peft_runtime import SharedPeftExecutor, PeftTextBackend, render_agent_prompt
from .schema import ValidationError, fields
from .sft_data import jsonl
from .synthetic_sft import canonical_hash


def load_config(path):
    c = strict_json(path.read_text())
    fixed = {'schema_version':'1','model':'Qwen/Qwen3-8B-Base',
        'model_revision':'49e3418fbbbca6ecbdf9608b4d22e5a407081db4','partition':'validation',
        'workflow':'research_forecast','engine':'langgraph','mode':'base','output_protocol':'grounded_json_v3',
        'response_transport':'single_json_fence','max_context_tokens':4096,'max_new_tokens':768,
        'max_repairs':0,'do_sample':False,'base_dtype':'bfloat16','attention':'sdpa','training':False,'load_adapters':False}
    fields(c,set(fixed)|{'run_name','seed','dataset_report_sha256'},'offline Base config')
    if (any(type(c[k]) is not type(v) or c[k]!=v for k,v in fixed.items())
            or type(c['seed']) is not int or not 0<=c['seed']<2**32):
        raise ValidationError('unsupported offline Base configuration')
    return c


def make_runner(agents,backend,c):
    agents=deepcopy(agents);agents['limits']['max_repairs']=c['max_repairs']
    return LangGraphRunner(agents,backend,output_protocol=c['output_protocol'],response_transport=c['response_transport'])


def replay(inputs,results,agents,c):
    if len(inputs)!=len(results):raise ValidationError('incomplete Base signal population')
    for row,r in zip(inputs,results):
        if r['sample_id']!=row['sample_id'] or r['input_sha256']!=canonical_hash(row['input']):raise ValidationError('Base signal identity changed')
        class Playback:
            available_adapters=set()
            def __init__(self):self.used=0;self.bad=False
            def generate(self,request):
                self.used+=1
                if self.used>len(r['calls']):self.bad=True;raise RuntimeError('missing call')
                call=r['calls'][self.used-1]
                if request!=call['request'] or canonical_hash(request)!=call['request_sha256'] or request['adapter'] is not None:self.bad=True
                if call['error_type']:raise type(call['error_type'],(Exception,),{})(call['error'])
                return call['output']
        backend=Playback()
        rebuilt=make_runner(agents,backend,c).run(input_context(row['input']),workflow=c['workflow'],mode='base')
        if backend.bad or backend.used!=len(r['calls']) or without_timing(rebuilt)!=without_timing(r['result']):
            raise ValidationError('Base signal raw replay failed')


def run(dataset,config_path,agent_config,model_manifest,paper_config,output):
    if output.exists():raise ValidationError('output already exists')
    c=load_config(config_path);manifest,inputs,_=read_inputs(dataset,c)
    inputs=sorted(inputs,key=lambda r:(r['input']['observation_time'],r['sample_id']))
    agents,_=load_capabilities(agent_config)
    model_path,model_hash=verify_model_manifest(model_manifest,c)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValidationError('BF16 CUDA required')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    from .agent_runtime import agent_instruction
    for row in inputs:
        prompt=render_agent_prompt({'instruction':agent_instruction('forecast',c['output_protocol']),'input':row['input'],'upstream':{}})
        if len(tokenizer.encode(prompt,add_special_tokens=False))+2*c['max_new_tokens']>c['max_context_tokens']:
            raise ValidationError('context exceeds fixed budget')
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    for path,name in ((config_path,'config.json'),(agent_config,'agent_config.json'),(paper_config,'paper_config.json')):shutil.copyfile(path,output/name)
    (output/'inputs.jsonl').write_text(jsonl(inputs))
    plan={'kind':'offline_base_multiagent_forecasts','frozen_at':datetime.now(timezone.utc).isoformat(),'config':c,
          'model_manifest_sha256':model_hash,'dataset_version':manifest['dataset_version'],'code':code_provenance(),
          'artifact_hashes':{str(p.relative_to(output)):sha256_file(p) for p in output.rglob('*') if p.is_file()}}
    (output/'plan.json').write_text(json_text(plan))
    report={'status':'running','plan_sha256':sha256_file(output/'plan.json'),'started_at':datetime.now(timezone.utc).isoformat(),
            'planned_jobs':len(inputs),'completed_jobs':0,'training':False,'loaded_adapters':[],
            'packages':{p:version(p) for p in ('torch','transformers','langgraph')},
            'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda}
    def save():
        p=output/'report.tmp';p.write_text(json_text(report));p.replace(output/'report.json')
    save();results=[]
    try:
        set_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
             dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        model.gradient_checkpointing_disable();model.config.use_cache=True
        executor=SharedPeftExecutor(model)
        if executor.available_adapters:raise ValidationError('this deployment must not load adapters')
        class Recording(PeftTextBackend):
            def generate(self,request):
                call={'request':deepcopy(request),'request_sha256':canonical_hash(request),'output':None,'error':None,'error_type':None,'usage':None}
                try:call['output']=super().generate(request);return call['output']
                except Exception as exc:call.update(error=str(exc),error_type=type(exc).__name__);raise
                finally:call['usage']=self.last_usage;self.calls.append(call)
        backend=Recording(executor,tokenizer,max_context_tokens=c['max_context_tokens'],max_new_tokens=c['max_new_tokens'])
        runtime=make_runner(agents,backend,c)
        report.update(base_model_loads=1,generation_started_at=datetime.now(timezone.utc).isoformat());save()
        with (output/'results.jsonl').open('x') as log:
            for row in inputs:
                backend.calls=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                value=runtime.run(input_context(row['input']),workflow=c['workflow'],mode='base')
                torch.cuda.synchronize()
                r={'sample_id':row['sample_id'],'input_sha256':canonical_hash(row['input']),'result':value,'calls':backend.calls,
                   'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                   'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                log.write(jsonl([r]));log.flush();results.append(r);report['completed_jobs']=len(results);save()
                print(f"{len(results)}/{len(inputs)} {value['status']}",flush=True)
                if any(x['error_type']=='OutOfMemoryError' for x in backend.calls):raise RuntimeError('CUDA OOM')
        replay(inputs,results,agents,c)
        report.update(status='completed',results_sha256=sha256_file(output/'results.jsonl'),
                      all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
                      valid_forecasts=sum(r['result']['status']=='completed' for r in results))
    except Exception as exc:report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:report['finished_at']=datetime.now(timezone.utc).isoformat();save()
    return report


def audit(root):
    report=strict_json((root/'report.json').read_text());plan=strict_json((root/'plan.json').read_text())
    if (report['status']!='completed' or report['plan_sha256']!=sha256_file(root/'plan.json')
            or report['results_sha256']!=sha256_file(root/'results.jsonl')):raise ValidationError('unfinished or changed Base run')
    for name,digest in plan['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root/name)!=digest:raise ValidationError('frozen Base artifact changed')
    c=load_config(root/'config.json');agents,_=load_capabilities(root/'agent_config.json')
    inputs=rows(root/'inputs.jsonl');results=rows(root/'results.jsonl');replay(inputs,results,agents,c)
    if (c!=plan['config'] or report['loaded_adapters'] or not report['all_parameters_frozen'] or report['base_model_loads']!=1
            or report['completed_jobs']!=len(inputs) or report['planned_jobs']!=len(inputs)
            or not plan['frozen_at']<report['started_at']<report['generation_started_at']<report['finished_at']):
        raise ValidationError('Base deployment invariants failed')
    return {'status':'passed','jobs':len(results),'raw_outputs_replayed':True,'report_sha256':sha256_file(root/'report.json'),
            'training':False,'loaded_adapters':[]}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run')
    for name in ('dataset','config','agent-config','model-manifest','paper-config','output'):r.add_argument('--'+name,type=Path,required=True)
    a=sub.add_parser('audit');a.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if args.command=='audit':print(json_text(audit(args.run)))
    else:
        kw=vars(args);kw.pop('command');kw['config_path']=kw.pop('config');r=run(**kw)
        print(json_text({'status':r['status'],'completed_jobs':r['completed_jobs']}))


if __name__=='__main__':main()
