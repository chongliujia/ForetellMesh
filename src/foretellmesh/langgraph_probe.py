"""Replay and optionally regenerate fixed archived jobs through LangGraph.

Select the first job per declared arm/role before reading its response quality.
This is an orchestration regression check, never a forecast quality benchmark.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import shutil
import time

from .agent_baseline_data import input_context
from .agent_check import ScriptedBackend
from .capabilities import load_capabilities
from .capability_evaluation import verify_training_run
from .data import strict_json, sha256_file
from .evaluation import code_provenance, json_text
from .langgraph_runtime import LangGraphRunner
from .lora_probe import verify_model_manifest
from .peft_runtime import PeftTextBackend, SharedPeftExecutor
from .schema import ValidationError
from .sft_data import jsonl
from .tool_assisted_evaluation import RecordingBackend, stable_result


def archived_jobs(tool_run: Path, system_run: Path) -> list[dict]:
    jobs=[]
    for path in (tool_run, system_run):
        report=strict_json((path/'report.json').read_text())
        if report['status']!='completed' or sha256_file(path/'results.jsonl')!=report['results_sha256']:
            raise ValidationError('source evaluation incomplete or changed')
        rows=[strict_json(s) for s in (path/'results.jsonl').read_text().splitlines()]
        seen=set()
        for row in rows:
            if row['arm'] not in ('base','candidate'):continue
            if path==tool_run:
                if row['variant']!='tool':continue
                key=(row['arm'],row['role'])
                calls=[{'request':c['request'],'raw_output':c['raw_output']} for c in row['calls']]
                workflow=row['workflow'];payload=row['input']
            else:
                if row['kind']!='system':continue
                key=(row['arm'],'system')
                calls=[{'request':c['request'],'raw_output':c['output']} for c in row['calls']]
                workflow=report['config']['system_workflow'];payload=calls[0]['request']['input']
            if key in seen:continue
            seen.add(key)
            jobs.append({'source_run':str(path.resolve()),'source_report_sha256':sha256_file(path/'report.json'),
                         'sample_id':row['sample_id'],'arm':row['arm'],'workflow':workflow,'input':payload,
                         'expected_calls':calls,'expected_result':stable_result(row['result'])})
        expected={('base','research'),('base','risk'),('candidate','research'),('candidate','risk')} if path==tool_run else {('base','system'),('candidate','system')}
        if seen!=expected:raise ValidationError('incomplete graph regression selection')
    return jobs


def run_graph(job,agents,backend):
    return LangGraphRunner(agents,backend,output_protocol='grounded_json_v3',response_transport='single_json_fence').run(
        input_context(job['input']),workflow=job['workflow'],mode='base' if job['arm']=='base' else 'capability',
        capability_scope=None if job['arm']=='base' else {'research_tool_lora'})


def replay(job,agents):
    responses=defaultdict(list)
    for call in job['expected_calls']:responses[call['request']['agent']].append(call['raw_output'])
    backend=ScriptedBackend(dict(responses),['research_tool_lora'])
    start=time.perf_counter();result=run_graph(job,agents,backend)
    if stable_result(result)!=job['expected_result'] or backend.calls!=[c['request'] for c in job['expected_calls']]:
        raise ValidationError('graph raw replay mismatch')
    return time.perf_counter()-start


def run(tool_run,system_run,agent_config,training_run,bundle,model_manifest,output):
    if output.exists():raise ValidationError('graph probe output exists')
    agents,agent_hash=load_capabilities(agent_config);jobs=archived_jobs(tool_run,system_run)
    trained,tc=verify_training_run(training_run,bundle,agents);model_path,model_hash=verify_model_manifest(model_manifest,tc)
    for job in jobs:replay(job,agents)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    if not torch.cuda.is_available():raise ValidationError('CUDA required for graph regeneration probe')
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    (output/'jobs.jsonl').write_text(jsonl(jobs));(output/'agent_config.json').write_bytes(agent_config.read_bytes())
    plan={'frozen_at':datetime.now(timezone.utc).isoformat(),'jobs_sha256':sha256_file(output/'jobs.jsonl'),
          'agent_config_sha256':agent_hash,'code':code_provenance(),'model_manifest_sha256':model_hash,
          'base_model':tc['model'],'base_revision':tc['model_revision'],'training_report_sha256':sha256_file(training_run/'report.json'),
          'adapter_hashes':trained['adapter_hashes'],'adapter_name':'research_tool_lora','seed':20260921,
          'output_protocol':'grounded_json_v3','response_transport':'single_json_fence','max_context_tokens':2048,'max_new_tokens':384,
          'dtype':'bfloat16','attention':'sdpa','do_sample':False,'max_concurrency':1,'model_instances':1,
          'selection':'first archived job per arm/role, plus first reviewed workflow per arm; no quality filtering'}
    (output/'plan.json').write_text(json_text(plan))
    report={'kind':'langgraph_orchestration_regression','status':'running','plan_sha256':sha256_file(output/'plan.json'),
            'started_at':datetime.now(timezone.utc).isoformat(),'planned_jobs':len(jobs),'completed_jobs':0,
            'packages':{p:version(p) for p in ('langgraph','langchain-core','langgraph-checkpoint','langsmith','torch','transformers','peft')},
            'gpu':torch.cuda.get_device_name(0),'cuda_version':torch.version.cuda,'default_promoted':False,'rl_started':False}
    def save():
        p=output/'report.tmp';p.write_text(json_text(report));p.replace(output/'report.json')
    save();records=[]
    try:
        set_seed(plan['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.empty_cache()
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
                                               device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        model=PeftModel.from_pretrained(base,training_run/'adapter',adapter_name='research_tool_lora',is_trainable=False,local_files_only=True)
        model.gradient_checkpointing_disable();model.config.use_cache=True
        backend=PeftTextBackend(SharedPeftExecutor(model),tokenizer,max_context_tokens=2048,max_new_tokens=384)
        report['generation_config']=model.generation_config.to_dict();report['base_model_loads']=1
        report['generation_started_at']=datetime.now(timezone.utc).isoformat();save()
        with (output/'results.jsonl').open('x') as log:
            for job in jobs:
                recorder=RecordingBackend(backend);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                result=run_graph(job,agents,recorder);torch.cuda.synchronize();wall=time.perf_counter()-start
                comparable=[{'request':c['request'],'raw_output':c['raw_output']} for c in recorder.calls]
                r={'sample_id':job['sample_id'],'arm':job['arm'],'workflow':job['workflow'],'result':result,'calls':recorder.calls,
                   'seconds':wall,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                   'requests_and_raw_outputs_equal':comparable==job['expected_calls'],
                   'validated_result_equal':stable_result(result)==job['expected_result']}
                records.append(r);log.write(jsonl([r]));log.flush();report['completed_jobs']=len(records);save()
        report.update(results_sha256=sha256_file(output/'results.jsonl'),model_calls=sum(len(r['calls']) for r in records),
            all_requests_and_raw_outputs_equal=all(r['requests_and_raw_outputs_equal'] for r in records),
            all_validated_results_equal=all(r['validated_result_equal'] for r in records),
            all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
            total_wall_seconds=sum(r['seconds'] for r in records),
            total_generation_seconds=sum(c['usage']['seconds'] for r in records for c in r['calls']))
        report['status']='passed' if report['all_requests_and_raw_outputs_equal'] and report['all_validated_results_equal'] else 'mismatch'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:report['finished_at']=datetime.now(timezone.utc).isoformat();save()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tool-run','system-run','agent-config','training-run','bundle','model-manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=run(**vars(a));print(json_text(r))
    if r['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
