"""Paired Base/candidate evaluation on frozen development tasks and workflows."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import sys
import time

from .agent_baseline import tool_call_matches
from .agent_baseline_data import input_context, read_cohort
from .agent_runtime import AgentRunner, agent_instruction, decode_agent_response, validate_agent_output
from .capabilities import load_capabilities
from .capability_training import load_training_config
from .capability_curriculum import EVALUATION_POLICY, select_information_validation, select_training_rows, summarize_information
from .data import sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .lora_probe import verify_model_manifest
from .metrics import score_predictions
from .peft_runtime import PeftTextBackend, SharedPeftExecutor
from .probability_tools import execute_probability_tool
from .research_tool_data import read_research_tool_data
from .schema import ValidationError, fields
from .synthetic_sft import canonical_hash
from .uncertainty_diagnostic import judge_uncertainty, read_uncertainty_cohort

DIAGNOSTIC_PROTOCOLS = ("plain_json_v1", "grounded_json_v2")
PREVIOUS_ADAPTER = "research_tool_previous_lora"


def load_evaluation_config(path: Path) -> dict:
    c = strict_json(path.read_text())
    fixed = {'schema_version':'1', 'capability':'research_tool_lora', 'partition':'validation',
             'role_evaluation':'one_response_per_frozen_task_including_feedback_variants',
             'system_workflow':'reviewed_forecast', 'response_transport':'single_json_fence',
             'max_context_tokens':2048, 'max_new_tokens':384, 'default_promotion':False}
    optional = {'uncertainty_protocols', 'information_cohort'} & set(c) if isinstance(c,dict) else set()
    fields(c,set(fixed)|{'run_name','seed','output_protocol','arms'}|optional,'capability evaluation config')
    if (any(type(c[k]) is not type(v) or c[k]!=v for k,v in fixed.items())
            or type(c['seed']) is not int or not 0<=c['seed']<2**32
            or c['output_protocol'] not in (*DIAGNOSTIC_PROTOCOLS,'grounded_json_v3')
            or c['arms'] not in (['base','candidate'],['base','previous','candidate'])
            or ('uncertainty_protocols' in optional and c['uncertainty_protocols'] not in (list(DIAGNOSTIC_PROTOCOLS),['grounded_json_v3']))
            or ('information_cohort' in optional and (c['information_cohort'] != EVALUATION_POLICY or c['output_protocol'] != 'grounded_json_v3'))
            or ('previous' in c['arms'] and (c['output_protocol']!='grounded_json_v3' or c.get('uncertainty_protocols')!=['grounded_json_v3']))):
        raise ValidationError('unsupported capability evaluation policy')
    if not isinstance(c['run_name'],str) or not c['run_name'].strip():raise ValidationError('missing run name')
    return c


def judge_role(row: dict, output: dict | None) -> dict:
    if output is None:return {'schema_valid':False, 'task_correct':False}
    validate_agent_output(row['request']['agent'], output, input_context(row['request']['input']))
    target = row['target']
    result = {'schema_valid':True, 'unknowns_exact':output.get('unknowns')==target.get('unknowns')}
    if row['request']['agent']=='quant':
        matches = tool_call_matches(output,target)
        result.update(task_correct=matches, arguments_match=matches,
                      result_match=abs(execute_probability_tool(output)['result']-row['oracle']['expected_result'])<=1e-6)
    elif row['request']['agent']=='research':
        expected = set(target['evidence_ids'])|set(target['counter_evidence_ids'])
        cited = set(output['evidence_ids'])|set(output['counter_evidence_ids'])
        result.update(task_correct=all(set(output[k])==set(target[k]) for k in ('evidence_ids','counter_evidence_ids')),
                      expected_id_recall=len(cited&expected)/len(expected),
                      includes_forbidden_evidence=bool(cited&set(row['oracle']['forbidden_evidence_ids'])))
    else:
        def claims(value):
            return {(r['scenario'].strip().rstrip('.。'),tuple(sorted(r['evidence_ids']))) for r in value['risks']}
        expected,actual = claims(target),claims(output)
        cited = {eid for risk in output['risks'] for eid in risk['evidence_ids']}
        result.update(task_correct=actual==expected, extractive_claims_match=actual==expected,
                      nonmatching_extractive_claims=len(actual-expected),
                      includes_forbidden_evidence=bool(cited&set(row['oracle']['forbidden_evidence_ids'])))
    return result


def summarize_capability(role_tasks: list[dict], system_cohort: dict, results: list[dict],
                         uncertainty_cohort: tuple[list[dict], list[dict]] | None = None, *,
                         arms: tuple[str,...] = ('base','candidate'),
                         uncertainty_protocols: tuple[str,...] = DIAGNOSTIC_PROTOCOLS,
                         information_tasks: list[dict] | None = None) -> dict:
    if arms not in (('base','candidate'),('base','previous','candidate')) or uncertainty_protocols not in (DIAGNOSTIC_PROTOCOLS,('grounded_json_v3',)):
        raise ValidationError('unsupported evaluation arms or diagnostic protocols')
    tasks = {r['sample_id']:r for r in role_tasks}
    judges = {r['sample_id']:r for r in system_cohort['judge.jsonl']}
    wanted = {(kind,arm,sid) for kind,ids in (('role',tasks),('system',judges)) for arm in arms for sid in ids}
    if information_tasks is not None:
        information = {row['sample_id']: row for row in information_tasks}
        if not information or len(information) != len(information_tasks):raise ValidationError('invalid information cohort identities')
        wanted.update(('information', arm, sid) for arm in arms for sid in information)
    if uncertainty_cohort is not None:
        probe_inputs, probe_judges = uncertainty_cohort
        probes = {r['sample_id']:r for r in probe_inputs}
        probe_labels = {r['sample_id']:r for r in probe_judges}
        if len(probes)!=len(probe_inputs) or len(probe_labels)!=len(probe_judges) or set(probes)!=set(probe_labels):
            raise ValidationError('uncertainty cohort identities mismatch')
        wanted.update(('uncertainty',arm+':'+protocol,sid) for protocol in uncertainty_protocols
                      for arm in arms for sid in probes)
    indexed = {}
    for row in results:
        key=(row['kind'],row['arm'],row['sample_id'])
        if key not in wanted or key in indexed:raise ValidationError('duplicate or unknown capability result')
        indexed[key]=row
    if set(indexed)!=wanted:raise ValidationError('incomplete capability evaluation')
    def resource(rows):
        calls=[c for r in rows for c in r['calls']]
        usage=[c['usage'] for c in calls if c.get('usage') is not None]
        generation_seconds=math.fsum(u['seconds'] for u in usage)
        return {'examples':len(rows),'model_calls':len(calls),'input_tokens':sum(u['input_tokens'] for u in usage),
                'output_tokens':sum(u['output_tokens'] for u in usage),
                'output_token_limit_calls':sum(u['output_reached_token_limit'] for u in usage),
                'mean_seconds':math.fsum(r['seconds'] for r in rows)/len(rows),
                'generation_seconds':generation_seconds,
                'output_tokens_per_generation_second':sum(u['output_tokens'] for u in usage)/generation_seconds if generation_seconds else None,
                'peak_allocated_bytes':max(r['memory']['peak_allocated_bytes'] for r in rows)}
    report={'role':{},'system':{},'real_forecasting_claim':False,'default_promoted':False}
    for arm in arms:
        rows=[indexed['role',arm,sid] for sid in tasks]
        evaluated={r['sample_id']:judge_role(tasks[r['sample_id']],r['output']) for r in rows}
        groups={name:[sid for sid,r in tasks.items() if r['task']==name] for name in sorted({r['task'] for r in tasks.values()})}
        stats={}
        for task,ids in groups.items():
            stats[task]={'examples':len(ids),'schema_valid':sum(evaluated[i]['schema_valid'] for i in ids),
                        'task_correct':sum(evaluated[i]['task_correct'] for i in ids),
                        'task_accuracy':sum(evaluated[i]['task_correct'] for i in ids)/len(ids),
                        'forbidden_reference_examples':sum(evaluated[i].get('includes_forbidden_evidence',False) for i in ids)}
        report['role'][arm]={'tasks':stats,'details':evaluated,'resources':resource(rows),
                              'native_json_valid':sum(r.get('decoded_transport')=='raw_json' and evaluated[r['sample_id']]['schema_valid'] for r in rows),
                              'fence_decoded_valid':sum(r.get('decoded_transport')=='json_fence' and evaluated[r['sample_id']]['schema_valid'] for r in rows)}
        rows=[indexed['system',arm,sid] for sid in judges]
        ps=[None if r['result']['prediction'] is None else r['result']['prediction']['probability'] for r in rows]
        ys=[judges[r['sample_id']]['outcome'] for r in rows]
        gaps=[abs(p-judges[r['sample_id']]['oracle_probability']) for r,p in zip(rows,ps) if p is not None]
        report['system'][arm]={'scores':score_predictions(ps,ys,ece_bins=5),'resources':resource(rows),
                              'oracle_mae':math.fsum(gaps)/len(gaps) if gaps else None,
                              'completed':sum(p is not None for p in ps),
                              'first_pass_complete':sum(r['result']['status']=='completed' and all(t['attempt']==0 for t in r['result']['trace']) for r in rows),
                              'failure_counts':dict(Counter(r['result'].get('stage','')+':'+r['result'].get('error','') for r in rows if r['result']['status']!='completed'))}
    common=[sid for sid in judges if all(indexed['system',arm,sid]['result']['prediction'] is not None for arm in arms)]
    report['system_common_coverage']={'sample_ids':common,'count':len(common),
        'scores':{arm:score_predictions([indexed['system',arm,sid]['result']['prediction']['probability'] for sid in common],
                                      [judges[sid]['outcome'] for sid in common],ece_bins=5) for arm in arms}}
    if uncertainty_cohort is not None:
        report['uncertainty'] = {}
        for protocol in uncertainty_protocols:
            report['uncertainty'][protocol] = {}
            for arm in arms:
                rows=[indexed['uncertainty',arm+':'+protocol,sid] for sid in probes]
                details={r['sample_id']:judge_uncertainty(probes[r['sample_id']],probe_labels[r['sample_id']],r['output']) for r in rows}
                def counts(ids):
                    return {'examples':len(ids),'correct':sum(details[i]['correct'] for i in ids),
                            'schema_valid':sum(details[i]['schema_valid'] for i in ids),
                            'vocabulary_valid':sum(details[i]['vocabulary_valid'] for i in ids),
                            'known_as_unknown_examples':sum(bool(details[i].get('known_as_unknown')) for i in ids),
                            'missing_unknown_examples':sum(bool(details[i].get('missing_unknowns')) for i in ids)}
                report['uncertainty'][protocol][arm] = {
                    **counts(list(probes)), 'resources':resource(rows), 'details':details,
                    'by_role':{role:counts([sid for sid,r in probes.items() if r['role']==role]) for role in ('research','risk')},
                    'by_condition':{condition:counts([sid for sid,r in probe_labels.items() if r['condition']==condition])
                                    for condition in sorted({r['condition'] for r in probe_labels.values()})}}
    if information_tasks is not None:
        report['information'] = {}
        for arm in arms:
            rows = [indexed['information', arm, sid] for sid in information]
            report['information'][arm] = {**summarize_information(information_tasks, {r['sample_id']: r['output'] for r in rows}),
                                          'resources': resource(rows)}
    return report


def verify_training_run(training_run: Path, bundle: Path, agents: dict) -> tuple[dict,dict]:
    trained=strict_json((training_run/'report.json').read_text())
    tc=load_training_config(training_run/'config.json')
    if (trained['status']!='completed' or trained['checkpoint_saved'] is not True or trained['config']!=tc
            or trained['dataset_manifest_sha256']!=sha256_file(bundle/'manifest.json')
            or trained['config_sha256']!=sha256_file(training_run/'config.json')
            or agents['base_model']!=tc['model'] or agents['base_revision']!=tc['model_revision']):
        raise ValidationError('candidate provenance mismatch or training incomplete')
    adapter=training_run/'adapter'
    if not {'adapter_config.json','adapter_model.safetensors'}<=set(trained['adapter_hashes']):raise ValidationError('missing safe adapter files')
    if set(p.name for p in adapter.iterdir() if p.is_file())!=set(trained['adapter_hashes']):raise ValidationError('adapter file set changed')
    for name,digest in trained['adapter_hashes'].items():
        if Path(name).name!=name or sha256_file(adapter/name)!=digest:raise ValidationError('adapter hash mismatch')
    ac=strict_json((adapter/'adapter_config.json').read_text())
    if (ac['base_model_name_or_path']!=tc['model'] or ac['revision']!=tc['model_revision'] or ac['r']!=tc['lora_r']
            or ac['lora_alpha']!=tc['lora_alpha'] or set(ac['target_modules'])!=set(tc['target_modules'])):
        raise ValidationError('adapter configuration mismatch')
    if 'curriculum' in tc:
        _, parts, _ = read_research_tool_data(bundle)
        selected, expected = select_training_rows(parts['train'], tc['curriculum'])
        if (trained.get('training_selection_sha256') != sha256_file(training_run/'training_selection.json')
                or strict_json((training_run/'training_selection.json').read_text()) != expected
                or trained['tokenized']['train']['examples'] != len(selected)):
            raise ValidationError('training curriculum provenance mismatch')
    return trained,tc


def evaluation_agents(agents: dict, adapter: str) -> dict:
    """Versioned checkpoint alias; retain one capability adapter per request."""
    if adapter not in ('research_tool_lora',PREVIOUS_ADAPTER):raise ValidationError('unknown experiment adapter')
    config=deepcopy(agents)
    if adapter!= 'research_tool_lora':
        config['capabilities'][adapter]=config['capabilities'].pop('research_tool_lora')
        config['agents']={role:adapter if cap=='research_tool_lora' else cap for role,cap in config['agents'].items()}
    return config


def evaluate_capability(bundle: Path, training_run: Path, system_cohort_path: Path, model_manifest: Path,
                        config_path: Path, output: Path, *, uncertainty_path: Path | None = None,
                        previous_training_run: Path | None = None, previous_bundle: Path | None = None) -> dict:
    if output.exists():raise ValidationError('capability evaluation output already exists')
    config=load_evaluation_config(config_path)
    if ('uncertainty_protocols' in config) != (uncertainty_path is not None):
        raise ValidationError('uncertainty configuration and cohort must be supplied together')
    probes = None
    if uncertainty_path is not None:
        probe_manifest,probe_inputs,probe_judges=read_uncertainty_cohort(uncertainty_path)
        probes=(probe_inputs,probe_judges)
    data_manifest,partitions,_=read_research_tool_data(bundle)
    selection=set(strict_json((bundle/'evaluation_ids.json').read_text()))
    role_tasks=[r for r in partitions['validation'] if r['sample_id'] in selection]
    information_tasks = (select_information_validation(partitions['validation'], config['information_cohort'])
                         if 'information_cohort' in config else None)
    _,system_cohort=read_cohort(system_cohort_path)
    agents,_=load_capabilities(system_cohort_path/'agent_config.json')
    trained,tc=verify_training_run(training_run,bundle,agents)
    adapter=training_run/'adapter'
    if (('previous' in config['arms'])!=(previous_training_run is not None)
            or (previous_training_run is None)!=(previous_bundle is None)):
        raise ValidationError('previous checkpoint arm requires its training run and dataset')
    adapter_names={'base':None,'candidate':config['capability']}
    if previous_training_run is not None:
        _,previous_parts,_=read_research_tool_data(previous_bundle)
        previous_trained,_=verify_training_run(previous_training_run,previous_bundle,agents)
        old_ids=set(strict_json((previous_bundle/'evaluation_ids.json').read_text()))
        old_tasks={r['sample_id']:r for r in previous_parts['validation'] if r['sample_id'] in old_ids}
        if set(old_tasks)!=selection or any((r['target'],r['request']['input'])!=(old_tasks[r['sample_id']]['target'],old_tasks[r['sample_id']]['request']['input']) for r in role_tasks):
            raise ValidationError('checkpoint comparisons require the same role targets and model inputs')
        adapter_names['previous']=PREVIOUS_ADAPTER
    model_path,model_hash=verify_model_manifest(model_manifest,tc)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    if not torch.cuda.is_available():raise ValidationError('CUDA is required')
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    (output/'config.json').write_bytes(config_path.read_bytes())
    report={'schema_version':'1','kind':'capability_adapter_validation','status':'running','started_at':datetime.now(timezone.utc).isoformat(),
            'config':config,'config_sha256':sha256_file(config_path),'training_report_sha256':sha256_file(training_run/'report.json'),
            'dataset_manifest_sha256':sha256_file(bundle/'manifest.json'),'system_cohort_manifest_sha256':sha256_file(system_cohort_path/'manifest.json'),
            'model_manifest_sha256':model_hash,'base_model':tc['model'],'base_revision':tc['model_revision'],
            'adapter_hashes':trained['adapter_hashes'],'code':code_provenance(),'python_executable':sys.executable,
            'packages':{p:version(p) for p in ('torch','transformers','peft')},'gpu':torch.cuda.get_device_name(0),'cuda_version':torch.version.cuda,
            'base_model_loads':0,'capability_scope':['research_tool_lora'],'forecast_and_critic_adapter':None,'default_promoted':False,
            'limitations':data_manifest['limitations']+['Role tasks use one response each. Repair tasks are preconstructed feedback variants, not an adaptive recovery rate.',
                'Risk scoring measures an extractive target contract, not arbitrary paraphrase factuality.',
                'System forecasting inputs include mathematical calculation evidence; scores use synthetic outcomes.',
                'Development validation only; no final-test selection, market performance claim or checkpoint promotion.']}
    if probes is not None:
        report['uncertainty_cohort_manifest_sha256']=sha256_file(uncertainty_path/'manifest.json')
        report['limitations'].extend(probe_manifest['limitations'])
    if information_tasks is not None:
        (output/'information_evaluation_ids.json').write_text(json_text([r['sample_id'] for r in information_tasks]))
        report['information_evaluation_ids_sha256'] = sha256_file(output/'information_evaluation_ids.json')
        report['limitations'].extend([
            'Information-state validation uses 128 rows from eight disjoint development groups, not 128 independent events.',
            'Each group uses one language; role and naming format are counterbalanced across the two languages, not fully crossed within a group.',
            'Information-state correctness requires exact unknown-name sets and grounded role fields; it is not an open-ended factuality score.'])
    if previous_training_run is not None:
        report.update(previous_training_report_sha256=sha256_file(previous_training_run/'report.json'),
                      previous_dataset_manifest_sha256=sha256_file(previous_bundle/'manifest.json'),
                      previous_adapter_hashes=previous_trained['adapter_hashes'],adapter_names=adapter_names)
    def save():
        temp=output/'report.tmp';temp.write_text(json_text(report));temp.replace(output/'report.json')
    save();results=[]
    try:
        torch.cuda.set_device(0);set_seed(config['seed']);torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
                                                device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        report['base_model_loads']=1
        model=PeftModel.from_pretrained(base,adapter,adapter_name=config['capability'],is_trainable=False,local_files_only=True)
        if previous_training_run is not None:
            model.load_adapter(previous_training_run/'adapter',adapter_name=PREVIOUS_ADAPTER,is_trainable=False,local_files_only=True)
        model.gradient_checkpointing_disable();model.config.use_cache=True
        runtime=SharedPeftExecutor(model)
        class RecordingBackend(PeftTextBackend):
            def __init__(self):
                super().__init__(runtime,tokenizer,max_context_tokens=config['max_context_tokens'],max_new_tokens=config['max_new_tokens'])
                self.calls=[]
            def generate(self,request):
                record={'request':deepcopy(request),'request_sha256':canonical_hash(request),'output':None,'usage':None,'error':None}
                try:
                    response=super().generate(request);record['output']=response;return response
                except Exception as exc:
                    record['error']=f'{type(exc).__name__}: {exc}';raise
                finally:
                    record['usage']=self.last_usage;self.calls.append(record)
        backend=RecordingBackend()
        runners={arm:AgentRunner(evaluation_agents(agents,name or config['capability']),backend,
                    output_protocol=config['output_protocol'],response_transport=config['response_transport']) for arm,name in adapter_names.items()}
        jobs=[]
        for kind,rows in (('role',role_tasks),('system',system_cohort['inputs.jsonl'])):
            for index,row in enumerate(rows):
                offset=index%len(config['arms'])
                arms=config['arms'][offset:]+config['arms'][:offset]
                jobs.extend((kind,arm,row) for arm in arms)
        if information_tasks is not None:
            for index, row in enumerate(information_tasks):
                offset = index % len(config['arms'])
                jobs.extend(('information', arm, row) for arm in config['arms'][offset:]+config['arms'][:offset])
        if probes is not None:
            combinations=[arm+':'+protocol for protocol in config['uncertainty_protocols'] for arm in config['arms']]
            for index,row in enumerate(probe_inputs):
                offset=index%len(combinations)
                jobs.extend(('uncertainty',arm,row) for arm in combinations[offset:]+combinations[:offset])
        report['planned_jobs']=len(jobs);save()
        with (output/'results.jsonl').open('x') as log:
            for index,(kind,arm,row) in enumerate(jobs):
                backend.calls=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                record={'kind':kind,'arm':arm,'sample_id':row['sample_id']}
                if kind in ('role','information','uncertainty'):
                    selected_arm,protocol=arm.split(':') if kind=='uncertainty' else (arm,config['output_protocol'])
                    request=deepcopy(row['request']) if kind in ('role','information') else {'agent':row['role'],'input':deepcopy(row['input']),'upstream':{}}
                    request['instruction']=agent_instruction(request['agent'],protocol)
                    request['adapter']=adapter_names[selected_arm]
                    record.update(output=None,decoded_transport=None,error=None)
                    try:
                        raw=backend.generate(request)
                        if len(raw)>agents['limits']['max_output_chars']:raise ValidationError('output budget exceeded')
                        decoded,transport=decode_agent_response(raw,config['response_transport'])
                        record['output']=validate_agent_output(request['agent'],decoded,input_context(request['input']))
                        record['decoded_transport']=transport
                    except (ValueError,TypeError) as exc:record['error']=f'{type(exc).__name__}: {exc}'
                else:
                    kwargs={} if arm=='base' else {'mode':'capability','capability_scope':{adapter_names[arm]}}
                    record['result']=runners[arm].run(input_context(row['input']),workflow=config['system_workflow'],**kwargs)
                torch.cuda.synchronize();record.update(seconds=time.perf_counter()-start,calls=backend.calls,
                    memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
                log.write(json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n');log.flush();results.append(record)
                report['completed_jobs']=len(results);save()
                print(json.dumps({'progress':f'{index+1}/{len(jobs)}','kind':kind,'arm':arm,'sample_id':row['sample_id'],
                                  'valid':record['output'] is not None if kind in ('role','information','uncertainty') else record['result']['status']=='completed',
                                  'seconds':round(record['seconds'],2)}),flush=True)
                if any(c['error'] and 'OutOfMemoryError' in c['error'] for c in backend.calls):raise RuntimeError('CUDA OOM; partial results only')
        report['metrics']=summarize_capability(role_tasks,system_cohort,results,probes,arms=tuple(config['arms']),
                                               uncertainty_protocols=tuple(config.get('uncertainty_protocols',DIAGNOSTIC_PROTOCOLS)),
                                               information_tasks=information_tasks)
        report['results_sha256']=sha256_file(output/'results.jsonl');report['status']='completed'
        report['all_parameters_frozen']=all(not p.requires_grad for p in model.parameters())
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        report['finished_at']=datetime.now(timezone.utc).isoformat();save()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','training-run','system-cohort','model-manifest','config','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--uncertainty-cohort',type=Path)
    parser.add_argument('--previous-training-run',type=Path)
    parser.add_argument('--previous-bundle',type=Path)
    a=parser.parse_args()
    r=evaluate_capability(a.bundle,a.training_run,a.system_cohort,a.model_manifest,a.config,a.output,uncertainty_path=a.uncertainty_cohort,
                          previous_training_run=a.previous_training_run,previous_bundle=a.previous_bundle)
    print(json_text({'status':r['status'],'jobs':r['completed_jobs'],'default_promoted':r['default_promoted']}))


if __name__=='__main__':main()
