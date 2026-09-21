"""Frozen paired context diagnostic for one already exposed task-state failure.

ABBA greedy repeats check reproducibility/order, not independent sample size.
Only upstream.training_history differs; no production loop is modified here.
"""
import argparse
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import shutil

from .data import strict_json,sha256_file
from .evaluation import json_text
from .synthetic_sft import canonical_hash
from .team_learning import require
from .team_discovery import DiscoveryRunner,decode_agent_response
from .team_task_negotiation import validate_assessment
from .team_research_loop import prepare as prepare_training,relation_fingerprint
from .team_lifecycle_experiment import RecordedBackend,AuditRunner
from .team_rsi_experiment import now

ORDER=['full_history','structured_history','structured_history','full_history']
TIMING={'min_review_seconds':3600,'max_review_seconds':2592000,'max_signal_valid_seconds':604800,
        'max_order_ttl_seconds':86400,'failure_review_seconds':86400}


def error_code(error):
    text=error or ''
    for token,code in [('task handoff mismatch:','task_binding_mismatch'),('duplicate_relation:','duplicate_relation'),
        ('assess every original contract','assessment_contract_coverage'),('source quantity/unit mismatch','source_measurement_mismatch')]:
        if token in text:return code
    return 'other_validation_failure' if text else None


def compact_history(history,past_calls,catalogue):
    receipts=[]
    feedbacks=[]
    if history.get('prior_training_failure'):feedbacks.append(history['prior_training_failure']['feedback'])
    feedbacks.extend(row['feedback'] for row in history.get('recent_feedback',[]))
    for feedback in feedbacks:
        receipts.append({'status':feedback['status'],'feedback_sha256':feedback['feedback_sha256'],
            'failure_codes':[error_code(f['error']) for f in feedback.get('failure_details',[])],
            'reviewer_opinions_are_not_verified_facts':True})
    rejected=[]
    for call in past_calls:
        if not call['error'] or call['request']['agent']!='relation_researcher':continue
        fingerprint=None
        try:
            from .team_multi_input import validate_named_relation,validate_relation
            value,_=decode_agent_response(call['output'],'single_json_fence')
            require(isinstance(value,dict),'candidate must be an object')
            checked=(validate_named_relation if 'prediction_kind' in (value.get('relation') or {}) else validate_relation)(value,catalogue)
            if checked['relation'] is not None:fingerprint=relation_fingerprint(checked['relation'])
        except (ValueError,TypeError,KeyError):pass
        rejected.append({'error_code':error_code(call['error']), 'failed_experiment_fingerprint':fingerprint,
            'output_sha256':canonical_hash(call['output']), 'request_sha256':canonical_hash(call['request'])})
    return {'kind':'structured_failure_history_v1','past_feedback_receipts':receipts,'rejected_experiments':rejected,
        'full_history_sha256':canonical_hash(history),'past_calls_sha256':canonical_hash(past_calls),
        'full_history_retained_in_audit':True,'current_task_authority':'input.original_task',
        'scope':'Past failures only; use the current task and current data catalogue supplied in input.',
        'predictive_effectiveness_verified':False}


def prepare(config):
    require(config['kind']=='team_context_comparison_v1' and config['order']==ORDER,'invalid fixed comparison design')
    require(config['max_new_tokens']==1536 and config['max_context_tokens']==12288,'generation budgets changed')
    root=Path(config['source_run']);report=strict_json((root/'report.json').read_text())
    require(sha256_file(root/'report.json')==config['source_report_sha256'],'source report changed')
    require(report['status']=='completed' and report['all_parameters_frozen'] and not report['development_opened']
        and not report['final_test_opened'],'source must be completed frozen training research')
    for name in ('calls.json','catalogue.json','config.json','attempt_02.json'):
        require(sha256_file(root/name)==report['artifact_hashes'][name],'source artifact changed')
    source_config=strict_json((root/'config.json').read_text())
    base,manifest,cat,_,_,_,_,_=prepare_training(source_config)
    require(cat==strict_json((root/'catalogue.json').read_text()),'admitted catalogue changed')
    attempt=strict_json((root/'attempt_02.json').read_text());calls=strict_json((root/'calls.json').read_text())
    candidates=[i for i in range(attempt['call_start'],attempt['call_end'])
        if calls[i]['request']['agent']=='research_task_assessor' and 'repair' not in calls[i]['request']]
    require(len(candidates)==1,'expected one initial second-round assessor request')
    i=candidates[0];original=deepcopy(calls[i]['request'])
    require(set(original['upstream'])=={'training_history'},'unexpected assessor upstream')
    require(original['input']['original_task']==attempt['task_negotiation']['original_task'],'current task changed')
    require(set(original['input']['original_task']['market_ids'])=={'251132','251306'},'fixed case task changed')
    require(original['input']['data_catalogue']==attempt['data_catalogue']['result'],'current directory changed')
    compact=deepcopy(original)
    compact['upstream']['training_history']=compact_history(original['upstream']['training_history'],calls[:i],cat)
    bundle={'source_report_sha256':config['source_report_sha256'],'source_call_index':i,
        'source_request_sha256':canonical_hash(original),'arms':{'full_history':original,'structured_history':compact},
        'source_initial_output':calls[i]['output'],
        'limitations':['One already exposed training task; no independent validation.',
            'History content and length change together; not a pure test of raw JSON alone.',
            'Greedy repeats test reproducibility/order, not independent evidence.']}
    return base,manifest,cat,bundle


def output_metrics(raw,task,catalogue):
    value=None;decode_error=None;validation_error=None;ids=None
    try:
        value,_=decode_agent_response(raw,'single_json_fence')
        rows=value.get('contract_assessments') if isinstance(value,dict) else None
        if isinstance(rows,list):ids=[r.get('market_id') if isinstance(r,dict) else None for r in rows]
        validate_assessment(value,task,catalogue)
    except (ValueError,TypeError,KeyError) as exc:
        if value is None:decode_error=str(exc)
        else:validation_error=str(exc)
    current=set(task['market_ids'])
    return {'schema_and_routing_valid':decode_error is None and validation_error is None,
        'decode_error':decode_error,'validation_error':validation_error,'assessed_ids':ids,
        'extra_assessed_ids':sorted({k for k in (ids or []) if isinstance(k,str)}-current),
        'missing_assessed_ids':sorted(current-{k for k in (ids or []) if isinstance(k,str)}),
        'proposal_semantics_verified':False,'effectiveness_verified':False}


def compare(runner,bundle,catalogue,save=lambda r:None):
    results=[]
    for number,arm in enumerate(ORDER,1):
        request=deepcopy(bundle['arms'][arm]);start=len(runner.calls);value=None;error=None
        try:
            value=runner.structured(request['agent'],request['instruction'],request['input'],request['upstream'],
                lambda v:validate_assessment(v,request['input']['original_task'],catalogue))
        except (ValueError,TypeError,KeyError) as exc:error=str(exc)
        calls=runner.calls[start:]
        record={'trial':number,'arm':arm,'request_sha256':canonical_hash(request),'call_start':start,
            'call_end':len(runner.calls),'result':value,'error':error,
            'initial':output_metrics(calls[0]['output'],request['input']['original_task'],catalogue),
            'final':output_metrics(calls[-1]['output'],request['input']['original_task'],catalogue),
            'initial_output_sha256':canonical_hash(calls[0]['output']),
            'initial_matches_archived_output':calls[0]['output']==bundle['source_initial_output'],
            'seconds':sum(c['seconds'] for c in calls)}
        record['trial_sha256']=canonical_hash(record);save(record);results.append(record)
        print('Context comparison',number,arm,'initial valid',record['initial']['schema_and_routing_valid'],
              'final valid',record['final']['schema_and_routing_valid'],flush=True)
    return results


def summarize(results):
    arms={}
    for arm in ('full_history','structured_history'):
        rows=[r for r in results if r['arm']==arm]
        arms[arm]={'repeats':len(rows),'initial_valid':sum(r['initial']['schema_and_routing_valid'] for r in rows),
            'valid_after_bounded_repair':sum(r['result'] is not None for r in rows),
            'initial_stale_task_ids':sum(bool(r['initial']['extra_assessed_ids']) for r in rows),
            'initial_outputs_identical':len({r['initial_output_sha256'] for r in rows})==1,
            'calls':sum(r['call_end']-r['call_start'] for r in rows)}
    return {'arms':arms,'independent_cases':0,'exposed_training_cases':1,'new_predictions_scored':0,
        'effectiveness_verified':False,'production_protocol_changed':False}


def run(config_path,output):
    config=strict_json(config_path.read_text());require(not output.exists() and output.name==config['run_name'],'output exists/name differs')
    base,manifest,cat,bundle=prepare(config)
    from .lora_probe import verify_model_manifest
    from .peft_runtime import SharedPeftExecutor,PeftTextBackend,render_agent_prompt
    model_path,_=verify_model_manifest(manifest,base)
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'BF16 GPU unavailable')
    output.mkdir(parents=True);shutil.copyfile(config_path,output/'config.json');shutil.copyfile(manifest,output/'model_manifest.json')
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    (output/'inputs.json').write_text(json_text(bundle));(output/'catalogue.json').write_text(json_text(cat))
    registration={'registered_at':now(),'config_sha256':sha256_file(config_path),'inputs_sha256':canonical_hash(bundle),
        'catalogue_sha256':canonical_hash(cat),'order':ORDER,'max_model_calls':8,'score_based_restarts':False,
        'source_hashes':{str(p.relative_to(output)):sha256_file(p) for p in (output/'source_snapshot').rglob('*.py')}}
    (output/'registration.json').write_text(json_text(registration))
    report={'kind':config['kind'],'status':'running','started_at':now(),'registration_sha256':sha256_file(output/'registration.json'),
        'base_model':base['model'],'model_revision':base['model_revision'],'seed':base['seed'],'gpu':torch.cuda.get_device_name(0),
        'cuda':torch.version.cuda,'packages':{p:version(p) for p in ('torch','transformers','peft')},
        'fine_tuning_triggered':False,'development_opened':False,'final_test_opened':False,'real_orders_sent':0,
        'effectiveness_verified':False,'production_protocol_changed':False}
    def save(): (output/'report.json').write_text(json_text(report))
    save();runner=None
    try:
        set_seed(base['seed']);torch.backends.cuda.matmul.allow_tf32=False
        tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        report['initial_prompt_tokens']={arm:len(tok.encode(render_agent_prompt(req),add_special_tokens=False)) for arm,req in bundle['arms'].items()}
        require(all(n+config['max_new_tokens']<=config['max_context_tokens'] for n in report['initial_prompt_tokens'].values()),'initial context overflow')
        save()
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        runner=DiscoveryRunner(PeftTextBackend(SharedPeftExecutor(model),tok,max_context_tokens=config['max_context_tokens'],
            max_new_tokens=config['max_new_tokens'],sampling=None,stop_on_json_object=True),None)
        def save_trial(record):
            (output/f'trial_{record["trial"]:02}.json').write_text(json_text(record))
            (output/'calls.json').write_text(json_text(runner.calls));report['completed_trials']=record['trial'];save()
        results=compare(runner,bundle,cat,save_trial)
        require(all(not p.requires_grad for p in model.parameters()),'weights unfrozen')
        report.update(status='completed',finished_at=now(),summary=summarize(results),all_parameters_frozen=True,base_model_loads=1,
            model_calls=len(runner.calls),invalid_calls=sum(c['error'] is not None for c in runner.calls),
            model_call_seconds=sum(c['seconds'] for c in runner.calls),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report['artifact_hashes']={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!='report.json'};save()
    except BaseException as exc:
        if runner is not None:(output/'calls.json').write_text(json_text(runner.calls))
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=now());save();raise
    return report


def audit(output):
    report=strict_json((output/'report.json').read_text());registration=strict_json((output/'registration.json').read_text())
    require(report['status']=='completed' and report['registration_sha256']==sha256_file(output/'registration.json'),'registration changed')
    for name,digest in {**registration['source_hashes'],**report['artifact_hashes']}.items():
        path=(output/name).resolve();require(path.is_relative_to(output.resolve()) and sha256_file(path)==digest,'artifact changed')
    config=strict_json((output/'config.json').read_text());_,_,cat,bundle=prepare(config)
    require(canonical_hash(bundle)==registration['inputs_sha256'] and canonical_hash(cat)==registration['catalogue_sha256'],'inputs changed')
    require(registration['config_sha256']==sha256_file(output/'config.json') and registration['order']==ORDER,'design changed')
    calls=strict_json((output/'calls.json').read_text());backend=RecordedBackend(calls);runner=AuditRunner(backend,None,TIMING)
    def verify(record):require(record==strict_json((output/f'trial_{record["trial"]:02}.json').read_text()),'trial differs')
    results=compare(runner,bundle,cat,verify)
    require(backend.index==len(calls) and runner.calls==calls and summarize(results)==report['summary'],'replay differs')
    require(len(calls)<=registration['max_model_calls'] and len(list(output.glob('trial_*.json')))==4,'budget changed')
    result={'status':'passed','report_sha256':sha256_file(output/'report.json'),'model_calls_replayed':len(calls),
        'trials_replayed':4,'new_model_calls':0,'gpu_required':False}
    (output/'audit.json').write_text(json_text(result));return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','run','audit'))
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.action=='preflight':
        _,_,_,bundle=prepare(strict_json(args.config.read_text()));print(json_text({'source_call_index':bundle['source_call_index'],
            'initial_history_chars':{k:len(json_text(v['upstream'])) for k,v in bundle['arms'].items()},'order':ORDER}))
    else:print(json_text(run(args.config,args.output) if args.action=='run' else audit(args.output)))

if __name__=='__main__':main()
