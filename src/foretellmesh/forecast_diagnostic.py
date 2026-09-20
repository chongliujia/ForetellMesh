"""Frozen price ablation and train-only rollout diagnostics; no optimizer updates."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import math
from pathlib import Path
import shutil
import statistics
import sys
import time

from .agent_baseline_data import input_context
from .capabilities import load_capabilities
from .capability_evaluation import verify_training_run
from .data import sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .lora_probe import verify_model_manifest
from .market_development import (DATA_FILES, audit as audit_control, load_config as control_config,
                                 read_inputs, replay as replay_control, rows, runner, without_timing)
from .metrics import score_predictions
from .peft_runtime import PeftTextBackend, SharedPeftExecutor, render_agent_prompt
from .rewards import REWARD_VERSION, brier_reward
from .schema import ValidationError, fields, parse_record
from .sft_data import jsonl
from .synthetic_sft import canonical_hash

FILES = (*DATA_FILES, 'partitions/train.inputs.jsonl')


def load_config(path: Path) -> dict:
    c = strict_json(path.read_text())
    fixed = {'schema_version': '1', 'output_protocol': 'grounded_json_v3',
             'response_transport': 'single_json_fence', 'max_context_tokens': 4096,
             'max_new_tokens': 768, 'max_repairs': 0, 'base_dtype': 'bfloat16', 'attention': 'sdpa',
             'validation_partition': 'validation', 'sampling_partition': 'train',
             'train_rows_per_event': 2, 'candidates_per_input': 4, 'visible_anchors': 3,
             'sampling': {'temperature': .8, 'top_p': .95, 'top_k': 50},
             'reward_version': REWARD_VERSION, 'invalid_reward': -1.0,
             'probability_tolerance': 1e-6, 'meaningful_probability_range': .01,
             'training': False, 'default_promotion': False, 'ece_bins': 10, 'log_loss_epsilon': 1e-15}
    fields(c, set(fixed) | {'run_name', 'seed', 'dataset_report_sha256', 'control_report_sha256'}, 'forecast diagnostic config')
    if (any(type(c[k]) is not type(v) or c[k] != v for k, v in fixed.items())
            or type(c['seed']) is not int or not 0 <= c['seed'] < 2**32
            or not isinstance(c['run_name'], str) or not c['run_name'].strip()
            or any(not isinstance(c[k], str) or len(c[k]) != 64
                   for k in ('dataset_report_sha256', 'control_report_sha256'))):
        raise ValidationError('unsupported forecast diagnostic configuration')
    return c


def read_data(root: Path, config: dict):
    """Only train/validation inputs are parsed; label bytes are hashed, not parsed."""
    manifest, validation, membership = read_inputs(root, config)
    name = 'partitions/train.inputs.jsonl'
    if sha256_file(root/name) != manifest['artifact_hashes'][name]:
        raise ValidationError('training inputs changed')
    train = rows(root/name)
    seen = set()
    for row in train:
        fields(row, {'sample_id', 'input'}, 'training diagnostic input')
        sid = row['sample_id']
        if sid in seen or sid not in membership or membership[sid]['split'] != 'train':
            raise ValidationError('duplicate or non-training input')
        seen.add(sid); input_context(row['input'])
    if not train or seen != {sid for sid, m in membership.items() if m['split'] == 'train'}:
        raise ValidationError('incomplete training partition')
    return manifest, {'train': train, 'validation': validation}, membership


def select_train(inputs: list[dict], membership: dict, config: dict) -> list[dict]:
    """Stable hash selection, first favor distinct contracts; never accepts labels."""
    grouped = defaultdict(list)
    for row in inputs:
        m = membership[row['sample_id']]
        if m['split'] != 'train':raise ValidationError('sampling outside training split')
        grouped[m['event_group_id']].append(row)
    selected = []
    for group in sorted(grouped):
        ordered = sorted(grouped[group], key=lambda r: canonical_hash([config['seed'], r['sample_id']]))
        preferred, remainder, seen = [], [], set()
        for row in ordered:
            event = membership[row['sample_id']]['event_id']
            (remainder if event in seen else preferred).append(row); seen.add(event)
        selected.extend((preferred + remainder)[:config['train_rows_per_event']])
    return selected


def visible_input(row: dict, price: str) -> dict:
    if price not in ('visible', 'hidden'):raise ValidationError('unknown price arm')
    value = deepcopy(row['input'])
    if price == 'hidden':value['market'] = None
    input_context(value)
    return value


def make_jobs(inputs: dict, membership: dict, config: dict) -> list[dict]:
    jobs = []
    def add(row, phase, price, candidate=None):
        job = {'sample_id': row['sample_id'], 'phase': phase, 'price': price, 'candidate': candidate,
               'input_sha256': canonical_hash(visible_input(row, price))}
        # Independent seeds per sample/candidate, paired between price arms.
        job['seed'] = int(canonical_hash([config['seed'], phase, row['sample_id'], candidate])[:8], 16)
        jobs.append(job)
    anchors = sorted(inputs['validation'], key=lambda r: canonical_hash([config['seed'], r['sample_id']]))[:config['visible_anchors']]
    for row in anchors:add(row, 'visible_anchor', 'visible')
    for row in inputs['validation']:add(row, 'price_ablation', 'hidden')
    for i, row in enumerate(select_train(inputs['train'], membership, config)):
        for candidate in range(config['candidates_per_input']):
            for price in (('visible', 'hidden') if (i+candidate) % 2 == 0 else ('hidden', 'visible')):
                add(row, 'train_sampling', price, candidate)
    return jobs


def check_control(root: Path, config: dict):
    """Verify frozen cached outputs without evaluating labels before generation."""
    if sha256_file(root/'report.json') != config['control_report_sha256']:
        raise ValidationError('cached control report changed')
    report = strict_json((root/'report.json').read_text())
    plan = strict_json((root/'plan.json').read_text())
    if (report['status'] != 'completed' or report['plan_sha256'] != sha256_file(root/'plan.json')
            or report['results_sha256'] != sha256_file(root/'results.jsonl')):
        raise ValidationError('invalid cached control')
    for name, digest in plan['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root/name) != digest:
            raise ValidationError('cached control artifact changed')
    cc = control_config(root/'config.json')
    for key in ('dataset_report_sha256', 'output_protocol', 'response_transport', 'max_context_tokens',
                'max_new_tokens', 'max_repairs', 'base_dtype', 'attention'):
        if cc[key] != config[key]:raise ValidationError('cached control configuration differs')
    _, ci, _ = read_inputs(root/'data', cc)
    agents, _ = load_capabilities(root/'agent_config.json')
    rr = rows(root/'results.jsonl'); replay_control(ci, rr, agents, cc)
    return report, plan, [r for r in rr if r['arm'] == 'single_base']


def replay(inputs: dict, membership: dict, results: list, agents: dict, config: dict):
    jobs = make_jobs(inputs, membership, config)
    if len(results) != len(jobs):raise ValidationError('incomplete diagnostic grid')
    indexed = {r['sample_id']: r for part in inputs.values() for r in part}
    for job, result in zip(jobs, results):
        if any(result.get(k) != v for k, v in job.items()):raise ValidationError('diagnostic job changed')
        class Playback:
            available_adapters = set()
            def __init__(self):self.used = 0; self.mismatch = False
            def generate(self, request):
                self.used += 1
                if self.used > len(result['calls']):self.mismatch = True; raise RuntimeError('missing call')
                call = result['calls'][self.used-1]
                if call['request'] != request or call['request_sha256'] != canonical_hash(request):self.mismatch = True
                if call['error_type']:raise type(call['error_type'], (Exception,), {})(call['error'])
                return call['output']
        backend = Playback()
        rebuilt = runner(agents, backend, config).run(input_context(visible_input(indexed[job['sample_id']], job['price'])),
                                                    workflow='single_forecast', mode='base')
        if (backend.mismatch or backend.used != 1 or len(result['calls']) != 1
                or without_timing(rebuilt) != without_timing(result['result'])):
            raise ValidationError('diagnostic raw replay failed')


def prediction(row: dict):
    result = row['result']
    return result['prediction']['probability'] if result['status'] == 'completed' else None


def spread(values: list) -> dict:
    return {'count': len(values), 'mean': statistics.fmean(values) if values else None,
            'std': statistics.pstdev(values) if values else None,
            'range': max(values)-min(values) if values else None,
            'unique': len(set(values))}


def candidate_summary(probabilities: list, outcome: int, config: dict) -> dict:
    valid = [p for p in probabilities if p is not None]
    valid_rewards = [brier_reward(p, outcome) for p in valid]
    rewards = [brier_reward(p, outcome) if p is not None else config['invalid_reward'] for p in probabilities]
    return {'eligible': len(probabilities), 'valid': len(valid), 'probabilities': probabilities,
            'probability_spread': spread(valid), 'valid_reward_spread': spread(valid_rewards),
            'all_reward_spread': spread(rewards), 'rewards': rewards,
            'market_independent_signal': False,  # diversity alone never establishes skill
            'format_only_reward_variation': len(valid) < len(probabilities) and bool(valid)
                and max(valid_rewards)-min(valid_rewards) <= config['probability_tolerance']
                and max(rewards)-min(rewards) > config['probability_tolerance']}


def score(root: Path, inputs: dict, membership: dict, results: list, cached: list, config: dict) -> dict:
    jobs = make_jobs(inputs, membership, config)
    if len(jobs) != len(results) or any(any(r.get(k) != v for k,v in j.items()) for j,r in zip(jobs,results)):
        raise ValidationError('invalid result grid for scoring')
    labels = {}
    for partition, part in inputs.items():
        ll = rows(root/f'partitions/{partition}.labels.jsonl')
        ix = {r['sample_id']: r['label'] for r in ll}
        if len(ix) != len(ll) or set(ix) != {r['sample_id'] for r in part}:raise ValidationError('label identity mismatch')
        for row in part:
            m = membership[row['sample_id']]
            parse_record({**row['input'], 'sample_id': row['sample_id'], 'event_id': m['event_id'],
                          'event_group_id': m['event_group_id'], 'dataset_source': 'pma_diagnostic',
                          'dataset_version': m['dataset_version'], 'label': ix[row['sample_id']]})
        labels.update(ix)
    ids = [r['sample_id'] for r in inputs['validation']]
    control = {r['sample_id']: r for r in cached}
    hidden = {r['sample_id']: r for r in results if r['phase'] == 'price_ablation'}
    if len(control) != len(cached) or set(control) != set(ids) or set(hidden) != set(ids):
        raise ValidationError('incomplete price ablation controls')
    pp = {'visible_cached': [prediction(control[s]) for s in ids], 'hidden': [prediction(hidden[s]) for s in ids],
          'market': [r['input']['market']['probability'] for r in inputs['validation']], 'constant_0_5': [.5]*len(ids)}
    gt = defaultdict(list)
    for row in inputs['train']:gt[membership[row['sample_id']]['event_group_id']].append(labels[row['sample_id']]['outcome'])
    prior = statistics.fmean(statistics.fmean(v) for v in gt.values())
    pp['training_event_prior'] = [prior]*len(ids)
    def scores(p, ix):
        values = score_predictions([p[i] for i in ix], [labels[ids[i]]['outcome'] for i in ix],
                                  ece_bins=config['ece_bins'], log_loss_epsilon=config['log_loss_epsilon'])
        gg = defaultdict(list)
        for i in ix:
            if p[i] is not None:gg[membership[ids[i]]['event_group_id']].append(i)
        by_group = {g: score_predictions([p[i] for i in ii], [labels[ids[i]]['outcome'] for i in ii]) for g,ii in gg.items()}
        values.update(event_group_count=len({membership[ids[i]]['event_group_id'] for i in ix}),
                      covered_event_groups=len(gg), per_event=by_group,
                      event_mean={k: statistics.fmean(s[k] for s in by_group.values()) if by_group else None for k in ('brier','log_loss')})
        return values
    all_ix = list(range(len(ids)))
    common = [i for i in all_ix if all(p[i] is not None for p in pp.values())]
    differences = [abs(pp['visible_cached'][i]-pp['hidden'][i]) for i in common]
    anchors = [r for r in results if r['phase'] == 'visible_anchor']
    anchor_checks = [{'sample_id': r['sample_id'],
                     'raw_equal': r['calls'][0]['output'] == control[r['sample_id']]['calls'][0]['output'],
                     'request_equal': r['calls'][0]['request'] == control[r['sample_id']]['calls'][0]['request'],
                     'probability_equal': prediction(r) == prediction(control[r['sample_id']])} for r in anchors]
    sampled = defaultdict(list)
    for r in results:
        if r['phase'] == 'train_sampling':sampled[r['price'],r['sample_id']].append(r)
    train_stats = {}
    for price in ('visible','hidden'):
        groups = []
        for (arm,sid), rr in sampled.items():
            if arm != price:continue
            stats = candidate_summary([prediction(r) for r in sorted(rr,key=lambda r:r['candidate'])],labels[sid]['outcome'],config)
            groups.append({'sample_id':sid, 'event_group_id':membership[sid]['event_group_id'],**stats})
        valid = sum(g['valid'] for g in groups); total = sum(g['eligible'] for g in groups)
        train_stats[price] = {'sample_groups':len(groups), 'event_groups':len({g['event_group_id'] for g in groups}),
            'valid':valid,'eligible':total,'coverage':valid/total if total else None,
            'groups_with_probability_variation':sum((g['probability_spread']['range'] or 0)>config['probability_tolerance'] for g in groups),
            'groups_with_meaningful_probability_variation':sum((g['probability_spread']['range'] or 0)>config['meaningful_probability_range'] for g in groups),
            'groups_with_valid_reward_variation':sum((g['valid_reward_spread']['range'] or 0)>config['probability_tolerance'] for g in groups),
            'groups_with_format_only_reward_variation':sum(g['format_only_reward_variation'] for g in groups),
            'groups':groups}
    errors = Counter(t.get('validation_error',t.get('error_type','')) for r in results for t in r['result']['trace'] if t['status']!='valid')
    return {'price_ablation':{'full':{a:scores(p,all_ix) for a,p in pp.items()},
            'common_sample_ids':[ids[i] for i in common], 'common':{a:scores(p,common) for a,p in pp.items()},
            'probability_absolute_change':spread(differences),
            'changed_above_tolerance':sum(d>config['probability_tolerance'] for d in differences),
            'hidden_probability_spread':spread([p for p in pp['hidden'] if p is not None]),
            'hidden_exact_half':sum(p==.5 for p in pp['hidden']), 'visible_anchor_checks':anchor_checks,
            'all_visible_anchors_reproduced':bool(anchors) and all(all(v for k,v in a.items() if k!='sample_id') for a in anchor_checks)},
            'training_candidates':train_stats, 'training_event_prior':prior, 'validation_errors':dict(errors)}


def evaluate(dataset, config_path, control_run, agent_config, training_run, bundle, model_manifest, output):
    if output.exists():raise ValidationError('output already exists')
    config = load_config(config_path); manifest, inputs, membership = read_data(dataset,config)
    previous, control_plan, cached = check_control(control_run,config)
    if rows(control_run/'data/partitions/validation.inputs.jsonl') != inputs['validation']:
        raise ValidationError('cached input population differs')
    agents,_ = load_capabilities(agent_config)
    if sha256_file(agent_config) != sha256_file(control_run/'agent_config.json'):raise ValidationError('agent configuration differs')
    trained,tc = verify_training_run(training_run,bundle,agents)
    model_path,model_hash = verify_model_manifest(model_manifest,tc)
    if (model_hash != control_plan['model_manifest_sha256'] or trained['adapter_hashes'] != control_plan['adapter_hashes']):
        raise ValidationError('model or disabled adapter differs from control')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValidationError('BF16 CUDA required')
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    jobs = make_jobs(inputs,membership,config)
    indexed = {r['sample_id']:r for part in inputs.values() for r in part}
    class Capture:
        available_adapters=set()
        def generate(self,request):self.request=deepcopy(request);return '{}'
    preflight=[]
    for job in jobs:
        capture=Capture(); runner(agents,capture,config).run(input_context(visible_input(indexed[job['sample_id']],job['price'])),workflow='single_forecast',mode='base')
        n=len(tokenizer.encode(render_agent_prompt(capture.request),add_special_tokens=False))
        if n+config['max_new_tokens']>config['max_context_tokens']:raise ValidationError('context overflow; no truncation')
        preflight.append({**job,'request_sha256':canonical_hash(capture.request),'input_tokens':n})
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(control_run,output/'cached_control',ignore=shutil.ignore_patterns('__pycache__'))
    for source,name in ((config_path,'config.json'),(agent_config,'agent_config.json'),(dataset/'report.json','data/report.json')):
        dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    for name in FILES:
        dest=output/'data'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(dataset/name,dest)
    (output/'jobs.jsonl').write_text(jsonl(jobs));(output/'token_preflight.json').write_text(json_text(preflight))
    plan={'kind':'price_ablation_and_train_rollout_diagnostic','frozen_at':datetime.now(timezone.utc).isoformat(),
          'config':config,'code':code_provenance(),'model':tc['model'],'model_revision':tc['model_revision'],
          'model_manifest_sha256':model_hash,'dataset_version':manifest['dataset_version'],
          'training':False,'final_test_scored':False,'default_promoted':False,
          'selection':'per train event, hash(seed,sample_id), distinct contracts first, then remaining snapshots',
          'artifact_hashes':{str(p.relative_to(output)):sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    (output/'plan.json').write_text(json_text(plan))
    report={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha256_file(output/'plan.json'),
            'planned_jobs':len(jobs),'completed_jobs':0,'base_model_loads':0,'training_started':False,'default_promoted':False,
            'packages':{p:version(p) for p in ('torch','transformers','peft')},'python_executable':sys.executable,
            'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda}
    if report['packages']!=previous['packages']:raise ValidationError('cached control package versions differ')
    def save():
        temp=output/'report.tmp';temp.write_text(json_text(report));temp.replace(output/'report.json')
    save();results=[]
    try:
        torch.cuda.set_device(0);set_seed(config['seed']);torch.backends.cuda.matmul.allow_tf32=False
        base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
                  dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        report['base_model_loads']=1
        model=PeftModel.from_pretrained(base,training_run/'adapter',adapter_name='research_tool_lora',is_trainable=False,local_files_only=True)
        model.gradient_checkpointing_disable();model.config.use_cache=True
        runtime=SharedPeftExecutor(model)
        class Recording(PeftTextBackend):
            def generate(self,request):
                call={'request':deepcopy(request),'request_sha256':canonical_hash(request),'output':None,'error':None,'error_type':None,'usage':None}
                try:call['output']=super().generate(request);return call['output']
                except Exception as exc:call.update(error=str(exc),error_type=type(exc).__name__);raise
                finally:call['usage']=self.last_usage;self.calls.append(call)
        backends={sampled:Recording(runtime,tokenizer,max_context_tokens=config['max_context_tokens'],max_new_tokens=config['max_new_tokens'],
                                    sampling=config['sampling'] if sampled else None) for sampled in (False,True)}
        report['generation_started_at']=datetime.now(timezone.utc).isoformat();save()
        with (output/'results.jsonl').open('x') as log:
            for job in jobs:
                set_seed(job['seed']);backend=backends[job['phase']=='train_sampling'];backend.calls=[]
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                result=runner(agents,backend,config).run(input_context(visible_input(indexed[job['sample_id']],job['price'])),workflow='single_forecast',mode='base')
                torch.cuda.synchronize()
                record={**job,'result':result,'calls':backend.calls,'seconds':time.perf_counter()-start,
                        'memory':{'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}}
                log.write(jsonl([record]));log.flush();results.append(record);report['completed_jobs']=len(results);save()
                print(json_text({'progress':f'{len(results)}/{len(jobs)}','phase':job['phase'],'price':job['price'],'status':result['status'],'seconds':round(record['seconds'],2)}).strip(),flush=True)
                if any(c['error_type']=='OutOfMemoryError' for c in backend.calls):raise RuntimeError('OOM; partial diagnostic only')
        replay(inputs,membership,results,agents,config)
        audit_control(output/'cached_control')
        report['metrics']=score(output/'data',inputs,membership,results,cached,config)
        report['all_parameters_frozen']=all(not p.requires_grad for p in model.parameters())
        report['results_sha256']=sha256_file(output/'results.jsonl');report['status']='completed'
    except Exception as exc:report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:report['finished_at']=datetime.now(timezone.utc).isoformat();save()
    return report


def audit(run: Path) -> dict:
    report=strict_json((run/'report.json').read_text());plan=strict_json((run/'plan.json').read_text())
    if (report['status']!='completed' or report['plan_sha256']!=sha256_file(run/'plan.json')
            or report['results_sha256']!=sha256_file(run/'results.jsonl')):raise ValidationError('unfinished or changed run')
    for name,digest in plan['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(run/name)!=digest:raise ValidationError('frozen artifact changed')
    config=load_config(run/'config.json');_,inputs,membership=read_data(run/'data',config)
    agents,_=load_capabilities(run/'agent_config.json');results=rows(run/'results.jsonl')
    if config!=plan['config'] or rows(run/'jobs.jsonl')!=make_jobs(inputs,membership,config):raise ValidationError('frozen jobs differ')
    _,_,cached=check_control(run/'cached_control',config);audit_control(run/'cached_control')
    replay(inputs,membership,results,agents,config)
    if score(run/'data',inputs,membership,results,cached,config)!=report['metrics']:raise ValidationError('metrics do not reproduce')
    if (report['completed_jobs']!=len(results) or report['planned_jobs']!=len(results)
            or report['base_model_loads']!=1 or not report['all_parameters_frozen']
            or not plan['frozen_at']<report['started_at']<report['generation_started_at']<report['finished_at']):
        raise ValidationError('execution invariants failed')
    return {'status':'passed','jobs':len(results),'report_sha256':sha256_file(run/'report.json'),
            'raw_outputs_replayed':True,'metrics_exactly_reproduced':True,'final_test_scored':False,'training_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run')
    for name in ('dataset','config','control-run','agent-config','training-run','bundle','model-manifest','output'):r.add_argument('--'+name,type=Path,required=True)
    a=sub.add_parser('audit');a.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if args.command=='audit':print(json_text(audit(args.run)))
    else:
        kw=vars(args);kw.pop('command');kw['config_path']=kw.pop('config')
        report=evaluate(**kw);print(json_text({'status':report['status'],'jobs':report['completed_jobs']}))


if __name__=='__main__':main()
