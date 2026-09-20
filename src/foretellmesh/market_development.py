"""Frozen real validation comparison; labels enter only CPU scoring, never requests."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import math
from pathlib import Path
import shutil
import sys
import time

from .agent_baseline_data import input_context
from .agent_runtime import AgentRunner, agent_instruction
from .capabilities import load_capabilities
from .capability_evaluation import verify_training_run
from .data import sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .lora_probe import verify_model_manifest
from .metrics import score_predictions
from .peft_runtime import PeftTextBackend, SharedPeftExecutor, render_agent_prompt
from .schema import ValidationError, fields, parse_record
from .sft_data import jsonl
from .synthetic_sft import canonical_hash

ARMS = {
    'single_base': {'workflow': 'single_forecast', 'mode': 'base'},
    'multi_base': {'workflow': 'research_forecast', 'mode': 'base'},
    'multi_research_lora': {'workflow': 'research_forecast', 'mode': 'capability',
                            'capability_scope': {'research_tool_lora'}},
}
DATA_FILES = ('membership.jsonl', 'partitions/validation.inputs.jsonl',
              'partitions/validation.labels.jsonl', 'partitions/train.labels.jsonl')


def rows(path):
    return [strict_json(line) for line in path.read_text().splitlines() if line.strip()]


def load_config(path):
    fixed = {'schema_version': '1', 'partition': 'validation', 'arms': list(ARMS),
             'output_protocol': 'grounded_json_v3', 'response_transport': 'single_json_fence',
             'max_context_tokens': 4096, 'max_new_tokens': 768, 'max_repairs': 0,
             'do_sample': False, 'base_dtype': 'bfloat16', 'attention': 'sdpa',
             'training': False, 'default_promotion': False, 'ece_bins': 10, 'log_loss_epsilon': 1e-15}
    c = fields(strict_json(path.read_text()), set(fixed) | {'run_name', 'seed', 'dataset_report_sha256'}, 'development config')
    if (any(type(c[k]) is not type(v) or c[k] != v for k, v in fixed.items())
            or type(c['seed']) is not int or not 0 <= c['seed'] < 2**32
            or not isinstance(c['run_name'], str) or not c['run_name'].strip()
            or not isinstance(c['dataset_report_sha256'], str) or len(c['dataset_report_sha256']) != 64):
        raise ValidationError('unsupported development evaluation configuration')
    return c


def read_inputs(root, config):
    """Hash labels without parsing them; never open train/test model inputs."""
    if sha256_file(root/'report.json') != config['dataset_report_sha256']:
        raise ValidationError('dataset report changed')
    manifest = strict_json((root/'report.json').read_text())
    for name in DATA_FILES:
        if sha256_file(root/name) != manifest['artifact_hashes'][name]:
            raise ValidationError('dataset artifact changed: ' + name)
    membership = rows(root/'membership.jsonl')
    index = {}
    groups = {}
    for row in membership:
        sid, group, split = row['sample_id'], row['event_group_id'], row['split']
        if (sid in index or split not in ('train', 'validation', 'test')
                or groups.get(group, split) != split or row['dataset_version'] != manifest['dataset_version']):
            raise ValidationError('duplicate identity or event split leakage')
        index[sid] = row; groups[group] = split
    inputs = rows(root/'partitions/validation.inputs.jsonl')
    seen = set()
    for row in inputs:
        fields(row, {'sample_id', 'input'}, 'development input')
        sid = row['sample_id']
        if sid in seen or sid not in index or index[sid]['split'] != 'validation':
            raise ValidationError('duplicate or non-validation input')
        seen.add(sid)
        input_context(row['input'])  # strict allowlist and timestamp validation
    if not inputs or seen != {sid for sid, m in index.items() if m['split'] == 'validation'}:
        raise ValidationError('incomplete validation partition')
    return manifest, inputs, index


def make_jobs(inputs):
    jobs = []
    arms = list(ARMS)
    for i, row in enumerate(inputs):
        offset = i % len(arms)
        for arm in arms[offset:] + arms[:offset]:
            jobs.append({'sample_id': row['sample_id'], 'arm': arm, 'input_sha256': canonical_hash(row['input'])})
    return jobs


def runner(agents, backend, config):
    agents = deepcopy(agents)
    agents['limits']['max_repairs'] = config['max_repairs']
    return AgentRunner(agents, backend, output_protocol=config['output_protocol'],
                       response_transport=config['response_transport'])


def without_timing(value):
    if isinstance(value, dict):return {k: without_timing(v) for k, v in value.items() if k != 'seconds'}
    if isinstance(value, list):return [without_timing(v) for v in value]
    return value


def replay(inputs, results, agents, config):
    """Regenerate routing/requests/validation from raw responses, including failures."""
    indexed = {r['sample_id']: r['input'] for r in inputs}
    jobs = make_jobs(inputs)
    if len(results) != len(jobs):raise ValidationError('incomplete result grid')
    for job, result in zip(jobs, results):
        if {k: result[k] for k in job} != job:raise ValidationError('result identity/order changed')
        class Playback:
            available_adapters = {'research_tool_lora'}
            def __init__(self):self.used = 0; self.mismatches = []
            def generate(self, request):
                if self.used >= len(result['calls']):
                    self.mismatches.append('missing raw call'); raise RuntimeError('missing call')
                call = result['calls'][self.used]; self.used += 1
                if request != call['request'] or canonical_hash(request) != call['request_sha256']:
                    self.mismatches.append('request/adapter differs')
                if call['error_type']:
                    # AgentRunner records only the exception type, not its content.
                    raise type(call['error_type'], (Exception,), {})(call['error'])
                return call['output']
        backend = Playback()
        rebuilt = runner(agents, backend, config).run(input_context(indexed[job['sample_id']]), **ARMS[job['arm']])
        if (backend.mismatches or backend.used != len(result['calls'])
                or without_timing(rebuilt) != without_timing(result['result'])):
            raise ValidationError('raw output replay/request integrity failed')


def score(root, inputs, membership, results, config):
    """Post-generation scoring; missing predictions never disappear from coverage."""
    jobs = make_jobs(inputs)
    if len(results) != len(jobs) or any({k: r[k] for k in j} != j for j, r in zip(jobs, results)):
        raise ValidationError('incomplete or duplicate result grid')
    labels = rows(root/'partitions/validation.labels.jsonl')
    label_index = {r['sample_id']: r['label'] for r in labels}
    if len(labels) != len(label_index) or set(label_index) != {r['sample_id'] for r in inputs}:
        raise ValidationError('labels differ from evaluation identities')
    for row in inputs:
        m = membership[row['sample_id']]
        parse_record({**row['input'], 'sample_id': row['sample_id'], 'event_id': m['event_id'],
                      'event_group_id': m['event_group_id'], 'dataset_source': 'pma_development',
                      'dataset_version': m['dataset_version'], 'label': label_index[row['sample_id']]})
    train = rows(root/'partitions/train.labels.jsonl')
    if (not train or len({r['sample_id'] for r in train}) != len(train)
            or {r['sample_id'] for r in train} != {sid for sid, m in membership.items() if m['split'] == 'train'}):
        raise ValidationError('invalid training baseline identities')
    grouped_train = defaultdict(list)
    for row in train:
        y = row['label']['outcome']
        if type(y) is not int or y not in (0, 1):raise ValidationError('invalid training outcome')
        grouped_train[membership[row['sample_id']]['event_group_id']].append(y)
    base_rate = math.fsum(sum(v)/len(v) for v in grouped_train.values())/len(grouped_train)
    ids = [r['sample_id'] for r in inputs]
    outcomes = [label_index[sid]['outcome'] for sid in ids]
    groups = [membership[sid]['event_group_id'] for sid in ids]
    market = [r['input']['market']['probability'] if r['input']['market'] else None for r in inputs]
    result_index = {(r['arm'], r['sample_id']): r for r in results}
    probabilities = {arm: [result_index[arm, sid]['result']['prediction']['probability']
                          if result_index[arm, sid]['result']['status'] == 'completed' else None for sid in ids] for arm in ARMS}
    def scores(p, indices):
        s = score_predictions([p[i] for i in indices], [outcomes[i] for i in indices],
                              ece_bins=config['ece_bins'], log_loss_epsilon=config['log_loss_epsilon'])
        by_group = defaultdict(list)
        for i in indices:
            if p[i] is not None:by_group[groups[i]].append(i)
        event_scores = {g: score_predictions([p[i] for i in ix], [outcomes[i] for i in ix],
                       ece_bins=config['ece_bins'], log_loss_epsilon=config['log_loss_epsilon']) for g, ix in by_group.items()}
        s.update(covered_event_groups=len(by_group), event_group_count=len({groups[i] for i in indices}),
                 event_mean={k: math.fsum(v[k] for v in event_scores.values())/len(event_scores) if event_scores else None
                             for k in ('brier', 'log_loss')}, per_event=event_scores)
        return s
    all_indices = list(range(len(ids)))
    baselines = {'market': market, 'constant_0_5': [.5]*len(ids), 'training_event_base_rate': [base_rate]*len(ids)}
    report = {'observation_count': len(ids), 'event_group_count': len(set(groups)), 'training_base_rate': base_rate,
              'baselines': {a: scores(p, all_indices) for a, p in baselines.items()}, 'arms': {}}
    for arm, p in probabilities.items():
        rr = [result_index[arm, sid] for sid in ids]; calls = [c for r in rr for c in r['calls']]
        matched = [i for i in all_indices if p[i] is not None and market[i] is not None]
        usage = [c['usage'] for c in calls if c['usage'] is not None]
        seconds = math.fsum(u['seconds'] for u in usage)
        tokens = sum(u['output_tokens'] for u in usage)
        roles = {role: [t for r in rr for t in r['result']['trace'] if t['agent'] == role]
                 for role in ('research', 'forecast')}
        report['arms'][arm] = {'scores': scores(p, all_indices),
            'market_matched': {'model': scores(p, matched), 'market': scores(market, matched)},
            'failures': dict(Counter(r['result'].get('stage', '')+':'+r['result'].get('error', '') for r in rr if r['result']['status'] != 'completed')),
            'validation_errors': dict(Counter(t.get('validation_error', t.get('error_type', '')) for r in rr for t in r['result']['trace'] if t['status'] != 'valid')),
            'role_validity': {role: {'calls': len(ts), 'valid': sum(t['status'] == 'valid' for t in ts)} for role, ts in roles.items()},
            'resources': {'model_calls': len(calls), 'input_tokens': sum(u['input_tokens'] for u in usage),
                'output_tokens': tokens, 'generation_seconds': seconds, 'output_tokens_per_second': tokens/seconds if seconds else None,
                'token_limit_calls': sum(u['output_reached_token_limit'] for u in usage),
                'mean_workflow_seconds': math.fsum(r['seconds'] for r in rr)/len(rr),
                'peak_allocated_bytes': max(r['memory']['peak_allocated_bytes'] for r in rr),
                'peak_reserved_bytes': max(r['memory']['peak_reserved_bytes'] for r in rr)}}
    common = [i for i in all_indices if all(p[i] is not None for p in probabilities.values()) and market[i] is not None]
    report['common_coverage'] = {'sample_ids': [ids[i] for i in common],
                                'scores': {a: scores(p, common) for a, p in {**probabilities, **baselines}.items()}}
    return report


def evaluate(dataset, config_path, agent_config, training_run, bundle, model_manifest, output):
    if output.exists():raise ValidationError('output already exists')
    config = load_config(config_path)
    manifest, inputs, membership = read_inputs(dataset, config)
    agents, _ = load_capabilities(agent_config)
    trained, tc = verify_training_run(training_run, bundle, agents)
    model_path, model_hash = verify_model_manifest(model_manifest, tc)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValidationError('BF16 CUDA required')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    preflight = []
    for row in inputs:
        for role in ('research', 'forecast'):
            prompt = render_agent_prompt({'input': row['input'], 'upstream': {}, 'instruction': agent_instruction(role, config['output_protocol'])})
            n = len(tokenizer.encode(prompt, add_special_tokens=False))
            # Reserve another completion-sized budget for research upstream output.
            if n + 2*config['max_new_tokens'] > config['max_context_tokens']:
                raise ValidationError('insufficient context; no truncation allowed')
            preflight.append({'sample_id': row['sample_id'], 'role': role, 'initial_input_tokens': n})
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent, output/'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(config_path, output/'config.json'); shutil.copyfile(agent_config, output/'agent_config.json')
    shutil.copyfile(dataset/'report.json', output/'dataset_report.json')
    for name in DATA_FILES:
        dest = output/'data'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(dataset/name, dest)
    shutil.copyfile(dataset/'report.json', output/'data/report.json')
    (output/'token_preflight.json').write_text(json_text(preflight))
    jobs = make_jobs(inputs); (output/'jobs.jsonl').write_text(jsonl(jobs))
    plan = {'kind': 'real_market_validation_diagnostic', 'frozen_at': datetime.now(timezone.utc).isoformat(),
            'config': config, 'dataset_version': manifest['dataset_version'], 'code': code_provenance(),
            'base_model': tc['model'], 'base_revision': tc['model_revision'], 'model_manifest_sha256': model_hash,
            'training_report_sha256': sha256_file(training_run/'report.json'), 'adapter_hashes': trained['adapter_hashes'],
            'forecast_adapter': None, 'training': False, 'default_promoted': False,
            'artifact_hashes': {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()},
            'limitations': ['Development validation only; final test is not opened or scored.',
                'Historical outcomes may be present in foundation pretraining; no prospective skill claim.',
                'Observation snapshots/contracts within the same macro release are correlated.',
                'Market prices are supplied to every arm; success alone does not establish incremental information.',
                'Research LoRA only; forecast uses Base in every arm. No forecasting LoRA has been trained.',
                'Equal per-request token budget; multi-agent workflows have a larger total budget.',
                'Schema validity does not establish semantic correctness or grounded reasoning.'] + manifest.get('limitations', [])}
    (output/'plan.json').write_text(json_text(plan))
    report = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha256_file(output/'plan.json'),
              'planned_jobs': len(jobs), 'completed_jobs': 0, 'base_model_loads': 0, 'default_promoted': False,
              'packages': {p: version(p) for p in ('torch', 'transformers', 'peft')}, 'python_executable': sys.executable,
              'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}
    def save():
        temp = output/'report.tmp'; temp.write_text(json_text(report)); temp.replace(output/'report.json')
    save(); results = []
    try:
        torch.cuda.set_device(0); set_seed(config['seed']); torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                   dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', use_safetensors=True)
        report['base_model_loads'] = 1
        model = PeftModel.from_pretrained(base, training_run/'adapter', adapter_name='research_tool_lora', is_trainable=False, local_files_only=True)
        model.gradient_checkpointing_disable(); model.config.use_cache = True
        runtime = SharedPeftExecutor(model)
        class RecordingBackend(PeftTextBackend):
            def __init__(self):
                super().__init__(runtime, tokenizer, max_context_tokens=config['max_context_tokens'], max_new_tokens=config['max_new_tokens'])
                self.calls = []
            def generate(self, request):
                c = {'request': deepcopy(request), 'request_sha256': canonical_hash(request), 'output': None,
                     'usage': None, 'error': None, 'error_type': None}
                try:
                    response = super().generate(request); c['output'] = response; return response
                except Exception as exc:
                    c.update(error=str(exc), error_type=type(exc).__name__); raise
                finally:
                    c['usage'] = self.last_usage; self.calls.append(c)
        backend = RecordingBackend(); run = runner(agents, backend, config)
        by_id = {row['sample_id']: row['input'] for row in inputs}
        report['generation_started_at'] = datetime.now(timezone.utc).isoformat(); save()
        with (output/'results.jsonl').open('x') as log:
            for job in jobs:
                backend.calls = []; torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
                result = run.run(input_context(by_id[job['sample_id']]), **ARMS[job['arm']])
                torch.cuda.synchronize()
                record = {**job, 'result': result, 'calls': backend.calls, 'seconds': time.perf_counter()-start,
                          'memory': {'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}}
                log.write(jsonl([record])); log.flush(); results.append(record)
                report['completed_jobs'] = len(results); save()
                print(json_text({'progress': f'{len(results)}/{len(jobs)}', 'arm': job['arm'], 'status': result['status'], 'seconds': round(record['seconds'], 2)}).strip(), flush=True)
                if any(c['error_type'] == 'OutOfMemoryError' for c in backend.calls):raise RuntimeError('CUDA OOM; partial results only')
        replay(inputs, results, agents, config)
        report['metrics'] = score(output/'data', inputs, membership, results, config)
        report['all_parameters_frozen'] = all(not p.requires_grad for p in model.parameters())
        report['results_sha256'] = sha256_file(output/'results.jsonl'); report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat(); save()
    return report


def audit(run):
    report = strict_json((run/'report.json').read_text()); plan = strict_json((run/'plan.json').read_text())
    if (report['status'] != 'completed' or report['plan_sha256'] != sha256_file(run/'plan.json')
            or report['results_sha256'] != sha256_file(run/'results.jsonl')):raise ValidationError('incomplete or changed run')
    for name, digest in plan['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(run/name) != digest:
            raise ValidationError('frozen artifact changed')
    config = load_config(run/'config.json'); agents, _ = load_capabilities(run/'agent_config.json')
    _, inputs, membership = read_inputs(run/'data', config)
    results = rows(run/'results.jsonl')
    if rows(run/'jobs.jsonl') != make_jobs(inputs) or config != plan['config']:raise ValidationError('frozen jobs/config changed')
    replay(inputs, results, agents, config)
    if score(run/'data', inputs, membership, results, config) != report['metrics']:raise ValidationError('metrics mismatch')
    if (not plan['frozen_at'] < report['started_at'] < report['generation_started_at'] < report['finished_at']
            or report['completed_jobs'] != report['planned_jobs'] or report['completed_jobs'] != len(results)
            or report['base_model_loads'] != 1 or not report['all_parameters_frozen']):raise ValidationError('execution invariants failed')
    return {'status': 'passed', 'jobs': len(results), 'report_sha256': sha256_file(run/'report.json'),
            'raw_outputs_replayed': True, 'routing_verified': True, 'metrics_exactly_reproduced': True,
            'final_test_scored': False, 'training_started': False, 'default_promoted': False}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    for name in ('dataset', 'config', 'agent-config', 'training-run', 'bundle', 'model-manifest', 'output'):
        r.add_argument('--'+name, type=Path, required=True)
    a = sub.add_parser('audit'); a.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'audit':print(json_text(audit(args.run)))
    else:
        kw = vars(args); kw.pop('command'); kw['config_path'] = kw.pop('config')
        result = evaluate(**kw); print(json_text({'status': result['status'], 'jobs': result['completed_jobs']}))


if __name__ == '__main__':main()
