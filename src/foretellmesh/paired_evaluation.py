"""Frozen three-checkpoint factorial diagnostic and CPU-only raw-output replay."""
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
from .agent_runtime import agent_instruction, decode_agent_response, validate_agent_output
from .capabilities import load_capabilities
from .capability_evaluation import PREVIOUS_ADAPTER, verify_training_run
from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .lora_probe import verify_model_manifest
from .paired_diagnostic import read_paired_cohort, summarize_paired
from .peft_runtime import PeftTextBackend, SharedPeftExecutor, render_agent_prompt
from .schema import ValidationError, fields
from .sft_data import jsonl
from .synthetic_sft import canonical_hash

ADAPTERS = {'base': None, 'previous': PREVIOUS_ADAPTER, 'candidate': 'research_tool_lora'}


def load_config(path: Path) -> dict:
    fixed = {'schema_version': '1', 'partition': 'validation', 'arms': list(ADAPTERS),
             'output_protocol': 'grounded_json_v3', 'response_transport': 'single_json_fence',
             'max_context_tokens': 2048, 'max_new_tokens': 384, 'do_sample': False,
             'attention': 'sdpa', 'base_dtype': 'bfloat16', 'default_promotion': False,
             'retries': 0, 'training': False}
    c = fields(strict_json(path.read_text()), set(fixed) | {'run_name', 'seed'}, 'paired evaluation config')
    if (any(type(c[k]) is not type(v) or c[k] != v for k, v in fixed.items())
            or type(c['seed']) is not int or not 0 <= c['seed'] < 2**32
            or not isinstance(c['run_name'], str) or not c['run_name'].strip()):
        raise ValidationError('unsupported paired evaluation config')
    return c


def make_jobs(inputs: list[dict], config: dict) -> list[dict]:
    jobs = []
    for index, row in enumerate(inputs):
        offset = index % len(ADAPTERS)
        for arm in config['arms'][offset:] + config['arms'][:offset]:
            request = {'agent': row['role'], 'adapter': ADAPTERS[arm], 'upstream': {},
                       'input': deepcopy(row['input']), 'instruction': agent_instruction(row['role'], config['output_protocol'])}
            jobs.append({'sample_id': row['sample_id'], 'arm': arm, 'request': request,
                         'request_sha256': canonical_hash(request)})
    return jobs


def decode(raw: str, request: dict, config: dict, max_chars: int) -> dict:
    try:
        if len(raw) > max_chars:raise ValidationError('output budget exceeded')
        value, transport = decode_agent_response(raw, config['response_transport'])
        output = validate_agent_output(request['agent'], value, input_context(request['input']))
        return {'output': output, 'decoded_transport': transport, 'error': None}
    except (ValueError, TypeError) as exc:
        return {'output': None, 'decoded_transport': None, 'error': f'{type(exc).__name__}: {exc}'}


def summarize(inputs: list[dict], judges: list[dict], results: list[dict]) -> dict:
    report = summarize_paired(inputs, judges, results)
    for arm, stats in report['arms'].items():
        rows = [r for r in results if r['arm'] == arm]
        seconds = math.fsum(r['usage']['seconds'] for r in rows)
        tokens = sum(r['usage']['output_tokens'] for r in rows)
        stats['resources'] = {'model_calls': len(rows), 'generation_seconds': seconds,
            'mean_seconds': math.fsum(r['seconds'] for r in rows)/len(rows),
            'input_tokens': sum(r['usage']['input_tokens'] for r in rows), 'output_tokens': tokens,
            'output_tokens_per_generation_second': tokens/seconds if seconds else None,
            'output_token_limit_calls': sum(r['usage']['output_reached_token_limit'] for r in rows),
            'peak_allocated_bytes': max(r['memory']['peak_allocated_bytes'] for r in rows)}
    return report


def verify_excluded_source(cohort_report: dict, cohort: Path, bundle: Path) -> None:
    """The paired scenario source must be the diagnostic excluded by v3 data."""
    manifest = strict_json((bundle/'manifest.json').read_text())
    if (manifest.get('excluded_diagnostic_manifest_sha256') != cohort_report['source_manifest_sha256']
            or strict_json((bundle/'excluded_diagnostic_config.json').read_text()) != strict_json((cohort/'source/config.json').read_text())):
        raise ValidationError('paired source is not the candidate data exclusion cohort')


def evaluate(cohort: Path, config_path: Path, agent_config: Path, training_run: Path, bundle: Path,
             previous_training_run: Path, previous_bundle: Path, model_manifest: Path, output: Path) -> dict:
    if output.exists():raise ValidationError('paired evaluation output already exists')
    config = load_config(config_path)
    cohort_report, inputs, judges = read_paired_cohort(cohort)
    agents, agent_hash = load_capabilities(agent_config)
    trained, tc = verify_training_run(training_run, bundle, agents)
    previous, _ = verify_training_run(previous_training_run, previous_bundle, agents)
    verify_excluded_source(cohort_report, cohort, bundle)
    model_path, model_hash = verify_model_manifest(model_manifest, tc)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValidationError('CUDA BF16 required')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    jobs = make_jobs(inputs, config)
    preflight = []
    for job in jobs:
        prompt = render_agent_prompt(job['request'])
        n = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n + config['max_new_tokens'] > config['max_context_tokens']:raise ValidationError('prompt budget exceeded')
        preflight.append({'sample_id': job['sample_id'], 'arm': job['arm'], 'input_tokens': n,
                          'prompt_sha256': sha256_bytes(prompt.encode())})
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent, output/'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    (output/'config.json').write_bytes(config_path.read_bytes())
    (output/'agent_config.json').write_bytes(agent_config.read_bytes())
    (output/'jobs.jsonl').write_text(jsonl(jobs))
    (output/'token_preflight.json').write_text(json_text(preflight))
    frozen = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'planned_jobs': len(jobs),
        'config_sha256': sha256_file(config_path), 'agent_config_sha256': agent_hash,
        'cohort_path': str(cohort.resolve()), 'cohort_manifest_sha256': sha256_file(cohort/'manifest.json'),
        'jobs_sha256': sha256_file(output/'jobs.jsonl'), 'token_preflight_sha256': sha256_file(output/'token_preflight.json'),
        'model_manifest_sha256': model_hash, 'base_model': tc['model'], 'base_revision': tc['model_revision'],
        'code': code_provenance(), 'adapter_names': ADAPTERS,
        'checkpoints': {arm: {'training_run': str(run.resolve()), 'bundle': str(data.resolve()),
            'training_report_sha256': sha256_file(run/'report.json'), 'adapter_hashes': r['adapter_hashes']}
            for arm, run, data, r in (('candidate', training_run, bundle, trained),
                                     ('previous', previous_training_run, previous_bundle, previous))},
        'primary_metric': 'exact_unknown_set_with_schema_validity',
        'secondary_metric': 'exact_unknown_set_plus_evidence_or_empty_risk_contract',
        'contrasts': 'all matched single-factor edges, improved and regressed counts; no row-wise significance claim',
        'default_promoted': False, 'rl_started': False}
    (output/'plan.json').write_text(json_text(frozen))
    report = {'schema_version': '1', 'kind': 'paired_diagnostic_evaluation', 'status': 'running',
        'plan_sha256': sha256_file(output/'plan.json'), 'started_at': datetime.now(timezone.utc).isoformat(),
        'planned_jobs': len(jobs), 'completed_jobs': 0, 'base_model_loads': 0,
        'python_executable': sys.executable, 'packages': {p: version(p) for p in ('torch', 'transformers', 'peft')},
        'gpu': torch.cuda.get_device_name(0), 'cuda_version': torch.version.cuda,
        'limitations': cohort_report['limitations'], 'default_promoted': False, 'rl_started': False}
    def save():
        temp = output/'report.tmp'; temp.write_text(json_text(report)); temp.replace(output/'report.json')
    results = []; save()
    try:
        torch.cuda.set_device(0); set_seed(config['seed']); torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', use_safetensors=True)
        report['base_model_loads'] = 1
        model = PeftModel.from_pretrained(base, training_run/'adapter', adapter_name=ADAPTERS['candidate'],
                                         is_trainable=False, local_files_only=True)
        model.load_adapter(previous_training_run/'adapter', adapter_name=ADAPTERS['previous'],
                           is_trainable=False, local_files_only=True)
        model.gradient_checkpointing_disable(); model.config.use_cache = True
        report['generation_config'] = model.generation_config.to_dict()
        backend = PeftTextBackend(SharedPeftExecutor(model), tokenizer,
            max_context_tokens=config['max_context_tokens'], max_new_tokens=config['max_new_tokens'])
        report['generation_started_at'] = datetime.now(timezone.utc).isoformat(); save()
        with (output/'results.jsonl').open('x') as log:
            for index, job in enumerate(jobs):
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
                raw = backend.generate(job['request'])
                record = {**job, 'raw_output': raw, 'usage': backend.last_usage,
                          **decode(raw, job['request'], config, agents['limits']['max_output_chars'])}
                torch.cuda.synchronize()
                record.update(seconds=time.perf_counter() - started, memory={
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
                log.write(jsonl([record])); log.flush(); results.append(record)
                report['completed_jobs'] = len(results); save()
                print(json_text({'progress': f'{index + 1}/{len(jobs)}', 'arm': job['arm'],
                    'sample_id': job['sample_id'], 'schema_valid': record['output'] is not None}).strip(), flush=True)
        report['metrics'] = summarize(inputs, judges, results)
        report['results_sha256'] = sha256_file(output/'results.jsonl')
        report['all_parameters_frozen'] = all(not p.requires_grad for p in model.parameters())
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'failed'; report['error'] = f'{type(exc).__name__}: {exc}'; raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat(); save()
    return report


def audit(run: Path) -> dict:
    report = strict_json((run/'report.json').read_text()); plan = strict_json((run/'plan.json').read_text())
    if report['status'] != 'completed':raise ValidationError('cannot audit an incomplete evaluation')
    if sha256_file(run/'plan.json') != report['plan_sha256']:raise ValidationError('plan changed')
    config = load_config(run/'config.json'); agents, agent_hash = load_capabilities(run/'agent_config.json')
    cohort = Path(plan['cohort_path']); manifest, inputs, judges = read_paired_cohort(cohort)
    jobs = make_jobs(inputs, config)
    for name, key in (('config.json', 'config_sha256'), ('jobs.jsonl', 'jobs_sha256'), ('token_preflight.json', 'token_preflight_sha256')):
        if sha256_file(run/name) != plan[key]:raise ValidationError('frozen artifact hash mismatch')
    if agent_hash != plan['agent_config_sha256'] or sha256_file(cohort/'manifest.json') != plan['cohort_manifest_sha256']:
        raise ValidationError('agent or cohort changed')
    chunks = []
    for file in sorted((run/'source_snapshot/foretellmesh').glob('*.py')):
        chunks.extend((file.name.encode(), b'\0', file.read_bytes(), b'\0'))
    if sha256_bytes(b''.join(chunks)) != plan['code']['source_sha256'] or code_provenance()['source_sha256'] != plan['code']['source_sha256']:
        raise ValidationError('evaluation source mismatch; run audit with the archived source revision')
    if [strict_json(s) for s in (run/'jobs.jsonl').read_text().splitlines()] != jobs:raise ValidationError('frozen jobs changed')
    for entry in plan['checkpoints'].values():
        training = Path(entry['training_run'])
        r, _ = verify_training_run(training, Path(entry['bundle']), agents)
        if sha256_file(training/'report.json') != entry['training_report_sha256'] or r['adapter_hashes'] != entry['adapter_hashes']:
            raise ValidationError('checkpoint provenance changed')
    verify_excluded_source(manifest, cohort, Path(plan['checkpoints']['candidate']['bundle']))
    records = [strict_json(s) for s in (run/'results.jsonl').read_text().splitlines()]
    if (len(records) != len(jobs) or len(records) != report['completed_jobs'] or len(records) != plan['planned_jobs']
            or sha256_file(run/'results.jsonl') != report['results_sha256']):
        raise ValidationError('result file changed or incomplete')
    tokens = strict_json((run/'token_preflight.json').read_text())
    if len(tokens) != len(jobs):raise ValidationError('preflight count mismatch')
    for job, record, token in zip(jobs, records, tokens):
        if {k: record[k] for k in job} != job:raise ValidationError('request or adapter changed')
        decoded = decode(record['raw_output'], job['request'], config, agents['limits']['max_output_chars'])
        if any(decoded[k] != record[k] for k in decoded):raise ValidationError('raw decoding mismatch')
        if (token['sample_id'] != job['sample_id'] or token['arm'] != job['arm']
                or token['prompt_sha256'] != sha256_bytes(render_agent_prompt(job['request']).encode())
                or token['input_tokens'] != record['usage']['input_tokens']
                or token['input_tokens'] + config['max_new_tokens'] > config['max_context_tokens']):
            raise ValidationError('prompt preflight mismatch')
    metrics = summarize(inputs, judges, records)
    if metrics != report['metrics']:raise ValidationError('metrics changed')
    if not (manifest['frozen_at'] < plan['frozen_at'] < report['started_at'] < report['generation_started_at'] < report['finished_at']):
        raise ValidationError('plan was not frozen before generation')
    if report['base_model_loads'] != 1 or not report['all_parameters_frozen'] or plan['adapter_names'] != ADAPTERS:
        raise ValidationError('shared frozen base invariant failed')
    return {'status': 'passed', 'jobs': len(records), 'report_sha256': sha256_file(run/'report.json'),
            'raw_outputs_replayed': True, 'metrics_exactly_reproduced': True, 'fixed_requests_and_adapters_verified': True,
            'source_and_artifact_hashes_verified': True, 'default_promoted': False, 'rl_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    for name in ('cohort', 'config', 'agent-config', 'training-run', 'bundle', 'previous-training-run',
                 'previous-bundle', 'model-manifest', 'output'):
        run.add_argument('--' + name, type=Path, required=True)
    check = sub.add_parser('audit'); check.add_argument('--run', type=Path, required=True)
    a = parser.parse_args()
    if a.command == 'audit':print(json_text(audit(a.run)))
    else:
        arguments = vars(a); arguments.pop('command'); arguments['config_path'] = arguments.pop('config')
        r = evaluate(**arguments)
        print(json_text({'status': r['status'], 'jobs': r['completed_jobs']}))


if __name__ == '__main__':main()
