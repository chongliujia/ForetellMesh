"""Frozen, causal Qwen multi-agent signals for allocation research (train only)."""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import math
from pathlib import Path
import shutil
import time

from .agent_baseline_data import input_context
from .capabilities import load_capabilities
from .data import sha256_file, strict_json
from .evaluation import json_text
from .lora_probe import verify_model_manifest
from .market_development import rows
from .market_experts import build_features, enrich
from .offline_base import make_runner, replay as replay_responses
from .paper_trading import TradePrintFeed
from .peft_runtime import PeftTextBackend, SharedPeftExecutor
from .schema import ValidationError, iso, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .trading_rl import artifacts
from .trading_rl_data import prepare


def read_spec(path):
    c = strict_json(path.read_text())
    expected = {'partition': 'train', 'model': 'Qwen/Qwen3-8B-Base',
        'model_revision': '49e3418fbbbca6ecbdf9608b4d22e5a407081db4',
        'workflow': 'market_quant_game_forecast', 'engine': 'langgraph', 'mode': 'base',
        'output_protocol': 'grounded_json_v3', 'response_transport': 'single_json_fence',
        'max_context_tokens': 8192, 'max_new_tokens': 768, 'max_repairs': 1,
        'stop_on_json_object': True, 'do_sample': False, 'base_dtype': 'bfloat16',
        'attention': 'sdpa', 'refresh_seconds': 604800, 'signal_ttl_seconds': 604800,
        'markets_per_group': 2, 'selection': 'highest_past_24h_trade_count_then_market_id',
        'training': False, 'load_adapters': False, 'final_test_opened': False,
        'real_orders_sent': 0}
    if (set(c) != set(expected) | {'run_name', 'seed', 'reference_report_sha256'}
            or any(type(c[k]) is not type(v) or c[k] != v for k, v in expected.items())
            or type(c['seed']) is not int or not 0 <= c['seed'] < 2**32):
        raise ValidationError('unsupported agent signal protocol')
    return c


def schedule(data, config, spec):
    """Causal sampling, independent of outcomes, future trades, or learned actions."""
    windows = {m.market_id: m for m in data.windows}
    last, jobs = {}, []
    p = config['environment']
    for t, state in zip(data.ticks, data.states):
        groups = defaultdict(list)
        for mid, s in state.items():
            if (s.get('status') == 'ready' and float(p['min_yes_price']) <= s['yes_price'] <= float(p['max_yes_price'])
                    and t + timedelta(days=1) < min(windows[mid].retire_at, data.end)):
                groups[windows[mid].event_group_id].append((-s['trade_count_24h'], mid))
        for group, candidates in sorted(groups.items()):
            if group in last and (t - last[group]).total_seconds() < spec['refresh_seconds']:
                continue
            last[group] = t
            for _, mid in sorted(candidates)[:spec['markets_per_group']]:
                jobs.append({'sample_id': f'agent-allocation:{mid}:{iso(t)}', 'market_id': mid,
                             'event_group_id': group, 'observation_time': iso(t)})
    return jobs


def historical_input(job, evidence_rows, quote):
    """Reconstruct only an immutable question and explicitly dated evidence <= T."""
    t = timestamp(job['observation_time'], 'signal cutoff')
    questions = {r['input']['question'] for r in evidence_rows}
    if len(questions) != 1:
        raise ValidationError('historical question is not immutable')
    evidence = {}
    for row in evidence_rows:
        for e in row['input']['evidence']:
            if max(timestamp(e[k], k) for k in ('published_at', 'available_at')) > t:
                continue
            key = e['evidence_id']
            if key in evidence and evidence[key] != e:
                raise ValidationError('conflicting historical evidence identity')
            evidence[key] = deepcopy(e)
    q, qt = quote
    item = {'sample_id': job['sample_id'], 'input': {'question': next(iter(questions)),
        'observation_time': iso(t), 'evidence': [evidence[k] for k in sorted(evidence)],
        'market': {'probability': float(q), 'observed_at': iso(qt), 'available_at': iso(qt)}}}
    input_context(item['input'])
    return item


def prepare_inputs(dataset, store, reference, spec):
    if sha256_file(reference / 'report.json') != spec['reference_report_sha256']:
        raise ValidationError('allocation reference changed')
    prior = strict_json((reference / 'report.json').read_text())
    audit = strict_json((reference / 'audit.json').read_text())
    if audit['status'] != 'passed' or audit['report_sha256'] != spec['reference_report_sha256']:
        raise ValidationError('audited reference required')
    if sha256_file(reference / 'plan.json') != prior['plan_sha256']:
        raise ValidationError('reference plan changed')
    config = strict_json((reference / 'plan.json').read_text())['original_config']
    data = prepare(dataset, store, 'train', config)
    grouped = defaultdict(list)
    for row in rows(dataset / 'partitions/train.inputs.jsonl'):
        grouped[data.sample_markets[row['sample_id']]].append(row)
    inputs, proofs = [], []
    jobs = schedule(data, config, spec)
    feed = TradePrintFeed(store / 'prices.sqlite')
    try:
        for job in jobs:
            mid = job['market_id']; t = timestamp(job['observation_time'], 'cutoff')
            started = time.perf_counter()
            feature, proof = build_features(feed, mid, t)
            item = historical_input(job, grouped[mid], feed.latest(mid, t))
            item = enrich(item, feature); input_context(item['input'])
            inputs.append(item)
            proofs.append({**proof, 'sample_id': job['sample_id'], 'seconds': time.perf_counter() - started})
    finally:
        feed.close()
    return data, config, jobs, inputs, proofs


def generate(dataset, store, reference, spec_path, agent_config, model_manifest, output):
    if output.exists():
        raise ValidationError('signal output already exists')
    c = read_spec(spec_path)
    agents, _ = load_capabilities(agent_config)
    data, config, jobs, inputs, proofs = prepare_inputs(dataset, store, reference, c)
    model_path, model_hash = verify_model_manifest(model_manifest, c)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValidationError('BF16 CUDA required')
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent, output / 'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    for source, name in ((spec_path, 'config.json'), (agent_config, 'agent_config.json'), (model_manifest, 'model_manifest.json')):
        shutil.copyfile(source, output / name)
    (output / 'inputs.jsonl').write_text(jsonl(inputs))
    (output / 'feature_audits.jsonl').write_text(jsonl(proofs))
    plan = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'spec': c, 'original_config': config,
        'jobs': jobs, 'data_bindings': data.bindings, 'model_manifest_sha256': model_hash,
        'paths': {k: str(p.resolve()) for k, p in dict(dataset=dataset, store=store, reference=reference).items()},
        'artifact_hashes': artifacts(output)}
    (output / 'plan.json').write_text(json_text(plan))
    report = {'status': 'running', 'plan_sha256': sha256_file(output / 'plan.json'),
        'planned_jobs': len(jobs), 'completed_jobs': 0, 'successful_jobs': 0, 'model_calls': 0,
        'packages': {p: version(p) for p in ('torch', 'transformers', 'langgraph')},
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'base_model_loads': 0,
        'training': False, 'loaded_adapters': [], 'validation_opened': False, 'final_test_opened': False,
        'real_orders_sent': 0, 'started_at': datetime.now(timezone.utc).isoformat()}
    def save():
        tmp = output / 'generation_report.tmp'; tmp.write_text(json_text(report)); tmp.replace(output / 'generation_report.json')
    save()
    try:
        set_seed(c['seed']); torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', use_safetensors=True)
        model.gradient_checkpointing_disable(); model.config.use_cache = True
        executor = SharedPeftExecutor(model)
        if executor.available_adapters:
            raise ValidationError('only frozen Base permitted')
        class Recording(PeftTextBackend):
            def generate(self, request):
                call = {'request': deepcopy(request), 'request_sha256': canonical_hash(request),
                        'output': None, 'error': None, 'error_type': None, 'usage': None}
                try:
                    call['output'] = super().generate(request); return call['output']
                except Exception as exc:
                    call.update(error=str(exc), error_type=type(exc).__name__); raise
                finally:
                    call['usage'] = self.last_usage; self.calls.append(call)
        backend = Recording(executor, tokenizer, max_context_tokens=c['max_context_tokens'],
                            max_new_tokens=c['max_new_tokens'], stop_on_json_object=True)
        runner = make_runner(agents, backend, c)
        report['base_model_loads'] = 1; save()
        with (output / 'results.jsonl').open('x') as log:
            for job, row, proof in zip(jobs, inputs, proofs):
                backend.calls = []; torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                started = time.perf_counter()
                value = runner.run(input_context(row['input']), workflow=c['workflow'], mode='base')
                torch.cuda.synchronize(); elapsed = time.perf_counter() - started
                r = {**job, 'input_sha256': canonical_hash(row['input']), 'result': value,
                    'calls': backend.calls, 'seconds': elapsed, 'feature_seconds': proof['seconds'],
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                log.write(jsonl([r])); log.flush()
                report['completed_jobs'] += 1; report['successful_jobs'] += value['status'] == 'completed'
                report['model_calls'] += len(backend.calls); save()
                print(f"{report['completed_jobs']}/{len(jobs)} {job['market_id']} {value['status']} {elapsed:.1f}s", flush=True)
                if any(x['error_type'] == 'OutOfMemoryError' for x in backend.calls):
                    raise RuntimeError('CUDA OOM')
        replay_responses(inputs, rows(output / 'results.jsonl'), agents, c)
        report.update(status='completed', all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
                      results_sha256=sha256_file(output / 'results.jsonl'))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat(); save()
    return report


def verify(root, *, rebuild_inputs=True):
    c = read_spec(root / 'config.json'); plan = strict_json((root / 'plan.json').read_text())
    report = strict_json((root / 'generation_report.json').read_text())
    if (report['status'] != 'completed' or report['plan_sha256'] != sha256_file(root / 'plan.json')
            or report['results_sha256'] != sha256_file(root / 'results.jsonl')
            or plan['spec'] != c or report['base_model_loads'] != 1 or not report['all_parameters_frozen']):
        raise ValidationError('agent generation not frozen')
    for name, digest in plan['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root / name) != digest:
            raise ValidationError('signal artifact changed: ' + name)
    inputs, results = rows(root / 'inputs.jsonl'), rows(root / 'results.jsonl')
    if len(results) != report['planned_jobs'] or report['completed_jobs'] != len(results):
        raise ValidationError('incomplete signal population')
    if [{k: r[k] for k in j} for j, r in zip(plan['jobs'], results)] != plan['jobs']:
        raise ValidationError('signal job metadata changed')
    if rebuild_inputs:
        data, config, jobs, rebuilt, proofs = prepare_inputs(**{k: Path(v) for k, v in plan['paths'].items()}, spec=c)
        if (inputs != rebuilt or jobs != plan['jobs'] or data.bindings != plan['data_bindings']
                or config != plan['original_config'] or [{k: v for k, v in p.items() if k != 'seconds'} for p in proofs]
                != [{k: v for k, v in p.items() if k != 'seconds'} for p in rows(root / 'feature_audits.jsonl')]):
            raise ValidationError('historical signal inputs do not reproduce')
    agents, _ = load_capabilities(root / 'agent_config.json')
    replay_responses(inputs, results, agents, c)
    return c, plan, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('generate')
    for name in ('dataset', 'store', 'reference', 'config', 'agent-config', 'model-manifest', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    sub.add_parser('audit').add_argument('--run', type=Path, required=True)
    a = parser.parse_args()
    if a.command == 'generate':
        print(json_text(generate(a.dataset, a.store, a.reference, a.config, a.agent_config, a.model_manifest, a.output)))
    else:
        _, _, results = verify(a.run)
        audit = {'status': 'passed', 'jobs_reproduced': len(results), 'raw_calls_replayed': sum(len(r['calls']) for r in results),
                 'generation_report_sha256': sha256_file(a.run / 'generation_report.json')}
        (a.run / 'audit.json').write_text(json_text(audit)); print(json_text(audit))


if __name__ == '__main__':
    main()
