"""Train-only paired evidence/price pilot; frozen Base, no allocation or tuning."""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
import shutil
import statistics
import time
from urllib.parse import urlparse

from .agent_baseline_data import input_context
from .capabilities import load_capabilities
from .data import sha256_file, strict_json
from .evaluation import json_text
from .lora_probe import verify_model_manifest
from .market_development import rows
from .metrics import score_predictions
from .offline_base import make_runner, replay
from .peft_runtime import PeftTextBackend, SharedPeftExecutor
from .schema import ValidationError, parse_record, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash

ARMS = ('original_market', 'original_blind', 'refreshed_blind')


def family(e: dict) -> str:
    url = urlparse(e['source'])
    if url.hostname == 'www.federalreserve.gov' and '/pressreleases/monetary' in url.path:
        return 'fomc'
    if url.hostname == 'www.bls.gov' and '/archives/cpi_' in url.path:
        return 'cpi'
    if url.hostname == 'www.bls.gov' and '/archives/empsit_' in url.path:
        return 'employment'
    raise ValidationError('evidence is outside the admitted official source families')


def select_samples(inputs: list, membership: list) -> list:
    members = {m['sample_id']: m for m in membership if m['split'] == 'train'}
    if (len({r['sample_id'] for r in inputs}) != len(inputs)
            or set(members) != {r['sample_id'] for r in inputs}
            or len({m['sample_id'] for m in membership}) != len(membership)):
        raise ValidationError('training input/membership identity mismatch')
    groups = defaultdict(list)
    for r in inputs:
        input_context(r['input'])
        groups[members[r['sample_id']]['event_group_id']].append(r)
    selected = []
    for group, samples in sorted(groups.items()):
        contract = min(members[r['sample_id']]['event_id'] for r in samples)
        selected.extend({**deepcopy(r), 'event_group_id': group} for r in samples
                        if members[r['sample_id']]['event_id'] == contract)
    return sorted(selected, key=lambda r: (r['input']['observation_time'], r['sample_id']))


def evidence_pool(inputs: list) -> list:
    pool = {}
    for row in inputs:
        input_context(row['input'])
        for e in row['input']['evidence']:
            family(e)
            if e['evidence_id'] in pool and pool[e['evidence_id']] != e:
                raise ValidationError('conflicting evidence identity')
            pool[e['evidence_id']] = deepcopy(e)
    return [pool[k] for k in sorted(pool)]


def transform(row: dict, pool: list, arm: str, c: dict) -> dict:
    if arm not in ARMS:
        raise ValidationError('unknown evidence arm')
    value = deepcopy(row['input'])
    input_context(value)
    if arm != 'original_market':
        value['market'] = None
    if arm == 'refreshed_blind':
        cutoff = timestamp(value['observation_time'], 'cutoff')
        groups = defaultdict(list)
        for e in pool:
            published = timestamp(e['published_at'], 'publication')
            available = timestamp(e['available_at'], 'availability')
            if max(published, available) <= cutoff and published >= cutoff - timedelta(days=c['evidence_window_days']):
                groups[family(e)].append(e)
        # Preserve original evidence, adding only explicitly dated admitted releases.
        chosen = {e['evidence_id']: e for e in value['evidence']}
        for values in groups.values():
            for e in sorted(values, key=lambda e: (timestamp(e['published_at'], 'publication'), e['evidence_id']),
                            reverse=True)[:c['evidence_per_family']]:
                chosen[e['evidence_id']] = e
        value['evidence'] = [deepcopy(chosen[k]) for k in sorted(chosen)]
    input_context(value)
    return {'sample_id': row['sample_id'] + ':' + arm, 'original_sample_id': row['sample_id'],
            'event_group_id': row['event_group_id'], 'arm': arm, 'input': value}


def prepare(c: dict):
    if (c['partition'] != 'train' or c['training'] or c['load_adapters'] or c['final_test_opened']
            or c['default_promotion'] or c['arms'] != list(ARMS) or c['workflow'] != 'research_forecast'
            or c['selection'] != 'lexicographically_first_contract_per_event_all_observations'
            or c['evidence_window_days'] != 120 or c['evidence_per_family'] != 2):
        raise ValidationError('unsupported evidence pilot protocol')
    root = Path(c['dataset'])
    if sha256_file(root / 'report.json') != c['dataset_report_sha256']:
        raise ValidationError('dataset report changed')
    manifest = strict_json((root / 'report.json').read_text())
    # No labels or held-out input files are opened during preparation/generation.
    for name in ('membership.jsonl', 'partitions/train.inputs.jsonl'):
        if sha256_file(root / name) != manifest['artifact_hashes'][name]:
            raise ValidationError('admitted dataset input changed')
    source = rows(root / 'partitions/train.inputs.jsonl')
    selected = select_samples(source, rows(root / 'membership.jsonl'))
    pool = evidence_pool(source)
    jobs = [transform(r, pool, arm, c) for r in selected for arm in ARMS]
    if c.get('external_evidence_bundle'):
        from .timely_macro_evidence import enrich_jobs
        jobs = enrich_jobs(jobs, selected, c['external_evidence_bundle'])
    return selected, pool, jobs


def cached_controls(c, jobs, agents):
    """Reuse exact matched controls, verifying raw responses without opening labels."""
    if not c.get('cached_control_run'): return {}
    root = Path(c['cached_control_run'])
    if sha256_file(root/'generation_report.json') != c['cached_control_report_sha256']:
        raise ValidationError('cached control report changed')
    report = strict_json((root/'generation_report.json').read_text())
    plan = strict_json((root/'plan.json').read_text())
    if (report['status'] != 'completed' or not report['all_parameters_frozen']
            or report['results_sha256'] != sha256_file(root/'results.jsonl')
            or report['plan_sha256'] != sha256_file(root/'plan.json')
            or report['base_model_loads'] != 1 or report['loaded_adapters']):
        raise ValidationError('unverified cached controls')
    for name, digest in plan['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root/name) != digest:
            raise ValidationError('cached artifact changed')
    cc = strict_json((root/'config.json').read_text())
    for key in ('model', 'model_revision', 'workflow', 'mode', 'engine', 'output_protocol', 'response_transport',
                'max_context_tokens', 'max_new_tokens', 'max_repairs', 'seed', 'base_dtype', 'attention', 'do_sample'):
        if c[key] != cc[key]: raise ValidationError('cached control protocol differs')
    old_agents, _ = load_capabilities(root/'agent_config.json')
    if old_agents != agents: raise ValidationError('cached agents differ')
    old_inputs, old_results = rows(root/'inputs.jsonl'), rows(root/'results.jsonl')
    replay(old_inputs, old_results, old_agents, cc)
    indexed = {r['sample_id']: (j, r) for j, r in zip(old_inputs, old_results)}
    output = {}
    for job in jobs:
        if job['arm'] not in ('original_market', 'original_blind'): continue
        old, result = indexed[job['sample_id']]
        if old != job: raise ValidationError('cached control input differs')
        output[job['sample_id']] = deepcopy(result)
        output[job['sample_id']]['cached_from'] = str(root)
    return output


def score(c, selected, results):
    root = Path(c['dataset'])
    manifest = strict_json((root / 'report.json').read_text())
    name = 'partitions/train.labels.jsonl'
    if sha256_file(root / name) != manifest['artifact_hashes'][name]:
        raise ValidationError('training labels changed')
    labels = {r['sample_id']: r['label'] for r in rows(root / name)}
    members = {r['sample_id']: r for r in rows(root / 'membership.jsonl') if r['split'] == 'train'}
    if set(labels) != set(members):
        raise ValidationError('training label identities changed')
    for r in selected:
        m = members[r['sample_id']]
        parse_record({**r['input'], **{k: v for k, v in m.items() if k != 'split'}, 'dataset_source': 'pma_evidence_pilot',
                      'label': labels[r['sample_id']]})
    indexed = {(r['original_sample_id'], r['arm']): r for r in results}
    expected = {(r['sample_id'], arm) for r in selected for arm in ARMS}
    if set(indexed) != expected or len(indexed) != len(results):
        raise ValidationError('incomplete or duplicate scoring population')
    outcomes = [labels[r['sample_id']]['outcome'] for r in selected]
    groups = [r['event_group_id'] for r in selected]
    probabilities = {arm: [indexed[r['sample_id'], arm]['result']['prediction']['probability']
                          if indexed[r['sample_id'], arm]['result']['status'] == 'completed' else None
                          for r in selected] for arm in ARMS}
    probabilities['market'] = [r['input']['market']['probability'] for r in selected]
    probabilities['constant_0.5'] = [.5] * len(selected)
    common = [i for i in range(len(selected)) if all(p[i] is not None for p in probabilities.values())]

    def summary(p, indices):
        value = score_predictions([p[i] for i in indices], [outcomes[i] for i in indices])
        event_scores = {g: score_predictions([p[i] for i in indices if groups[i] == g],
                                            [outcomes[i] for i in indices if groups[i] == g])
                        for g in sorted(set(groups[i] for i in indices))}
        valid_events = [s['brier'] for s in event_scores.values() if s['brier'] is not None]
        value.update(event_macro_brier=statistics.fmean(valid_events) if valid_events else None,
                     event_count_with_predictions=len(valid_events), by_event=event_scores)
        return value

    return {'scope': 'training_diagnostic_only', 'selected_observations': len(selected),
            'selected_events': len(set(groups)), 'common_observations': len(common),
            'all_eligible': {a: summary(p, list(range(len(selected)))) for a, p in probabilities.items()},
            'common': {a: summary(p, common) for a, p in probabilities.items()},
            'probabilities': probabilities, 'default_promotion': False, 'trading_signals_exported': False}


def run(config: Path, agents_path: Path, manifest_path: Path, output: Path):
    if output.exists():
        raise ValidationError('output already exists')
    c = strict_json(config.read_text())
    selected, pool, jobs = prepare(c)
    agents, _ = load_capabilities(agents_path)
    cached = cached_controls(c, jobs, agents)
    model_path, model_hash = verify_model_manifest(manifest_path, c)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValidationError('BF16 CUDA required')
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent, output / 'source_snapshot/foretellmesh',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for source, name in ((config, 'config.json'), (agents_path, 'agent_config.json'), (manifest_path, 'model_manifest.json')):
        shutil.copyfile(source, output / name)
    for name, values in (('selected', selected), ('evidence_pool', pool), ('inputs', jobs)):
        (output / (name + '.jsonl')).write_text(jsonl(values))
    plan = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'config': c, 'model_sha256': model_hash,
            'artifact_hashes': {str(p.relative_to(output)): sha256_file(p) for p in output.rglob('*') if p.is_file()}}
    (output / 'plan.json').write_text(json_text(plan))
    report = {'status': 'running', 'plan_sha256': sha256_file(output / 'plan.json'),
              'planned_jobs': len(jobs), 'completed_jobs': 0, 'training': False, 'final_test_opened': False,
              'validation_opened': False, 'base_model_loads': 0, 'loaded_adapters': [],
              'packages': {p: version(p) for p in ('torch', 'transformers', 'langgraph')},
              'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
              'started_at': datetime.now(timezone.utc).isoformat()}
    if cached:
        report.update(cached_jobs=len(cached), planned_new_jobs=len(jobs)-len(cached), completed_new_jobs=0)

    def save():
        temp = output / 'generation_report.tmp'
        temp.write_text(json_text(report)); temp.replace(output / 'generation_report.json')

    save()
    try:
        set_seed(c['seed']); torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', use_safetensors=True)
        model.gradient_checkpointing_disable(); model.config.use_cache = True
        executor = SharedPeftExecutor(model)
        if executor.available_adapters:
            raise ValidationError('Base only')

        class Recording(PeftTextBackend):
            def generate(self, request):
                call = {'request': deepcopy(request), 'request_sha256': canonical_hash(request),
                        'output': None, 'error': None, 'error_type': None, 'usage': None}
                try:
                    call['output'] = super().generate(request)
                    return call['output']
                except Exception as exc:
                    call.update(error=str(exc), error_type=type(exc).__name__)
                    raise
                finally:
                    call['usage'] = self.last_usage; self.calls.append(call)

        backend = Recording(executor, tokenizer, max_context_tokens=c['max_context_tokens'],
                            max_new_tokens=c['max_new_tokens'], stop_on_json_object=True)
        runtime = make_runner(agents, backend, c)
        report['base_model_loads'] = 1; save()
        results = []
        with (output / 'results.jsonl').open('x') as log:
            for row in jobs:
                if row['sample_id'] in cached:
                    result = cached[row['sample_id']]
                    log.write(jsonl([result])); log.flush(); results.append(result)
                    report['completed_jobs'] = len(results); save()
                    print(f"{len(results)}/{len(jobs)} {row['arm']} cached", flush=True)
                    continue
                backend.calls = []; torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                started = time.perf_counter()
                value = runtime.run(input_context(row['input']), workflow=c['workflow'], mode='base')
                torch.cuda.synchronize()
                result = {k: v for k, v in row.items() if k != 'input'}
                result.update(input_sha256=canonical_hash(row['input']), result=value, calls=backend.calls,
                              seconds=time.perf_counter() - started,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated())
                log.write(jsonl([result])); log.flush(); results.append(result)
                report['completed_jobs'] = len(results); save()
                if cached:
                    report['completed_new_jobs'] += 1; save()
                print(f"{len(results)}/{len(jobs)} {row['arm']} {value['status']} {result['seconds']:.1f}s", flush=True)
                if any(x['error_type'] == 'OutOfMemoryError' for x in backend.calls):
                    raise RuntimeError('CUDA OOM')
        replay(jobs, results, agents, c)
        (output / 'scores.json').write_text(json_text(score(c, selected, results)))
        report.update(status='completed', results_sha256=sha256_file(output / 'results.jsonl'),
                      scores_sha256=sha256_file(output / 'scores.json'),
                      all_parameters_frozen=all(not p.requires_grad for p in model.parameters()))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat(); save()
    return report


def audit(output: Path):
    report = strict_json((output / 'generation_report.json').read_text())
    plan = strict_json((output / 'plan.json').read_text())
    if (report['status'] != 'completed' or report['plan_sha256'] != sha256_file(output / 'plan.json')
            or report['results_sha256'] != sha256_file(output / 'results.jsonl')
            or report['scores_sha256'] != sha256_file(output / 'scores.json')
            or report['base_model_loads'] != 1 or not report['all_parameters_frozen'] or report['loaded_adapters']):
        raise ValidationError('run incomplete or changed')
    for name, digest in plan['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(output / name) != digest:
            raise ValidationError('frozen artifact changed')
    c = strict_json((output / 'config.json').read_text())
    if c != plan['config']:
        raise ValidationError('config changed')
    selected, pool, jobs = prepare(c)
    for name, values in (('selected', selected), ('evidence_pool', pool), ('inputs', jobs)):
        if rows(output / (name + '.jsonl')) != values:
            raise ValidationError('historical inputs do not reproduce')
    results = rows(output / 'results.jsonl')
    if len(results) != report['planned_jobs'] or len(results) != report['completed_jobs']:
        raise ValidationError('job count changed')
    if any(any(r[k] != j[k] for k in ('sample_id', 'original_sample_id', 'arm', 'event_group_id'))
           for r, j in zip(results, jobs)):
        raise ValidationError('job identities changed')
    agents, _ = load_capabilities(output / 'agent_config.json')
    cached = cached_controls(c, jobs, agents)
    if cached:
        if (report.get('cached_jobs') != len(cached) or report.get('completed_new_jobs') != len(jobs)-len(cached)
                or report.get('planned_new_jobs') != len(jobs)-len(cached)):
            raise ValidationError('cached/new generation counts differ')
        for r in results:
            if r['sample_id'] in cached and r != cached[r['sample_id']]:
                raise ValidationError('cached result differs from its source')
            if r['sample_id'] not in cached and 'cached_from' in r:
                raise ValidationError('unexpected cached experimental output')
    replay(jobs, results, agents, c)
    if score(c, selected, results) != strict_json((output / 'scores.json').read_text()):
        raise ValidationError('scores do not reproduce')
    result = {'status': 'passed', 'jobs': len(jobs), 'historical_inputs_rebuilt': True,
              'raw_responses_replayed': True, 'scores_recomputed': True,
              'generation_report_sha256': sha256_file(output / 'generation_report.json')}
    (output / 'audit.json').write_text(json_text(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('run')
    for arg in ('config', 'agent-config', 'model-manifest', 'output'):
        gen.add_argument('--' + arg, type=Path, required=True)
    sub.add_parser('audit').add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json_text(run(args.config, args.agent_config, args.model_manifest, args.output)
                    if args.command == 'run' else audit(args.run)))


if __name__ == '__main__':
    main()
