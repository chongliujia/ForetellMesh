"""Run/replay a real Base team interface smoke test on admitted training inputs.

This is explicitly not the broad-market learning experiment or SFT admission.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta, timezone, datetime
from importlib.metadata import version
from pathlib import Path
import shutil

from .data import sha256_file, strict_json
from .evaluation import json_text
from .lora_probe import verify_model_manifest
from .mean_reversion import HistoricalBars
from .peft_runtime import SharedPeftExecutor, PeftTextBackend
from .schema import ValidationError, timestamp, iso
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .team_learning import MemoryBank, TeamRunner, run_episode, experience_candidates, require, validate_context


def read_rows(path):
    return [strict_json(s) for s in path.read_text().splitlines() if s]


def prepare(config):
    require(config['kind'] == 'team_learning_interface_smoke_v1' and config['partition'] == 'train'
            and config['training'] is False and config['scope'] == 'existing_admitted_inputs_interface_check_only'
            and config['final_test_opened'] is False and type(config['max_episodes']) is int
            and 1 <= config['max_episodes'] <= 12 and type(config['max_markets']) is int
            and 1 <= config['max_markets'] <= 4, 'invalid smoke scope')
    dataset = Path(config['dataset']); store = Path(config['price_store'])
    require(sha256_file(dataset/'report.json') == config['dataset_report_sha256']
            and sha256_file(store/'report.json') == config['price_store_report_sha256'], 'smoke data binding changed')
    report = strict_json((dataset/'report.json').read_text()); prices = strict_json((store/'report.json').read_text())
    for name in ('partitions/train.inputs.jsonl', 'partitions/train.labels.jsonl', 'membership.jsonl'):
        require(sha256_file(dataset/name) == report['artifact_hashes'][name], 'training artifact changed: '+name)
    require(sha256_file(store/'prices.sqlite') == prices['artifact_hashes']['prices.sqlite'], 'price database changed')
    members = {r['sample_id']: r for r in read_rows(dataset/'membership.jsonl') if r['split'] == 'train'}
    inputs = read_rows(dataset/'partitions/train.inputs.jsonl'); labels = read_rows(dataset/'partitions/train.labels.jsonl')
    require(set(members) == {r['sample_id'] for r in inputs} == {r['sample_id'] for r in labels}, 'training population differs')
    label_by_id = {r['sample_id']: r['label'] for r in labels}; grouped = defaultdict(list)
    for row in inputs:
        grouped[members[row['sample_id']]['event_group_id']].append(row)
    selected = []
    for group, rows in grouped.items():
        first = min(timestamp(r['input']['observation_time'], 'time') for r in rows)
        batch = sorted([r for r in rows if timestamp(r['input']['observation_time'], 'time') == first],
                       key=lambda r: r['sample_id'])[:config['max_markets']]
        context = {'episode_id': 'team-smoke:'+group, 'observation_time': iso(first),
                   'markets': [{'market_id': members[r['sample_id']]['event_id'].removeprefix('polymarket:'),
                                'event_group_id': group, 'input': deepcopy(r['input'])} for r in batch],
                   'account': {'initial_cash': '100', 'simulation_only': True}, 'memory': []}
        validate_context(context)
        selected.append({'context': context, 'labels': {m['market_id']: label_by_id[r['sample_id']]
                                                       for r, m in zip(batch, context['markets'])}})
    selected = sorted(selected, key=lambda r: (r['context']['observation_time'], r['context']['episode_id']))[:config['max_episodes']]
    require(selected, 'empty episode scope')
    market_ids = {m['market_id'] for r in selected for m in r['context']['markets']}
    feed = HistoricalBars.from_store(store/'prices.sqlite', market_ids,
        timestamp(selected[0]['context']['observation_time'], 'start')-timedelta(days=30),
        timestamp(config['price_read_before'], 'price upper bound'))
    return selected, feed


def run(config_path, model_manifest, output):
    require(not output.exists(), 'team run output exists')
    config = strict_json(config_path.read_text()); jobs, feed = prepare(config)
    print(f'Prepared {len(jobs)} admitted interface episodes; verifying model', flush=True)
    model_path, model_hash = verify_model_manifest(model_manifest, config)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'BF16 CUDA unavailable')
    output.mkdir(parents=True); shutil.copyfile(config_path, output/'config.json')
    shutil.copyfile(model_manifest, output/'model_manifest.json')
    shutil.copytree(Path(__file__).parent, output/'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    (output/'jobs.jsonl').write_text(jsonl(jobs))
    plan = {'config_sha256': sha256_file(config_path), 'jobs_sha256': sha256_file(output/'jobs.jsonl'),
            'model_manifest_sha256': model_hash,
            'source_hashes': {str(p.relative_to(output)): sha256_file(p) for p in sorted((output/'source_snapshot').rglob('*.py'))}}
    (output/'plan.json').write_text(json_text(plan))
    report = {'status': 'running', 'scope': config['scope'], 'training': False, 'default_promoted': False,
              'planned_episodes': len(jobs), 'completed_episodes': 0, 'successful_decisions': 0,
              'model_calls': 0, 'real_orders_sent': 0, 'final_test_opened': False,
              'base_model_loads': 0, 'packages': {p: version(p) for p in ('torch', 'transformers', 'peft', 'langgraph')},
              'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
              'started_at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha256_file(output/'plan.json')}
    def save():
        path = output/'report.tmp'; path.write_text(json_text(report)); path.replace(output/'report.json')
    save(); episodes = []
    try:
        set_seed(config['seed']); tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', use_safetensors=True)
        model.gradient_checkpointing_disable(); model.config.use_cache = True
        executor = SharedPeftExecutor(model)
        backend = PeftTextBackend(executor, tokenizer, max_context_tokens=config['max_context_tokens'],
                                  max_new_tokens=config['max_new_tokens'], sampling=config['sampling'], stop_on_json_object=True)
        runner = TeamRunner(backend); bank = MemoryBank(); report['base_model_loads'] = 1; save()
        with (output/'episodes.jsonl').open('x') as stream:
            for job in jobs:
                context = deepcopy(job['context'])
                context['memory'] = bank.at(context['observation_time'],
                                           excluded_groups={m['event_group_id'] for m in context['markets']})
                result = run_episode(context, job['labels'], feed, config['execution_policy'], runner)
                episodes.append(result); stream.write(jsonl([result])); stream.flush()
                if result['memory'] is not None: bank.add(result['memory'])
                report['completed_episodes'] += 1; report['successful_decisions'] += result['decision'] is not None
                report['model_calls'] += len(result['calls']); save()
                print(f"Episode {len(episodes)}/{len(jobs)}: decision={result['decision'] is not None}, "
                      f"queries={len(result['decision']['tools']) if result['decision'] else 0}, "
                      f"cash={result['feedback']['account']['final_cash']}, memory_visible={len(context['memory'])}", flush=True)
        (output/'memory.jsonl').write_text(jsonl(bank.rows))
        candidates = experience_candidates(episodes)
        (output/'experience_candidates.jsonl').write_text(jsonl(candidates))
        report.update(status='completed', all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
            generated_candidates=len(candidates), admitted_training_examples=0,
            memory_entries=len(bank.rows), memory_entries_used=sum(len(e['context']['memory']) for e in episodes),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            per_episode=[{'episode_id': e['episode_id'], 'facts': e['feedback']['facts'],
                         'decision_error': e['decision_error'], 'reflection_error': e['reflection_error']} for e in episodes],
            limitations=['Existing macro training inputs validate interfaces only, not broad-market learning or generalization.',
                'Independent $100 episodic accounts, not a continuous portfolio return.',
                'Feedback availability retains source label available_at; no backdated feedback memory.',
                'Generated lessons are hypotheses; no automatic SFT admission or forecast skill claim.',
                'Hypothetical trade-print fills have no order-book/depth guarantee.'],
            artifact_hashes={name: sha256_file(output/name) for name in
                ('config.json', 'jobs.jsonl', 'episodes.jsonl', 'memory.jsonl', 'experience_candidates.jsonl')})
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat(); save()
    return report


def audit(output):
    report = strict_json((output/'report.json').read_text()); config = strict_json((output/'config.json').read_text())
    plan = strict_json((output/'plan.json').read_text())
    require(report['status'] == 'completed' and report['plan_sha256'] == sha256_file(output/'plan.json'), 'unfinished or changed run')
    for name, digest in {**plan['source_hashes'], **report['artifact_hashes']}.items():
        path = (output/name).resolve()
        require(path.is_relative_to(output.resolve()) and sha256_file(path) == digest, 'run artifact changed')
    jobs, feed = prepare(config); episodes = read_rows(output/'episodes.jsonl')
    require(jobs == read_rows(output/'jobs.jsonl') and len(jobs) == len(episodes), 'episode source population changed')
    bank = MemoryBank(); replayed = 0
    for job, episode in zip(jobs, episodes):
        context = deepcopy(job['context'])
        context['memory'] = bank.at(context['observation_time'], excluded_groups={m['event_group_id'] for m in context['markets']})
        require(context == episode['context'], 'historical context or memory differs')
        class Replay:
            def __init__(self): self.index = 0
            def generate(self, request):
                require(self.index < len(episode['calls']), 'extra model call during replay')
                call = episode['calls'][self.index]; self.index += 1
                require(request == call['request'], 'model request differs on replay')
                if call['output'] is None: raise ValidationError(call['error'])
                return call['output']
        backend = Replay()
        rebuilt = run_episode(context, job['labels'], feed, config['execution_policy'], TeamRunner(backend),
                              recorded_latency=episode['decision_seconds'])
        for field in ('decision', 'decision_error', 'feedback', 'reflection', 'reflection_error', 'decision_call_count'):
            require(rebuilt[field] == episode[field], 'episode replay differs: '+field)
        require(backend.index == len(episode['calls']), 'unused recorded calls')
        if episode['memory'] is not None:
            memory = episode['memory']
            require(memory['reflection'] == episode['reflection']
                    and memory['source_episode_id'] == episode['episode_id']
                    and memory['feedback_sha256'] == canonical_hash(episode['feedback'])
                    and timestamp(memory['available_at'], 'memory') >= timestamp(episode['feedback']['available_at'], 'feedback'),
                    'memory source/availability differs')
            bank.add(memory)
        replayed += 1
    require(bank.rows == read_rows(output/'memory.jsonl'), 'memory ledger differs')
    require(experience_candidates(episodes) == read_rows(output/'experience_candidates.jsonl'), 'candidate export differs')
    result = {'status': 'passed', 'report_sha256': sha256_file(output/'report.json'), 'replayed_episodes': replayed,
              'raw_requests_outputs_tools_scores_and_ledgers_reproduced': True, 'new_model_calls': 0,
              'training_performed': False, 'final_test_opened': False}
    (output/'audit.json').write_text(json_text(result)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    generate = sub.add_parser('run')
    for key in ('config', 'model-manifest', 'output'): generate.add_argument('--'+key, type=Path, required=True)
    sub.add_parser('audit').add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json_text(audit(a.output) if a.command == 'audit' else run(a.config, a.model_manifest, a.output)))
