"""Frozen-weight team experience collection; historical fitting/audit helpers retained.

Independent simulated accounts, not a continuous-portfolio or final-test claim.
Only the forecast capability is adapted; researcher/risk/critic share frozen Base.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from importlib.metadata import version
import math
from pathlib import Path
import random
import shutil
import statistics

from .data import sha256_file, strict_json
from .evaluation import json_text
from .lora_probe import verify_model_manifest
from .mean_reversion import HistoricalBars
from .metrics import score_predictions
from .peft_runtime import SharedPeftExecutor, PeftTextBackend
from .schema import timestamp, ValidationError
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .team_learning import TeamRunner, run_episode, require, experience_candidates
from .team_learning_experiment import read_rows
from .team_outcome_learning import OutcomeBackend, binary_prompt, binary_probability, answer_tokens


def now():
    return datetime.now(timezone.utc).isoformat()


def load_partition(config, split):
    require(split in ('train', 'development'), 'unknown partition')
    root = Path(config['cohort']); report = strict_json((root/'report.json').read_text())
    require(sha256_file(root/'report.json') == config['cohort_report_sha256'], 'cohort changed')
    path = root/(split+'.jobs.jsonl')
    require(sha256_file(path) == report['artifact_hashes'][path.name], 'partition changed')
    jobs = read_rows(path)
    require(all(j['partition'] == split for j in jobs), 'partition membership differs')
    if split == 'development':
        # Predeclared earliest observation per anchor event; no outcome selection.
        selected = {}; jobs.sort(key=lambda j: (j['context']['observation_time'], j['context']['episode_id']))
        for job in jobs: selected.setdefault(job['anchor_group_id'], job)
        jobs = list(selected.values())
    return jobs


def make_feed(config, jobs):
    mids = {m['market_id'] for j in jobs for m in j['context']['markets']}
    return HistoricalBars.from_store(Path(config['price_store'])/'prices.sqlite', mids,
        min(timestamp(j['context']['observation_time'], 'start') for j in jobs)-timedelta(days=30),
        timestamp(config['price_read_before'], 'end'))


def training_rows(jobs, episodes):
    """Admit genuine outcomes, not self-written answers, profitability or confidence."""
    require(len(jobs) == len(episodes), 'incomplete training experience')
    rows = []; seen = set()
    for job, episode in zip(jobs, episodes):
        require(job['partition'] == 'train' and job['context'] == episode['context'], 'training source differs')
        require(episode['feedback']['label_sha256'] == canonical_hash(job['labels']), 'training labels differ')
        calls = {canonical_hash(c['request']): c for c in episode['calls'] if c['request']['agent'] == 'forecast'}
        for readout in episode['readouts']:
            call = calls[readout['request_sha256']]
            # Only complete schema-valid forecast calls. Risk vetoes/losses are NOT exclusions.
            if call['error'] is not None: continue
            mid = readout['market_id']; request = call['request']; label = job['labels'][mid]
            require(readout['prompt'] == binary_prompt(request, mid)
                    and readout['prompt_sha256'] == canonical_hash(readout['prompt']), 'readout prompt changed')
            require(readout['probability'] == binary_probability(readout['logits_no_yes']), 'readout probability changed')
            market = next(m for m in job['context']['markets'] if m['market_id'] == mid)
            require(timestamp(market['input']['observation_time'], 'input') < timestamp(label['resolution_time'], 'label'),
                    'resolved target in training context')
            key = (episode['episode_id'], mid)
            if key in seen: continue
            seen.add(key)
            row = {'episode_id': episode['episode_id'], 'market_id': mid, 'event_group_id': market['event_group_id'],
                'prompt': readout['prompt'], 'outcome': label['outcome'], 'label': deepcopy(label),
                'proof': job['provenance'][mid], 'source_episode_sha256': canonical_hash(episode),
                'source_request_sha256': readout['request_sha256'], 'partition': 'train',
                'supervision': 'verified_binary_outcome_conditional_log_loss_not_probability_teacher'}
            row['row_sha256'] = canonical_hash(row); rows.append(row)
    require(rows, 'no independently supervised examples')
    return rows


def build_artifact(jobs, episodes, *, built_at=None):
    require(len(jobs) == len(episodes) and all(j['partition'] == 'train' for j in jobs), 'artifact training scope differs')
    selected = []; lessons = []
    for job, episode in zip(jobs, episodes):
        require(job['context'] == episode['context'], 'artifact source context differs')
        require(episode['feedback']['label_sha256'] == canonical_hash(job['labels']), 'artifact labels changed')
        if episode['reflection'] is None: continue
        selected.append((job, episode))
        for lesson in episode['reflection']['lessons']:
            if lesson not in lessons and len(lesson) <= 256 and len(lessons) < 6: lessons.append(lesson)
    require(selected, 'no valid training reflections')
    artifact = {'kind': 'offline_training_reflections_v1', 'built_at': built_at or now(),
        'source_public_through': max(l['resolution_time'] for j, _ in selected for l in j['labels'].values()),
        'source_event_group_ids': sorted({m['event_group_id'] for j, _ in selected for m in j['context']['markets']}),
        'source_episode_hashes': [canonical_hash(e) for _, e in selected], 'lessons': lessons}
    artifact['artifact_sha256'] = canonical_hash(artifact); return artifact


def reused_experience(config, jobs):
    """Resume a complete training-only collection, never regenerate until success."""
    source = Path(config['reuse_training_experience'])
    previous = strict_json((source/'config.json').read_text())
    comparable = {k: v for k, v in config.items() if k not in ('run_name', 'reuse_training_experience')}
    require(comparable == {k: v for k, v in previous.items() if k != 'run_name'}, 'reuse protocol differs')
    report = strict_json((source/'report.json').read_text()); plan = strict_json((source/'plan.json').read_text())
    require(report['status'] == 'failed' and report['stage'] == 'training_experience'
            and report['completed_development_episodes'] == 0 and report['completed_training_episodes'] == len(jobs)
            and not (source/'training_report.json').exists(), 'reuse is not an unfitted completed training collection')
    require(report['plan_sha256'] == sha256_file(source/'plan.json')
            and plan['config_sha256'] == sha256_file(source/'config.json')
            and plan['model_manifest_sha256'] == sha256_file(source/'model_manifest.json')
            and plan['train_jobs_sha256'] == sha256_file(source/'train.jobs.jsonl')
            and read_rows(source/'train.jobs.jsonl') == jobs, 'reused jobs differ')
    for name, digest in plan['source_hashes'].items(): require(sha256_file(source/name) == digest, 'source snapshot changed')
    episodes = read_rows(source/'train.episodes.jsonl'); require(len(episodes) == len(jobs), 'incomplete reused experience')
    feed = make_feed(config, jobs)
    for job, episode in zip(jobs, episodes):
        require(job['context'] == episode['context'], 'reused context changed')
        class Replay:
            def __init__(self): self.i = 0
            def generate(self, request):
                call = episode['calls'][self.i]; self.i += 1
                require(request == call['request'], 'reused request differs under current workflow')
                if call['output'] is None: raise ValidationError(call['error'])
                return call['output']
        replay = Replay()
        rebuilt = run_episode(job['context'], job['labels'], feed, config['execution_policy'], TeamRunner(replay),
                              recorded_latency=episode['decision_seconds'])
        for field in ('decision', 'decision_error', 'reflection', 'reflection_error', 'feedback', 'decision_call_count'):
            require(rebuilt[field] == episode[field], 'reused episode replay differs: '+field)
        require(replay.i == len(episode['calls']), 'unused reused calls')
    training_rows(jobs, episodes)
    return episodes, {'source': str(source), 'episodes_sha256': sha256_file(source/'train.episodes.jsonl'),
                      'report_sha256': sha256_file(source/'report.json'), 'replayed_episodes': len(episodes)}


def fit_outcomes(model, tokenizer, rows, config, output):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed
    tc = config['training']; set_seed(config['seed']); token_ids = answer_tokens(tokenizer)
    encoded = [tokenizer.encode(r['prompt'], add_special_tokens=False) for r in rows]
    require(all(0 < len(ids) <= config['max_scoring_tokens'] for ids in encoded), 'training sequence over budget')
    counts = Counter(r['event_group_id'] for r in rows)
    # Full-dataset normalized weights: each independent event has equal total weight.
    weights = [len(rows)/(len(counts)*counts[r['event_group_id']]) for r in rows]
    (output/'training_tokens.jsonl').write_text(jsonl([
        {'row_sha256': r['row_sha256'], 'input_ids': ids, 'outcome': r['outcome'], 'weight': w}
        for r, ids, w in zip(rows, encoded, weights)]))
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=tc['r'], lora_alpha=tc['alpha'], lora_dropout=tc['dropout'],
        target_modules=tc['target_modules'], bias='none', task_type='CAUSAL_LM',
        base_model_name_or_path=config['model'], revision=config['model_revision']), autocast_adapter_dtype=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    require(trainable and all('lora_' in n and p.dtype == torch.float32 for n, p in trainable), 'unexpected trainable weights')
    params = [p for _, p in trainable]; tracked = next(p for n, p in trainable if 'lora_B' in n)
    before = tracked.detach().cpu().clone()
    optimizer = torch.optim.AdamW(params, lr=tc['learning_rate'], weight_decay=0, foreach=False)
    result = {'kind': 'team_outcome_lora_fit_v1', 'started_at': now(), 'status': 'running',
        'examples': len(rows), 'event_groups': len(counts), 'optimizer_steps': [],
        'trainable_parameters': sum(p.numel() for p in params), 'base_parameters_frozen': True,
        'objective': 'event_balanced_two_token_conditional_log_loss', 'quantization': None,
        'base_dtype': 'bfloat16', 'adapter_dtype': 'float32', 'development_loaded': False}
    def save(): (output/'training_report.json').write_text(json_text(result))
    save()
    try:
        for epoch in range(tc['epochs']):
            model.train(); order = list(range(len(rows))); random.Random(config['seed']+epoch).shuffle(order)
            for start in range(0, len(order), tc['gradient_accumulation_steps']):
                batch = order[start:start+tc['gradient_accumulation_steps']]
                optimizer.zero_grad(set_to_none=True); losses = []
                for i in batch:
                    tokens = torch.tensor([encoded[i]], device='cuda')
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                       logits_to_keep=1, use_cache=False).logits[0, -1, token_ids].float()
                        raw = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([rows[i]['outcome']], device='cuda'))
                        loss = raw*weights[i]/len(batch)
                    require(math.isfinite(raw.item()), 'nonfinite outcome loss')
                    losses.append(raw.item()); loss.backward(); del loss, raw, logits, tokens
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                require(math.isfinite(norm.item()) and all(p.grad is None for p in frozen), 'invalid or unfrozen base gradient')
                optimizer.step(); torch.cuda.synchronize()
                result['optimizer_steps'].append({'epoch': epoch+1, 'examples': len(batch),
                    'mean_unweighted_nll': statistics.fmean(losses), 'gradient_norm': norm.item()})
                save(); print('Training step', len(result['optimizer_steps']), result['optimizer_steps'][-1], flush=True)
        require(not torch.equal(before, tracked.detach().cpu()), 'adapter did not update')
        optimizer.zero_grad(set_to_none=True)
        model.save_pretrained(output/'forecast_adapter', safe_serialization=True)
        torch.save(optimizer.state_dict(), output/'optimizer_state.pt')
        result.update(status='completed', adapter_updated=True, finished_at=now(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            checkpoint_hashes={p.name: sha256_file(p) for p in (output/'forecast_adapter').iterdir() if p.is_file()},
            optimizer_state_sha256=sha256_file(output/'optimizer_state.pt'))
        save()
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}'); save(); raise
    del optimizer; model.gradient_checkpointing_disable(); model.config.use_cache = True
    model.eval(); model.requires_grad_(False); torch.cuda.empty_cache()
    return model, result


def summarize(episodes):
    ps = []; ys = []; qs = []; group_scores = defaultdict(list)
    for e in episodes:
        predictions = {} if e['decision'] is None else {r['market_id']: r['probability'] for r in e['decision']['forecast']['forecasts']}
        for market in e['context']['markets']:
            mid = market['market_id']; p = predictions.get(mid); y = e['feedback']['outcomes'][mid]
            ps.append(p); ys.append(y); qs.append(market['input']['market']['probability'])
            if p is not None: group_scores[market['event_group_id']].append((p-y)**2)
    accounts = [e['feedback']['account'] for e in episodes]
    return {'episodes': len(episodes), 'decision_successes': sum(e['decision'] is not None for e in episodes),
        'forecast': score_predictions(ps, ys), 'market': score_predictions(qs, ys),
        'event_balanced_brier_on_covered_groups': statistics.fmean(statistics.fmean(v) for v in group_scores.values()) if group_scores else None,
        'covered_event_groups': len(group_scores), 'mean_independent_account_final_cash': statistics.fmean(float(a['final_cash']) for a in accounts),
        'mean_independent_account_net_pnl': statistics.fmean(float(a['net_pnl']) for a in accounts),
        'total_filled_trades': sum(a['filled_trades'] for a in accounts),
        'total_fees_across_independent_accounts': sum(float(a['fees_paid']) for a in accounts),
        'mean_decision_seconds': statistics.fmean(e['decision_seconds'] for e in episodes),
        'model_calls': sum(len(e['calls']) for e in episodes),
        'invalid_calls': sum(c['error'] is not None for e in episodes for c in e['calls']),
        'valid_reflections': sum(e['reflection'] is not None for e in episodes),
        'simulation_only': True, 'continuous_portfolio': False, 'cash_baseline_per_account': 100}


def validate_exploration_config(config):
    require(config.get('kind') == 'team_exploration_v1'
            and config.get('automatic_fine_tuning') is False
            and config.get('effectiveness_evidence_required') is True
            and config.get('final_test_opened') is False
            and config.get('sampling') is None
            and 'training' not in config and 'reuse_training_experience' not in config,
            'Automatic RSI fine-tuning is retired; use a frozen-weight team_exploration_v1 config. '
            'Independent evidence of team improvement is required before a separate fine-tuning experiment.')


def finish_exploration(output, jobs, episodes):
    require(len(jobs) == len(episodes) and jobs, 'incomplete exploration')
    for job, episode in zip(jobs, episodes):
        require(job['partition'] == 'train' and job['context'] == episode['context']
                and episode['feedback']['label_sha256'] == canonical_hash(job['labels']), 'exploration source differs')
    candidates = experience_candidates(episodes)
    artifact = build_artifact(jobs, episodes) if any(e['reflection'] is not None for e in episodes) else None
    (output/'experience_candidates.jsonl').write_text(jsonl(candidates))
    (output/'learning_artifact.json').write_text(json_text(artifact))
    return {'learning_summary': summarize(episodes), 'generated_candidates': len(candidates),
            'learned_memory_lessons': len(artifact['lessons']) if artifact else 0,
            'effectiveness_verified': False, 'fine_tuning_triggered': False, 'admitted_training_examples': 0,
            'next_stage': 'freeze_candidate_team_method_and_validate_on_fresh_event_time_groups'}


def run(config_path, model_manifest, output):
    require(not output.exists(), 'RSI run exists')
    config = strict_json(config_path.read_text())
    validate_exploration_config(config)
    train_jobs = load_partition(config, 'train')
    reused, reuse_record = reused_experience(config, train_jobs) if config.get('reuse_training_experience') else (None, None)
    store = Path(config['price_store']); prices = strict_json((store/'report.json').read_text())
    require(sha256_file(store/'report.json') == config['price_store_report_sha256']
            and sha256_file(store/'prices.sqlite') == prices['artifact_hashes']['prices.sqlite'], 'prices changed')
    print('Frozen training jobs:', len(train_jobs), 'verifying local model', flush=True)
    model_path, model_hash = verify_model_manifest(model_manifest, config)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'BF16 CUDA unavailable')
    output.mkdir(parents=True); shutil.copyfile(config_path, output/'config.json')
    shutil.copyfile(model_manifest, output/'model_manifest.json')
    shutil.copytree(Path(__file__).parent, output/'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    (output/'train.jobs.jsonl').write_text(jsonl(train_jobs))
    plan = {'created_at': now(), 'config_sha256': sha256_file(config_path), 'model_manifest_sha256': model_hash,
            'train_jobs_sha256': sha256_file(output/'train.jobs.jsonl'),
            'cohort_report_sha256': config['cohort_report_sha256'],
            'source_hashes': {str(p.relative_to(output)): sha256_file(p) for p in sorted((output/'source_snapshot').rglob('*.py'))}}
    if reuse_record is not None: plan['reused_training_experience'] = reuse_record
    (output/'plan.json').write_text(json_text(plan))
    report = {'status': 'running', 'stage': 'load_base', 'started_at': now(), 'plan_sha256': sha256_file(output/'plan.json'),
        'packages': {p: version(p) for p in ('torch', 'transformers', 'peft', 'langgraph')},
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'base_model_loads': 0,
        'final_test_opened': False, 'existing_holdouts_opened': False, 'default_promoted': False, 'real_orders_sent': 0,
        'completed_training_episodes': 0, 'completed_development_episodes': 0}
    def save():
        p = output/'report.tmp'; p.write_text(json_text(report)); p.replace(output/'report.json')
    save()
    try:
        set_seed(config['seed']); torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', use_safetensors=True)
        report['base_model_loads'] = 1
        def backend_for(executor, adapter=None):
            return OutcomeBackend(PeftTextBackend(executor, tokenizer, max_context_tokens=config['max_context_tokens'],
                max_new_tokens=config['max_new_tokens'], sampling=None, stop_on_json_object=True),
                forecast_adapter=adapter, max_scoring_tokens=config['max_scoring_tokens'])
        executor = SharedPeftExecutor(model); backend = backend_for(executor)
        feed = make_feed(config, train_jobs); train_episodes = []; report['stage'] = 'training_experience'; save()
        with (output/'train.episodes.jsonl').open('x') as stream:
            for index, job in enumerate(train_jobs):
                backend.readouts = []; set_seed(config['seed'])
                if reused is None:
                    episode = run_episode(job['context'], job['labels'], feed, config['execution_policy'], TeamRunner(backend))
                    episode['readouts'] = deepcopy(backend.readouts)
                else: episode = deepcopy(reused[index])
                train_episodes.append(episode)
                stream.write(jsonl([episode])); stream.flush(); report['completed_training_episodes'] += 1; save()
                print('Learning episode', report['completed_training_episodes'], 'decision', episode['decision'] is not None, flush=True)
        # Learning episodes never automatically update model parameters. Effectiveness
        # must first be established by a separate, frozen, independent comparison.
        report.update(finish_exploration(output, train_jobs, train_episodes))
        require(all(not p.requires_grad for p in model.parameters()), 'exploration unfroze model parameters')
        report.update(status='completed', stage='exploration_completed', finished_at=now(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            all_parameters_frozen=True,
            limitations=['Experience collection is not proof of effective autonomous learning.',
                'No independent effectiveness evaluation or fine-tuning is performed by this entry point.',
                'Generated experience remains unqualified; no automatic adapter or policy promotion.',
                'Independent $100 episodic simulations, not a continuous portfolio.',
                'Sparse peers, historical print fills and possible foundation-model historical contamination remain limitations.'])
        report['artifact_hashes'] = {p.name: sha256_file(p) for p in output.iterdir()
            if p.is_file() and p.name not in ('report.json', 'report.tmp', 'audit.json')}
        save()
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=now()); save(); raise
    return report


def audit_exploration(output, report, config):
    jobs = load_partition(config, 'train'); episodes = read_rows(output/'train.episodes.jsonl')
    require(jobs == read_rows(output/'train.jobs.jsonl') and len(jobs) == len(episodes), 'exploration population differs')
    require(report['fine_tuning_triggered'] is False and report['effectiveness_verified'] is False
            and report['admitted_training_examples'] == 0
            and not (output/'forecast_adapter').exists() and not (output/'training_report.json').exists(),
            'unexpected fitting in exploration')
    feed = make_feed(config, jobs)
    for job, episode in zip(jobs, episodes):
        require(job['context'] == episode['context'], 'exploration context changed')
        class Replay:
            def __init__(self): self.i = 0
            def generate(self, request):
                call = episode['calls'][self.i]; self.i += 1
                require(request == call['request'], 'exploration request changed')
                if call['output'] is None: raise ValidationError(call['error'])
                return call['output']
        replay = Replay()
        rebuilt = run_episode(job['context'], job['labels'], feed, config['execution_policy'], TeamRunner(replay),
                              recorded_latency=episode['decision_seconds'])
        for key in ('decision', 'decision_error', 'reflection', 'reflection_error', 'feedback', 'decision_call_count'):
            require(rebuilt[key] == episode[key], 'exploration replay differs: '+key)
        require(replay.i == len(episode['calls']), 'unused exploration calls')
    require(experience_candidates(episodes) == read_rows(output/'experience_candidates.jsonl')
            and summarize(episodes) == report['learning_summary'], 'exploration exports changed')
    artifact = strict_json((output/'learning_artifact.json').read_text())
    expected = build_artifact(jobs, episodes, built_at=artifact['built_at']) if artifact is not None else None
    require(artifact == expected and (artifact is not None or all(e['reflection'] is None for e in episodes)),
            'exploration reflection artifact differs')
    result = {'status': 'passed', 'report_sha256': sha256_file(output/'report.json'), 'replayed_episodes': len(episodes),
              'fine_tuning_performed': False, 'effectiveness_verified': False, 'final_test_opened': False}
    (output/'audit.json').write_text(json_text(result)); return result


def audit(output):
    report = strict_json((output/'report.json').read_text()); config = strict_json((output/'config.json').read_text())
    plan = strict_json((output/'plan.json').read_text()); require(report['status'] == 'completed', 'unfinished run')
    require(report['plan_sha256'] == sha256_file(output/'plan.json'), 'plan changed')
    for name, digest in {**report['artifact_hashes'], **plan['source_hashes']}.items():
        p = (output/name).resolve(); require(p.is_relative_to(output.resolve()) and sha256_file(p) == digest, 'artifact changed')
    if config['kind'] == 'team_exploration_v1':
        validate_exploration_config(config)
        return audit_exploration(output, report, config)
    freeze = strict_json((output/'freeze.json').read_text())
    for name, digest in freeze['checkpoint_hashes'].items():
        require(sha256_file(output/'forecast_adapter'/name) == digest, 'checkpoint changed')
    train_jobs = load_partition(config, 'train'); train = read_rows(output/'train.episodes.jsonl')
    if config.get('reuse_training_experience'):
        original, record = reused_experience(config, train_jobs)
        require(original == train and record == plan['reused_training_experience'], 'reused collection differs')
    require(train_jobs == read_rows(output/'train.jobs.jsonl') and training_rows(train_jobs, train) == read_rows(output/'training_examples.jsonl'),
            'training admission does not reproduce')
    artifact = strict_json((output/'learning_artifact.json').read_text())
    require(artifact == build_artifact(train_jobs, train, built_at=artifact['built_at']), 'reflection source changed')
    dev_jobs = load_partition(config, 'development'); require(dev_jobs == read_rows(output/'development.jobs.jsonl'), 'dev scope changed')
    replayed = 0
    for name in ('train', 'base', 'base_memory', 'lora', 'lora_memory'):
        jobs = train_jobs if name == 'train' else dev_jobs; feed = make_feed(config, jobs)
        episodes = train if name == 'train' else read_rows(output/(name+'.episodes.jsonl'))
        require(len(jobs) == len(episodes), 'missing episodes')
        for job, episode in zip(jobs, episodes):
            require(job['context'] == episode['context'], 'evaluation context differs')
            class Replay:
                def __init__(self): self.i = 0
                def generate(self, request):
                    call = episode['calls'][self.i]; self.i += 1
                    require(request == call['request'], 'replay request differs')
                    if call['output'] is None: raise ValidationError(call['error'])
                    return call['output']
            backend = Replay(); artifacts = [artifact] if name.endswith('_memory') and artifact['lessons'] else []
            rebuilt = run_episode(job['context'], job['labels'], feed, config['execution_policy'],
                TeamRunner(backend, learning_artifacts=artifacts), recorded_latency=episode['decision_seconds'])
            for field in ('decision', 'decision_error', 'reflection', 'reflection_error', 'feedback', 'decision_call_count'):
                require(rebuilt[field] == episode[field], 'episode replay differs: '+field)
            require(backend.i == len(episode['calls']), 'unused calls')
            calls = {canonical_hash(c['request']): c for c in episode['calls']}
            for row in episode['readouts']:
                require(row['prompt'] == binary_prompt(calls[row['request_sha256']]['request'], row['market_id'])
                        and row['probability'] == binary_probability(row['logits_no_yes']), 'scoring readout differs')
            replayed += 1
        expected = report['training_summary'] if name == 'train' else report['arms'][name]
        require(summarize(episodes) == expected, 'aggregate metrics differ')
    result = {'status': 'passed', 'report_sha256': sha256_file(output/'report.json'), 'replayed_episodes': replayed,
        'raw_requests_tools_feedback_ledgers_and_metrics_reproduced': True,
        'training_supervision_and_reflection_sources_reproduced': True, 'checkpoint_hashes_verified': True,
        'new_model_calls': 0, 'final_test_opened': False}
    (output/'audit.json').write_text(json_text(result)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    for k in ('config', 'model-manifest', 'output'): r.add_argument('--'+k, type=Path, required=True)
    sub.add_parser('audit').add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(json_text(audit(a.output) if a.command == 'audit' else run(a.config, a.model_manifest, a.output)))
