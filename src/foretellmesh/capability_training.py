"""Small, fixed-budget BF16 capability LoRA training, with saved provenance."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

from .data import sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .lora_probe import verify_model_manifest
from .peft_runtime import render_agent_prompt
from .research_tool_data import read_research_tool_data
from .schema import ValidationError, fields
from .sft_data import encode_sft_pair, jsonl


def load_training_config(path: Path) -> dict:
    c = strict_json(path.read_text())
    fixed = {'schema_version': '1', 'model': 'Qwen/Qwen3-8B-Base',
             'model_revision': '49e3418fbbbca6ecbdf9608b4d22e5a407081db4',
             'capability': 'research_tool_lora', 'micro_batch_size': 1, 'base_dtype': 'bfloat16',
             'adapter_dtype': 'float32', 'quantization': None, 'attention': 'sdpa',
             'gradient_checkpointing': True, 'optimizer': 'adamw', 'loss': 'mean_completion_loss_per_example',
             'checkpoint_selection': 'last_fixed_epoch_no_validation_selection'}
    fields(c, set(fixed) | {'run_name', 'seed', 'epochs', 'max_sequence_length', 'gradient_accumulation_steps',
           'learning_rate', 'weight_decay', 'max_grad_norm', 'lora_r', 'lora_alpha', 'lora_dropout', 'target_modules'}, 'capability training config')
    if any(type(c[k]) is not type(v) or c[k] != v for k, v in fixed.items()):
        raise ValidationError('unsupported capability training policy')
    if not isinstance(c['run_name'], str) or not c['run_name'].strip():raise ValidationError('run name is required')
    for key, low, high in (('seed', 0, 2**32-1), ('epochs', 1, 4), ('max_sequence_length', 256, 2048),
                           ('gradient_accumulation_steps', 1, 32), ('lora_r', 16, 32), ('lora_alpha', 16, 64)):
        if type(c[key]) is not int or not low <= c[key] <= high:raise ValidationError('invalid training integer')
    for key in ('learning_rate', 'weight_decay', 'max_grad_norm', 'lora_dropout'):
        if type(c[key]) not in (int, float) or not math.isfinite(c[key]) or c[key] < 0:raise ValidationError('invalid training number')
    if not 0 < c['learning_rate'] <= .001 or not 0 < c['max_grad_norm'] <= 10 or c['lora_dropout'] >= 1:
        raise ValidationError('invalid learning rate/clipping/dropout')
    if c['target_modules'] != ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        raise ValidationError('expected seven LoRA projection targets')
    return c


def encode_capability_row(tokenizer, row: dict, max_length: int) -> dict:
    # The supervised response is never concatenated into the model prompt.
    encoded = encode_sft_pair(tokenizer, {'prompt': render_agent_prompt(row['request']),
        'completion': json.dumps(row['target'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))}, max_length)
    return {'sample_id': row['sample_id'], **encoded}


def epoch_batches(count: int, accumulation: int, seed: int, epoch: int) -> list[list[int]]:
    if count < 1 or accumulation < 1:raise ValidationError('empty training set or invalid accumulation')
    order = list(range(count)); random.Random(seed + epoch).shuffle(order)
    return [order[start:start + accumulation] for start in range(0, count, accumulation)]


def train_capability(bundle: Path, model_manifest: Path, config_path: Path, output: Path) -> dict:
    if output.exists():raise ValidationError('training output already exists')
    config = load_training_config(config_path)
    data_manifest, partitions, _ = read_research_tool_data(bundle)
    model_path, model_hash = verify_model_manifest(model_manifest, config)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValidationError('CUDA BF16 is required')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    encoded = {p: [encode_capability_row(tokenizer, row, config['max_sequence_length']) for row in partitions[p]]
               for p in ('train', 'validation')}
    if not encoded['train'] or not encoded['validation']:raise ValidationError('empty training/validation partition')
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent, output / 'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    (output / 'config.json').write_bytes(config_path.read_bytes())
    (output / 'dataset_manifest.json').write_bytes((bundle / 'manifest.json').read_bytes())
    for p, rows in encoded.items():(output / (p + '_tokens.jsonl')).write_text(jsonl(rows))
    report = {'schema_version': '1', 'kind': 'synthetic_capability_lora_training', 'status': 'running',
              'started_at': datetime.now(timezone.utc).isoformat(), 'config': config, 'config_sha256': sha256_file(config_path),
              'dataset_manifest_sha256': sha256_file(bundle / 'manifest.json'), 'dataset_version': data_manifest['dataset_version'],
              'split_version': data_manifest['split_version'], 'model_manifest_sha256': model_hash, 'code': code_provenance(),
              'python_executable': sys.executable, 'packages': {p: version(p) for p in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors')},
              'gpu': torch.cuda.get_device_name(0), 'cuda_version': torch.version.cuda, 'quantization': None,
              'tokenized': {p: {'sha256': sha256_file(output / (p + '_tokens.jsonl')), 'examples': len(rows),
                                'max_length': max(len(r['input_ids']) for r in rows)} for p, rows in encoded.items()},
              'optimizer_steps': [], 'validation_losses': [], 'checkpoint_saved': False, 'default_promoted': False,
              'reward_implementation': None, 'rl_performed': False,
              'limitations': data_manifest['limitations'] + ['Fixed two-epoch behavior experiment; NLL is not capability or forecasting quality.']}
    def save():
        temp = output / 'report.tmp'; temp.write_text(json_text(report)); temp.replace(output / 'report.json')
    def memory():
        return {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    save()
    try:
        torch.cuda.set_device(0); set_seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                    dtype=torch.bfloat16, device_map={'': 0}, attn_implementation=config['attention'], use_safetensors=True)
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=config['lora_r'], lora_alpha=config['lora_alpha'],
                    lora_dropout=config['lora_dropout'], target_modules=config['target_modules'], bias='none',
                    task_type='CAUSAL_LM', base_model_name_or_path=config['model'], revision=config['model_revision']), autocast_adapter_dtype=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        frozen = [(n,p) for n,p in model.named_parameters() if not p.requires_grad]
        if not trainable or any('lora_' not in n or p.dtype != torch.float32 for n,p in trainable):
            raise RuntimeError('only FP32 LoRA parameters may be trained')
        if any(p.dtype != torch.bfloat16 or p.device.type != 'cuda' for _,p in frozen):
            raise RuntimeError('base must remain BF16 on CUDA')
        params = [p for _,p in trainable]
        report['trainable_parameters'] = sum(p.numel() for p in params)
        report['frozen_parameters'] = sum(p.numel() for _,p in frozen)
        tracked = next(p for n,p in trainable if 'lora_B' in n)
        before = tracked.detach().cpu().clone()
        optimizer = torch.optim.AdamW(params, lr=config['learning_rate'], weight_decay=config['weight_decay'], foreach=False)
        report['optimizer_defaults'] = dict(optimizer.defaults)
        def batch(row):
            return {k: torch.tensor([row[k]], dtype=torch.long, device='cuda') for k in ('input_ids', 'attention_mask', 'labels')}
        def validation_loss(epoch):
            model.eval(); weighted, tokens = 0., 0
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for row in encoded['validation']:
                    n = sum(x != -100 for x in row['labels'])
                    value = model(**batch(row)).loss.item()
                    if not math.isfinite(value):raise RuntimeError('non-finite validation loss')
                    weighted += value * n; tokens += n
            return {'epoch': epoch, 'mean_completion_token_nll': weighted / tokens, 'supervised_tokens': tokens}
        report['validation_losses'].append(validation_loss(0)); save()
        for epoch in range(config['epochs']):
            model.train()
            batches = epoch_batches(len(encoded['train']), config['gradient_accumulation_steps'], config['seed'], epoch)
            for indices in batches:
                torch.cuda.synchronize(); start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                losses, input_tokens, supervised_tokens = [], 0, 0
                for index in indices:
                    row = encoded['train'][index]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = model(**batch(row)); loss = result.loss / len(indices)
                    value = result.loss.detach().item()
                    if not math.isfinite(value):raise RuntimeError('non-finite training loss')
                    losses.append(value); loss.backward()
                    input_tokens += len(row['input_ids']); supervised_tokens += sum(t != -100 for t in row['labels'])
                    del result, loss
                norm = torch.nn.utils.clip_grad_norm_(params, config['max_grad_norm'], error_if_nonfinite=True)
                if norm.item() == 0 or any(p.grad is not None for _,p in frozen):
                    raise RuntimeError('invalid adapter gradient or frozen base received gradients')
                optimizer.step(); torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                step = {'step': len(report['optimizer_steps']) + 1, 'epoch': epoch + 1, 'examples': len(indices),
                        'mean_example_completion_loss': math.fsum(losses) / len(losses), 'grad_norm': norm.item(),
                        'seconds': elapsed, 'input_tokens_per_second': input_tokens / elapsed,
                        'supervised_tokens_per_second': supervised_tokens / elapsed, 'memory': memory()}
                report['optimizer_steps'].append(step); save()
                print(json.dumps(step), flush=True)
            optimizer.zero_grad(set_to_none=True)
            report['validation_losses'].append(validation_loss(epoch + 1)); save()
        if torch.equal(before, tracked.detach().cpu()):raise RuntimeError('adapter did not update')
        # Persist only the candidate adapter. Never merge into the frozen base.
        model.peft_config['default'].base_model_name_or_path = config['model']
        model.peft_config['default'].revision = config['model_revision']
        model.save_pretrained(output / 'adapter', safe_serialization=True, save_embedding_layers=False)
        torch.save(optimizer.state_dict(), output / 'optimizer_state.pt')
        report['adapter_hashes'] = {p.name: sha256_file(p) for p in sorted((output / 'adapter').iterdir()) if p.is_file()}
        report['optimizer_state_sha256'] = sha256_file(output / 'optimizer_state.pt')
        report.update(checkpoint_saved=True, adapter_updated=True, base_gradients_absent=True, status='completed')
    except Exception as exc:
        report['status'] = 'failed'; report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['memory'] = memory(); report['finished_at'] = datetime.now(timezone.utc).isoformat(); save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'model-manifest', 'config', 'output'):parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    r = train_capability(args.bundle, args.model_manifest, args.config, args.output)
    print(json_text({'status': r['status'], 'steps': len(r['optimizer_steps']), 'checkpoint_saved': r['checkpoint_saved']}))


if __name__ == '__main__':main()
