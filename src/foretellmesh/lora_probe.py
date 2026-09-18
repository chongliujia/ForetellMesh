"""Synthetic BF16 LoRA resource probe; never produces a forecasting checkpoint.

The normal evaluation CLI stays CPU-only. Heavy dependencies are imported only
when this separate module runs a GPU probe. All model files must already be local.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .schema import ValidationError


PROMPT = (
    "Synthetic hardware test only. Observation time: 2024-01-01T00:00:00Z. "
    "A fair coin will be tossed after the observation time. Forecast heads. "
    "No outcome has been observed. Return the requested forecast as JSON.\n"
)
CONTEXT = "Synthetic context: the coin is fair; the future result is unknown.\n"
ANSWER = json.dumps({
    "event": "Synthetic fair coin lands heads", "probability": 0.5,
    "confidence": "high", "base_rate": 0.5,
    "key_evidence": ["The synthetic setup specifies a fair coin."],
    "counter_evidence": [], "unknowns": ["The future toss result."],
    "observation_time": "2024-01-01T00:00:00Z",
}, sort_keys=True)


def load_probe_config(path: Path) -> dict:
    config = strict_json(path.read_text(encoding="utf-8"))
    fixed = {
        "model": "Qwen/Qwen3-8B-Base", "base_dtype": "bfloat16",
        "adapter_dtype": "float32", "quantization": None, "attention": "sdpa",
        "gradient_checkpointing": True, "use_reentrant": False,
        "use_cache": False, "optimizer": "adamw", "optimizer_foreach": False,
        "micro_batch_size": 1,
    }
    integers = ("seed", "gradient_accumulation_steps", "warmup_optimizer_steps",
                "measured_optimizer_steps", "lora_r", "lora_alpha")
    numbers = ("learning_rate", "weight_decay", "max_grad_norm", "lora_dropout")
    expected = set(fixed) | set(integers) | set(numbers) | {
        "run_name", "model_revision", "sequence_lengths", "target_modules"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValidationError("probe config has missing or unknown fields")
    for name, value in fixed.items():
        if type(config[name]) is not type(value) or config[name] != value:
            raise ValidationError(f"probe requires {name}={value!r}")
    if not isinstance(config["run_name"], str) or not config["run_name"].strip():
        raise ValidationError("run_name must be nonempty")
    if not isinstance(config["model_revision"], str) or not re.fullmatch(
            r"[0-9a-f]{40}", config["model_revision"]):
        raise ValidationError("model_revision must be an immutable commit")
    for name in integers:
        if type(config[name]) is not int or config[name] < (0 if name == "seed" else 1):
            raise ValidationError(f"invalid {name}")
    for name in numbers:
        value = config[name]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValidationError(f"invalid {name}")
    if config["learning_rate"] == 0 or config["max_grad_norm"] == 0 or config["lora_dropout"] >= 1:
        raise ValidationError("invalid learning rate, gradient norm or dropout")
    lengths = config["sequence_lengths"]
    if (not isinstance(lengths, list) or not lengths
            or any(type(n) is not int or not 256 <= n <= 32768 for n in lengths)
            or len(lengths) != len(set(lengths))):
        raise ValidationError("sequence_lengths must be unique integers in [256, 32768]")
    modules = config["target_modules"]
    if (not isinstance(modules, list) or len(modules) != 7
            or any(not isinstance(m, str) for m in modules)
            or set(modules) != {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}):
        raise ValidationError("probe must target all seven configured projection types")
    return config


def build_probe_tokens(tokenizer, sequence_length: int) -> dict:
    """Fill the entire context; mask the prompt and supervise only the answer.

    Explicit text serialization avoids assuming a chat template on a Base model.
    Padding cannot make this memory probe artificially easier.
    """
    prefix = tokenizer.encode(PROMPT, add_special_tokens=False)
    context = tokenizer.encode(CONTEXT, add_special_tokens=False)
    suffix = tokenizer.encode("\nAnswer:\n", add_special_tokens=False)
    answer = tokenizer.encode(ANSWER, add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValidationError("tokenizer must define EOS")
    answer.append(tokenizer.eos_token_id)
    remaining = sequence_length - len(prefix) - len(suffix) - len(answer)
    if remaining < 0 or not context:
        raise ValidationError("sequence too short for synthetic prompt and completion")
    prompt = prefix + (context * (remaining // len(context) + 1))[:remaining] + suffix
    ids = prompt + answer
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt) + answer}


def verify_model_manifest(path: Path, config: dict) -> tuple[Path, str]:
    content = path.read_bytes()
    manifest = strict_json(content.decode("utf-8"))
    if manifest["model"] != config["model"] or manifest["revision"] != config["model_revision"]:
        raise ValidationError("model manifest does not match pinned config")
    root = Path(manifest["snapshot"])
    names = set()
    for item in manifest["files"]:
        name = item["name"]
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise ValidationError("invalid model manifest file name")
        names.add(name)
        file = root / name
        if file.stat().st_size != item["bytes"] or sha256_file(file) != item["sha256"]:
            raise ValidationError(f"model file failed checksum: {name}")
    required = {"config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"}
    if not required <= names:
        raise ValidationError("model manifest is missing required files")
    index = strict_json((root / "model.safetensors.index.json").read_text())
    if not set(index["weight_map"].values()) <= names:
        raise ValidationError("unverified model weight shards")
    return root, sha256_bytes(content)


def run_probe(config_path: Path, manifest_path: Path, sequence_length: int, output: Path) -> dict:
    config_bytes = config_path.read_bytes()
    config = load_probe_config(config_path)
    if sequence_length not in config["sequence_lengths"]:
        raise ValidationError("sequence length is not in the versioned config")
    if output.exists():
        raise ValidationError("output already exists; choose a new run directory")
    model_path, manifest_hash = verify_model_manifest(manifest_path, config)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValidationError("probe requires a CUDA GPU with BF16 support")
    torch.cuda.set_device(0)
    set_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def memory() -> dict:
        free, total = torch.cuda.mem_get_info()
        return {"allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "device_free_bytes": free, "device_total_bytes": total}

    report = {
        "schema_version": "1", "kind": "synthetic_lora_resource_probe",
        "run_name": f"{config['run_name']}_seq{sequence_length}", "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "sequence_length": sequence_length,
        "config_sha256": sha256_bytes(config_bytes), "model_manifest_sha256": manifest_hash,
        "code": code_provenance(), "python_executable": sys.executable,
        "packages": {name: version(name) for name in (
            "torch", "transformers", "peft", "accelerate", "huggingface-hub", "safetensors")},
        "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": True, "tf32": False, "cpu_offload": False,
        "initial_memory": memory(), "optimizer_steps": [],
        "checkpoint_saved": False, "forecasting_metrics": None,
        "limitations": "Synthetic repeated input; resource feasibility only. No forecasting quality evaluation.",
    }
    try:
        report["nvidia_smi"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        report["nvidia_smi"] = None
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_bytes(config_bytes)

    def save():
        (output / "report.json").write_text(json_text(report), encoding="utf-8")

    save()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        tokens = build_probe_tokens(tokenizer, sequence_length)
        (output / "synthetic_batch.json").write_text(json_text(tokens), encoding="utf-8")
        report["dataset"] = {
            "version": "synthetic_fair_coin_resource_probe_v1", "synthetic": True,
            "sha256": sha256_bytes(json_text(tokens).encode()), "split_version": "not_applicable_resource_probe",
            "unmasked_tokens_per_microstep": sum(x != -100 for x in tokens["labels"]),
        }
        print("Loading pinned BF16 base model on cuda:0", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation=config["attention"], local_files_only=True,
            trust_remote_code=False, use_safetensors=True,
        )
        model.config.use_cache = False
        lora = LoraConfig(
            r=config["lora_r"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], target_modules=config["target_modules"],
            task_type="CAUSAL_LM", bias="none", base_model_name_or_path=config["model"],
            revision=config["model_revision"],
        )
        model = get_peft_model(model, lora, autocast_adapter_dtype=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
        if not trainable or any("lora_" not in n or p.dtype != torch.float32 for n, p in trainable):
            raise RuntimeError("only FP32 LoRA adapters may be trainable")
        if any(p.dtype != torch.bfloat16 for _, p in frozen):
            raise RuntimeError("base parameters must remain BF16")
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise RuntimeError("all parameters must reside on GPU")
        report["parameters"] = {
            "trainable": sum(p.numel() for _, p in trainable),
            "frozen": sum(p.numel() for _, p in frozen),
            "trainable_dtypes": dict(Counter(str(p.dtype) for _, p in trainable)),
            "frozen_dtypes": dict(Counter(str(p.dtype) for _, p in frozen)),
        }
        report["loaded_memory"] = memory()
        tracked = next(p for n, p in trainable if "lora_B" in n)
        before = tracked.detach().cpu().clone()
        params = [p for _, p in trainable]
        optimizer = torch.optim.AdamW(params, lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"], foreach=False)
        report["optimizer_defaults"] = dict(optimizer.defaults)
        batch = {key: torch.tensor([value], dtype=torch.long, device="cuda") for key, value in tokens.items()}

        def eval_loss() -> float:
            model.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(**batch)
                loss = result.loss.item()
            if not math.isfinite(loss):
                raise RuntimeError("non-finite synthetic evaluation loss")
            return loss

        report["synthetic_loss_before"] = eval_loss()
        model.train()
        accumulation = config["gradient_accumulation_steps"]
        count = config["warmup_optimizer_steps"] + config["measured_optimizer_steps"]
        for step in range(count):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(accumulation):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(**batch)
                    loss = result.loss / accumulation
                losses.append(result.loss.detach().item())
                loss.backward()
                del result, loss
                if (micro + 1) % 4 == 0:
                    print(f"seq={sequence_length} optimizer={step+1}/{count} micro={micro+1}/{accumulation}", flush=True)
            norm = torch.nn.utils.clip_grad_norm_(params, config["max_grad_norm"], error_if_nonfinite=True)
            if not all(math.isfinite(value) for value in losses) or norm.item() == 0:
                raise RuntimeError("non-finite loss or zero adapter gradient")
            if any(p.grad is not None for _, p in frozen):
                raise RuntimeError("frozen base received gradients")
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            row = {"step": step + 1, "warmup": step < config["warmup_optimizer_steps"],
                   "seconds": seconds, "mean_loss": sum(losses) / len(losses),
                   "grad_norm": norm.item(), "input_tokens_per_second": sequence_length * accumulation / seconds,
                   "memory": memory()}
            report["optimizer_steps"].append(row)
            save()
            print(json.dumps(row), flush=True)
        optimizer.zero_grad(set_to_none=True)
        report["synthetic_loss_after"] = eval_loss()
        report["adapter_updated"] = not torch.equal(before, tracked.detach().cpu())
        if not report["adapter_updated"]:
            raise RuntimeError("adapter did not update")
        measured = [row for row in report["optimizer_steps"] if not row["warmup"]]
        elapsed = sum(row["seconds"] for row in measured)
        report["measured_input_tokens_per_second"] = sequence_length * accumulation * len(measured) / elapsed
        report["measured_supervised_tokens_per_second"] = (
            report["dataset"]["unmasked_tokens_per_microstep"] * accumulation * len(measured) / elapsed)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["final_memory"] = memory()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_probe(args.config, args.model_manifest, args.sequence_length, args.output)
    print(f"{report['status']}: {args.output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
