"""Offline shared-base adapter save/load/switch probe; no forecasting training."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import statistics
import time

from .capabilities import load_capabilities
from .data import sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .lora_probe import verify_model_manifest
from .peft_runtime import SharedPeftExecutor
from .schema import ValidationError, fields, parse_record


def run_multilora_probe(config_path: Path, model_manifest: Path, output: Path,
                       base_agent_fixture: Path | None = None) -> dict:
    if output.exists():
        raise ValidationError("probe output already exists")
    config, config_hash = load_capabilities(config_path)
    model_path, model_hash = verify_model_manifest(model_manifest, {
        "model": config["base_model"], "model_revision": config["base_revision"]})
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValidationError("probe requires CUDA/BF16")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "1", "kind": "synthetic_multilora_switch_probe", "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat(), "config_sha256": config_hash,
              "model_manifest_sha256": model_hash, "base_model": config["base_model"], "base_revision": config["base_revision"],
              "code": code_provenance(), "packages": {p: version(p) for p in ("torch", "peft", "transformers")},
              "gpu": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
              "seed": 123, "dtype": "bfloat16", "quantization": None,
              "base_model_loads": 0, "adapter_artifacts": {}, "measurements": [],
              "trained_adapters": False, "forecasting_metrics": None,
              "limitations": "Random diagnostic adapter weights. Validates lifecycle, request isolation and resource use only; not predictive capability or vLLM serving."}
    def save():
        (output / "report.json").write_text(json_text(report))
    def memory():
        free, total = torch.cuda.mem_get_info()
        return {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "device_free_bytes": free, "device_total_bytes": total}
    save()
    try:
        torch.manual_seed(123)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                   dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
        report["base_model_loads"] += 1
        base.eval()
        base.requires_grad_(False)
        embedding_pointer = base.get_input_embeddings().weight.data_ptr()
        ids = tokenizer.encode("Synthetic adapter switching test. A fair coin has probability 0.5 of heads. Return a probability.", add_special_tokens=False)
        tokens = torch.tensor([ids], device="cuda")
        def last_logits(model):
            with torch.inference_mode():
                return model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits[0, -1].float().cpu()
        baseline = last_logits(base)
        lora = config["lora"]
        def adapter_config():
            return LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                              target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM",
                              base_model_name_or_path=config["base_model"], revision=config["base_revision"])
        names = list(config["capabilities"])
        if len(names) != 2:
            raise ValidationError("this bounded probe requires two capability adapters")
        model = get_peft_model(base, adapter_config(), adapter_name=names[0])
        model.add_adapter(names[1], adapter_config())
        # LoRA B is normally initialized at zero, making both adapters identical
        # to Base. Nonzero diagnostic weights make an accidental wrong route visible.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(mean=0, std=0.002)
        model.eval()
        model.requires_grad_(False)
        before = SharedPeftExecutor(model)
        references = {name: before.execute(name, last_logits) for name in names}
        model.save_pretrained(output / "diagnostic_adapters", selected_adapters=names,
                              safe_serialization=True, save_embedding_layers=False)
        for name in names:
            path = output / "diagnostic_adapters" / name
            report["adapter_artifacts"][name] = {p.name: sha256_file(p) for p in path.iterdir() if p.is_file()}
        # Unload without merging; retain the exact same base parameter storage.
        base = model.unload()
        del before, model
        model = PeftModel.from_pretrained(base, output / "diagnostic_adapters" / names[0],
                                          adapter_name=names[0], is_trainable=False, local_files_only=True)
        model.load_adapter(output / "diagnostic_adapters" / names[1], adapter_name=names[1],
                           is_trainable=False, local_files_only=True)
        runtime = SharedPeftExecutor(model)
        report["same_base_storage"] = embedding_pointer == model.get_input_embeddings().weight.data_ptr()
        if not report["same_base_storage"]:
            raise RuntimeError("base model storage changed during adapter reload")
        for adapter in (None, names[0], names[1], names[0], None):
            # Each pass includes adapter switching and a CPU-visible completion.
            torch.cuda.synchronize()
            started = time.perf_counter()
            logits = runtime.execute(adapter, last_logits)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            expected = baseline if adapter is None else references[adapter]
            if not torch.equal(expected, logits):
                raise RuntimeError("adapter reload/switch failed exact-logit consistency")
            report["measurements"].append({"adapter": adapter, "seconds": elapsed, "input_tokens": len(ids),
                                            "input_tokens_per_second": len(ids) / elapsed, "logits_equal_reference": True,
                                            "memory": memory()})
            save()
        report["adapters_produce_distinct_logits"] = not torch.equal(references[names[0]], references[names[1]])
        report["adapters_differ_from_base"] = all(not torch.equal(baseline, references[name]) for name in names)
        if not report["adapters_produce_distinct_logits"] or not report["adapters_differ_from_base"]:
            raise RuntimeError("diagnostic adapters were not distinguishable")
        try:
            runtime.execute(names[1], lambda _: (_ for _ in ()).throw(RuntimeError("probe operation failure")))
        except RuntimeError:
            pass
        report["exception_restores_active_adapter"] = model.active_adapter == names[0]
        report["all_parameters_frozen"] = all(not p.requires_grad for p in model.parameters())
        if not report["exception_restores_active_adapter"] or not report["all_parameters_frozen"]:
            raise RuntimeError("inference restoration/freeze invariant failed")
        report["median_forward_seconds"] = statistics.median(row["seconds"] for row in report["measurements"])
        if base_agent_fixture is not None:
            from .agent_runtime import AgentRunner
            from .peft_runtime import PeftTextBackend
            fixture = strict_json(base_agent_fixture.read_text())
            if fixture.get("kind") != "scripted_agent_test_only":
                raise ValidationError("base behavior check requires the synthetic fixture")
            payload = fields(fixture["input"], {"question", "observation_time", "evidence", "market"}, "behavior input")
            context = parse_record({**payload, "sample_id": "probe", "dataset_source": "synthetic",
                                    "dataset_version": "1", "event_id": "probe", "event_group_id": "probe", "label": None}).forecast_input
            generations = []
            class RecordingBackend(PeftTextBackend):
                def generate(self, request):
                    text = super().generate(request)
                    generations.append({"agent": request["agent"], "adapter": request["adapter"], "output": text})
                    return text
            backend = RecordingBackend(runtime, tokenizer, max_context_tokens=2048, max_new_tokens=384)
            report["base_behavior_fixture_sha256"] = sha256_file(base_agent_fixture)
            report["base_behavior_checks"] = {}
            for workflow in ("single_forecast", "research_forecast"):
                report["base_behavior_checks"][workflow] = AgentRunner(config, backend).run(context, workflow=workflow, mode="base")
                save()
            (output / "base_behavior_generations.json").write_text(json_text(generations))
            report["base_behavior_generations_sha256"] = sha256_file(output / "base_behavior_generations.json")
            # Agent schema failure is a measured result, not a failed adapter
            # switching test. It does not qualify this raw Base for production RL.
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["memory"] = memory()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/capability_agents_v1.json"))
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-agent-fixture", type=Path)
    args = parser.parse_args()
    result = run_multilora_probe(args.config, args.model_manifest, args.output, args.base_agent_fixture)
    print(json_text({"status": result["status"], "memory": result["memory"], "median_forward_seconds": result["median_forward_seconds"]}))


if __name__ == "__main__":
    main()
