"""Run and score a frozen synthetic development cohort with one local Base."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics
import sys
import time

from .agent_baseline_data import input_context, load_baseline_config, read_cohort
from .agent_runtime import AgentRunner
from .capabilities import load_capabilities
from .data import sha256_file
from .evaluation import code_provenance, json_text
from .lora_probe import verify_model_manifest
from .metrics import score_predictions
from .peft_runtime import PeftTextBackend, SharedPeftExecutor
from .probability_tools import execute_probability_tool
from .schema import ValidationError
from .synthetic_sft import canonical_hash


def mean(values):
    return math.fsum(values) / len(values) if values else None


def tool_call_matches(actual: dict | None, expected: dict) -> bool:
    if actual is None:
        return False
    try:
        execute_probability_tool(actual)
    except (ValueError, TypeError):
        return False
    if actual["name"] != expected["name"]:
        return False
    if actual["name"] == "weighted_probability":
        def pairs(call):
            args = call["arguments"]
            return sorted(zip(args["probabilities"], args["weights"]))
        return pairs(actual) == pairs(expected)
    return actual == expected


def summarize_baseline(cohort: dict, rows: list[dict]) -> dict:
    config = cohort["evaluation_config.json"]
    judges = {r["sample_id"]: r for r in cohort["judge.jsonl"]}
    tool_judges = {r["sample_id"]: r for r in cohort["tool_judge.jsonl"]}
    by_arm = defaultdict(dict)
    allowed = set(config["workflows"]) | {"calculate"}
    for row in rows:
        arm, sid = row["workflow"], row["sample_id"]
        if arm not in allowed or sid not in (tool_judges if arm == "calculate" else judges) or sid in by_arm[arm]:
            raise ValidationError("unknown or duplicate baseline result")
        by_arm[arm][sid] = row
    expected = {(arm, sid) for arm in config["workflows"] for sid in judges} | {("calculate", sid) for sid in tool_judges}
    if {(arm, sid) for arm, cases in by_arm.items() for sid in cases} != expected:
        raise ValidationError("baseline incomplete; refusing a final metric report")
    report = {"counts": {"forecast_examples": len(judges), "event_groups": len({j["event_group_id"] for j in judges.values()}),
                         "tool_examples": len(tool_judges), "scored_workflow_examples": len(rows)}, "arms": {}}
    def score(ps, ys):
        return score_predictions(ps, ys, ece_bins=config["ece_bins"], log_loss_epsilon=config["log_loss_epsilon"])
    def probabilities(arm, key="prediction"):
        result = {}
        for sid in judges:
            row = by_arm[arm][sid]["result"]
            prediction = row.get("prediction") if key == "prediction" else row.get("stages", {}).get("forecast")
            result[sid] = None if prediction is None else prediction["probability"]
        return result
    def resources(arm):
        members = list(by_arm[arm].values())
        attempts = [t for row in members for t in row["result"]["trace"]]
        calls = [c for row in members for c in row["calls"]]
        completed = [row for row in members if row["result"]["status"] == "completed"]
        tokens = [c["usage"] for c in calls if c.get("usage") is not None]
        generator_seconds = math.fsum(t["seconds"] for t in tokens)
        result = {"completed_examples": len(completed), "completion_rate": len(completed) / len(members),
                "first_pass_complete_examples": sum(all(t["attempt"] == 0 for t in row["result"]["trace"]) for row in completed),
                "model_calls": len(calls), "mean_model_calls": len(calls) / len(members),
                "schema_valid_calls": sum(t["status"] == "valid" for t in attempts),
                "schema_valid_call_rate": sum(t["status"] == "valid" for t in attempts) / len(attempts) if attempts else None,
                "repair_calls": sum(t["attempt"] > 0 for t in attempts),
                "failure_counts": dict(Counter((row["result"].get("stage", "") + ":" + row["result"].get("error", "")) for row in members if row["result"]["status"] != "completed")),
                "input_tokens": sum(t["input_tokens"] for t in tokens), "output_tokens": sum(t["output_tokens"] for t in tokens),
                "calls_reaching_output_token_limit": sum(t["output_reached_token_limit"] for t in tokens),
                "generation_seconds": generator_seconds,
                "output_tokens_per_generation_second": sum(t["output_tokens"] for t in tokens) / generator_seconds if generator_seconds else None,
                "total_seconds": math.fsum(row["seconds"] for row in members),
                "mean_seconds": mean([row["seconds"] for row in members]),
                "median_seconds": statistics.median(row["seconds"] for row in members),
                "peak_allocated_bytes": max((row.get("memory", {}).get("peak_allocated_bytes", 0) for row in members), default=0)}
        if "response_transport" in config:
            result["valid_calls_requiring_fence_decoding"] = sum(t.get("decoded_transport") == "json_fence" for t in attempts)
            result["valid_raw_json_calls"] = sum(t.get("decoded_transport") == "raw_json" for t in attempts)
        return result
    for arm in config["workflows"]:
        ps = probabilities(arm)
        ids, covered = list(judges), [sid for sid in judges if ps[sid] is not None]
        ys = [judges[sid]["outcome"] for sid in ids]
        gaps = [abs(ps[sid] - judges[sid]["oracle_probability"]) for sid in covered]
        references = []
        for sid in covered:
            p = by_arm[arm][sid]["result"]["prediction"]
            cited = set(p["key_evidence"]) | set(p["counter_evidence"])
            required = set(judges[sid]["required_evidence_ids"])
            references.append(len(cited & required) / len(required))
        stats = {"resources_and_format": resources(arm),
                 "simulated_outcome_scores": score([ps[sid] for sid in ids], ys),
                 "constant_0_5_same_covered_rows": score([.5 for _ in covered], [judges[sid]["outcome"] for sid in covered]),
                 "oracle_same_covered_rows": score([judges[sid]["oracle_probability"] for sid in covered], [judges[sid]["outcome"] for sid in covered]),
                 "oracle_probability_mae": mean(gaps), "oracle_probability_mse": mean([d*d for d in gaps]),
                 "within_oracle_tolerance": sum(d <= config["probability_tolerance"] for d in gaps),
                 "within_oracle_tolerance_fraction_all_examples": sum(d <= config["probability_tolerance"] for d in gaps) / len(ids),
                 "required_evidence_recall_covered": mean(references),
                 "failure_penalized_brier_diagnostic": mean([1 if ps[sid] is None else (ps[sid] - judges[sid]["outcome"])**2 for sid in ids]),
                 "covered_event_groups": len({judges[sid]["event_group_id"] for sid in covered}),
                 "by_language": {}, "by_family": {}}
        for field, target in (("language", "by_language"), ("family", "by_family")):
            for name in sorted({j[field] for j in judges.values()}):
                subset = [sid for sid in ids if judges[sid][field] == name]
                valid = [sid for sid in subset if ps[sid] is not None]
                stats[target][name] = {"examples": len(subset), "covered": len(valid),
                    "oracle_probability_mae": mean([abs(ps[sid]-judges[sid]["oracle_probability"]) for sid in valid]),
                    "simulated_scores": score([ps[sid] for sid in subset], [judges[sid]["outcome"] for sid in subset])}
        report["arms"][arm] = stats
    common = [sid for sid in judges if all(probabilities(a)[sid] is not None for a in config["workflows"])]
    report["common_coverage_comparison"] = {"count": len(common), "sample_ids": common,
        "scores": {arm: score([probabilities(arm)[sid] for sid in common], [judges[sid]["outcome"] for sid in common]) for arm in config["workflows"]}}
    reviewed = "reviewed_forecast"
    before, after = probabilities(reviewed, "before"), probabilities(reviewed)
    paired = [sid for sid in judges if before[sid] is not None and after[sid] is not None]
    report["critic_comparison"] = {"paired_count": len(paired), "forecast_available_final_missing": sum(before[sid] is not None and after[sid] is None for sid in judges),
        "before_scores_same_rows": score([before[sid] for sid in paired], [judges[sid]["outcome"] for sid in paired]),
        "after_scores_same_rows": score([after[sid] for sid in paired], [judges[sid]["outcome"] for sid in paired]),
        "probability_changes": sum(before[sid] != after[sid] for sid in paired)}
    tool_details = []
    for sid, judge in tool_judges.items():
        result = by_arm["calculate"][sid]["result"]
        actual = result.get("tool_result")
        call = result.get("stages", {}).get("quant")
        tool_details.append({"sample_id": sid, "valid": actual is not None,
                             "arguments_match": tool_call_matches(call, judge["expected_call"]),
                             "result_matches": actual is not None and abs(actual["result"] - judge["expected_result"]) <= config["probability_tolerance"]})
    report["tools"] = {"resources_and_format": resources("calculate"), "details": tool_details,
                       "arguments_match_count": sum(t["arguments_match"] for t in tool_details),
                       "result_match_count": sum(t["result_matches"] for t in tool_details)}
    return report


def run_agent_baseline(cohort_path: Path, model_manifest: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError("baseline output already exists")
    cohort_manifest, cohort = read_cohort(cohort_path)
    config, _ = load_baseline_config(cohort_path / "evaluation_config.json")
    agents, _ = load_capabilities(cohort_path / "agent_config.json")
    model_path, model_hash = verify_model_manifest(model_manifest, {"model": agents["base_model"], "model_revision": agents["base_revision"]})
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValidationError("baseline requires CUDA/BF16")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "1", "kind": "synthetic_agent_behavior_baseline", "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat(), "config": config,
              "cohort_manifest_sha256": sha256_file(cohort_path / "manifest.json"), "cohort_counts": cohort_manifest["counts"],
              "model_manifest_sha256": model_hash, "base_model": agents["base_model"], "base_revision": agents["base_revision"],
              "code": code_provenance(), "packages": {p: version(p) for p in ("torch", "transformers", "peft")},
              "python_executable": sys.executable, "gpu": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
              "base_model_loads": 0, "trained_adapters": False, "dtype": "bfloat16", "quantization": None,
              "output_protocol": config.get("output_protocol", "baseline_v1"),
              "response_transport": config.get("response_transport", "strict_json"),
              "sampling": {"do_sample": False, "max_new_tokens": config["max_new_tokens"], "seed": config["seed"]},
              "limitations": cohort_manifest["limitations"] + [
                  "Per-call budgets match, total computation differs across workflows. Not an equal-total-compute causal comparison.",
                  "Brier/Log Loss/ECE here use simulated outcomes. Oracle error and coverage are the primary behavior diagnostics.",
                  "Evidence recall checks reference inclusion, not real-world factual groundedness.",
                  "No model training, real-market evaluation, or model promotion performed."]}
    rows = []
    def save():
        (output / "report.json").write_text(json_text(report))
    save()
    try:
        set_seed(config["seed"])
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                    dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
        report["base_model_loads"] = 1
        report["generation_config"] = model.generation_config.to_dict()
        runtime = SharedPeftExecutor(model)
        class RecordingBackend(PeftTextBackend):
            def __init__(self):
                super().__init__(runtime, tokenizer, max_context_tokens=config["max_context_tokens"], max_new_tokens=config["max_new_tokens"])
                self.calls = []
            def generate(self, request):
                entry = {"agent": request["agent"], "adapter": request["adapter"], "request": request,
                         "request_sha256": canonical_hash(request), "output": None, "usage": None, "error_type": None}
                try:
                    result = super().generate(request)
                    entry["output"] = result
                    return result
                except Exception as exc:
                    entry["error_type"] = type(exc).__name__
                    raise
                finally:
                    entry["usage"] = self.last_usage
                    self.calls.append(entry)
        backend = RecordingBackend()
        runner = AgentRunner(agents, backend, output_protocol=config.get("output_protocol", "baseline_v1"),
                             response_transport=config.get("response_transport", "strict_json"))
        jobs = []
        for index, row in enumerate(cohort["inputs.jsonl"]):
            arms = config["workflows"]
            rotation = index % len(arms)
            jobs.extend((row, arm) for arm in arms[rotation:] + arms[:rotation])
        jobs.extend((row, "calculate") for row in cohort["tool_inputs.jsonl"])
        report["planned_workflow_examples"] = len(jobs)
        save()
        with (output / "results.jsonl").open("x") as log:
            for index, (row, workflow) in enumerate(jobs):
                backend.calls = []
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); started = time.perf_counter()
                result = runner.run(input_context(row["input"]), workflow=workflow, mode="base")
                torch.cuda.synchronize(); elapsed = time.perf_counter() - started
                record = {"sample_id": row["sample_id"], "workflow": workflow, "input_sha256": canonical_hash(row["input"]),
                          "result": result, "calls": backend.calls, "seconds": elapsed,
                          "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}}
                log.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"); log.flush()
                rows.append(record)
                report["completed_workflow_examples"] = len(rows)
                report["completed_status_counts"] = dict(Counter(r["result"]["status"] for r in rows))
                save()
                print(json.dumps({"progress": f"{index+1}/{len(jobs)}", "sample_id": row["sample_id"], "workflow": workflow,
                                  "status": result["status"], "calls": len(backend.calls), "seconds": round(elapsed, 2)}), flush=True)
                if any(c["error_type"] == "OutOfMemoryError" for c in backend.calls):
                    raise RuntimeError("CUDA OOM; partial run preserved, no final scores")
        report["metrics"] = summarize_baseline(cohort, rows)
        report["results_sha256"] = sha256_file(output / "results.jsonl")
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_agent_baseline(args.cohort, args.model_manifest, args.output)
    print(json_text({"status": result["status"], "counts": result["metrics"]["counts"]}))


if __name__ == "__main__":
    main()
