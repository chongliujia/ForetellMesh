"""Freeze a development-only synthetic cohort before any model responses exist."""
from collections import defaultdict
from pathlib import Path
import tempfile

from .agent_runtime import OUTPUT_PROTOCOLS, RESPONSE_TRANSPORTS, validated_input
from .capabilities import load_capabilities
from .config import load_config
from .data import load_records, sha256_bytes, sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .schema import ValidationError, fields, parse_record, probability
from .sft_data import jsonl, validate_target
from .splits import split_records
from .synthetic_sft import FAMILIES, SOURCE, VERSION, render_case


def load_baseline_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = strict_json(raw.decode())
    optional = {key for key in ("output_protocol", "response_transport") if isinstance(value, dict) and key in value}
    config = fields(value, {"schema_version", "run_name", "source_partition", "groups_per_family",
        "workflows", "tool_families", "max_context_tokens", "max_new_tokens", "seed", "probability_tolerance",
        "ece_bins", "log_loss_epsilon", "selection", "mode", "purpose"} | optional, "agent baseline config")
    protocol = config.get("output_protocol", "baseline_v1")
    if not isinstance(protocol, str) or protocol not in OUTPUT_PROTOCOLS:
        raise ValidationError("unknown output protocol")
    transport = config.get("response_transport", "strict_json")
    if not isinstance(transport, str) or transport not in RESPONSE_TRANSPORTS:
        raise ValidationError("unknown response transport")
    fixed = {"schema_version": "1", "source_partition": "validation", "mode": "base",
             "selection": "lexicographically_first_event_groups_per_family_both_languages",
             "purpose": "synthetic_validation_diagnostic_not_real_forecasting_benchmark"}
    if any(config[k] != v for k, v in fixed.items()):
        raise ValidationError("baseline must use the declared synthetic validation policy")
    if not isinstance(config["run_name"], str) or not config["run_name"].strip():
        raise ValidationError("baseline run name is required")
    if config["workflows"] != ["single_forecast", "research_forecast", "reviewed_forecast"] or config["tool_families"] != ["mixture", "bayes_signal"]:
        raise ValidationError("unsupported baseline arms/tool families")
    for key, low, high in (("groups_per_family", 1, 3), ("max_context_tokens", 512, 4096),
                           ("max_new_tokens", 64, 1024), ("ece_bins", 1, 20), ("seed", 0, 2**32 - 1)):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValidationError("invalid bounded baseline config")
    if config["max_context_tokens"] <= config["max_new_tokens"]:
        raise ValidationError("invalid token budget")
    if not 0 < probability(config["probability_tolerance"]) < .01 or not 0 < probability(config["log_loss_epsilon"]) < .5:
        raise ValidationError("invalid metric tolerance")
    return config, sha256_bytes(raw)


def input_context(payload: dict):
    fields(payload, {"question", "observation_time", "evidence", "market"}, "baseline model input")
    return validated_input(parse_record({**payload, "sample_id": "input", "dataset_source": "synthetic", "dataset_version": "1",
                          "event_id": "input", "event_group_id": "input", "label": None}).forecast_input)


def expected_tool(case: dict) -> dict:
    p = case["parameters"]
    if case["family"] == "bayes_signal":
        return {"name": "bayes_binary", "arguments": {"prior": p["prior"] / 100, "sensitivity": p["sensitivity"] / 100,
                                                     "false_positive_rate": p["false_positive"] / 100}}
    if case["family"] == "mixture":
        return {"name": "weighted_probability", "arguments": {"probabilities": [p["low"] / 100, p["high"] / 100],
                                                              "weights": [(100-p["weight"]) / 100, p["weight"] / 100]}}
    raise ValidationError("unsupported tool task")


def prepare_agent_baseline(raw_dataset: Path, split_config: Path, agent_config: Path,
                           evaluation_config: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError("baseline cohort output already exists")
    config, config_hash = load_baseline_config(evaluation_config)
    agents, agents_hash = load_capabilities(agent_config)
    splits, split_hash = load_config(split_config)
    raw_manifest_bytes = (raw_dataset / "manifest.json").read_bytes()
    raw_manifest = strict_json(raw_manifest_bytes.decode())
    records, records_hash = load_records(raw_dataset / "records.jsonl")
    target_bytes = (raw_dataset / "targets.jsonl").read_bytes()
    if (raw_manifest.get("kind") != "synthetic_schema_warmup" or raw_manifest.get("historical_data") is not False
            or records_hash != raw_manifest["records_sha256"] or sha256_bytes(target_bytes) != raw_manifest["targets_sha256"]
            or sha256_file(raw_dataset / "config.json") != raw_manifest["config_sha256"]):
        raise ValidationError("synthetic source archive hash/type mismatch")
    targets = {}
    for line in target_bytes.decode().splitlines():
        row = strict_json(line)
        if not isinstance(row, dict) or row.get("sample_id") in targets:
            raise ValidationError("invalid or duplicate synthetic target")
        targets[row["sample_id"]] = row
    if set(targets) != {r.sample_id for r in records}:
        raise ValidationError("source record/target identities differ")
    by_family = defaultdict(lambda: defaultdict(list))
    cases = {}
    validation = split_records(records, splits).partitions["validation"]
    for record in validation:
        if record.dataset_source != SOURCE or record.dataset_version != VERSION:
            raise ValidationError("only the fixed synthetic source is supported")
        row = targets[record.sample_id]
        validate_target(row, record, splits.sources[SOURCE], raw_dataset)
        case = strict_json((raw_dataset / row["provenance"]["artifact"]).read_text())
        reference, target = render_case(case)
        if record != parse_record(reference):
            raise ValidationError("synthetic input/label differs from deterministic oracle replay")
        cases[record.sample_id] = (case, target)
        by_family[case["family"]][record.event_group_id].append(record)
    if set(by_family) != set(FAMILIES):
        raise ValidationError("validation cohort does not cover all declared families")
    selected = []
    for family in FAMILIES:
        groups = sorted(by_family[family])
        if len(groups) < config["groups_per_family"]:
            raise ValidationError("insufficient validation event groups")
        for group in groups[:config["groups_per_family"]]:
            members = by_family[family][group]
            if len(members) != 2 or {cases[r.sample_id][0]["language"] for r in members} != {"en", "zh"}:
                raise ValidationError("selected groups require both language variants")
            selected.extend(members)
    selected.sort(key=lambda r: r.sample_id)
    inputs, judges, tool_inputs, tool_judges = [], [], [], []
    for record in selected:
        case, target = cases[record.sample_id]
        payload = record.forecast_input.to_payload()
        inputs.append({"sample_id": record.sample_id, "input": payload})
        judges.append({"sample_id": record.sample_id, "event_group_id": record.event_group_id,
                       "family": case["family"], "language": case["language"], "oracle_probability": target["probability"],
                       "outcome": record.label.outcome, "required_evidence_ids": target["key_evidence"] + target["counter_evidence"]})
        if case["family"] in config["tool_families"]:
            # Quant must construct the call from the scenario. A previously
            # computed answer is intentionally absent from this task's input.
            tool_payload = {**payload, "evidence": [e for e in payload["evidence"] if e["evidence_id"] == "setup"]}
            sid = record.sample_id + ":tool"
            tool_inputs.append({"sample_id": sid, "input": tool_payload})
            tool_judges.append({"sample_id": sid, "event_group_id": record.event_group_id, "family": case["family"],
                                "language": case["language"], "expected_call": expected_tool(case), "expected_result": target["probability"]})
    artifacts = {"inputs.jsonl": jsonl(inputs), "judge.jsonl": jsonl(judges),
                 "tool_inputs.jsonl": jsonl(tool_inputs), "tool_judge.jsonl": jsonl(tool_judges),
                 "agent_config.json": json_text(agents), "evaluation_config.json": json_text(config)}
    report = {"schema_version": "1", "kind": "synthetic_agent_validation_cohort", "config_sha256": config_hash,
              "agent_config_sha256": agents_hash, "split_config_sha256": split_hash,
              "source_manifest_sha256": sha256_bytes(raw_manifest_bytes), "source_records_sha256": records_hash,
              "source_targets_sha256": sha256_bytes(target_bytes), "source_partition": "validation",
              "counts": {"forecast_inputs": len(inputs), "independent_event_groups": len(selected) // 2,
                         "tool_inputs": len(tool_inputs), "families": len(FAMILIES)},
              "artifact_hashes": {k: sha256_bytes(v.encode()) for k, v in artifacts.items()}, "code": code_provenance(),
              "limitations": ["Development validation subset; no final-test selection or real-event performance claims.",
                              "Forecast inputs include oracle calculation evidence; measures reading/using supplied calculations, not unaided arithmetic.",
                              "Language variants share outcomes and are not independent events; no independent-event confidence intervals.",
                              "Tool tasks remove calculation evidence; expected calls and outcomes live only on the judge side."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".agent-cohort-", dir=output.parent) as temp:
        stage = Path(temp) / "cohort"; stage.mkdir()
        for name, contents in artifacts.items():
            (stage / name).write_text(contents)
        (stage / "manifest.json").write_text(json_text(report))
        stage.rename(output)
    return report


def read_cohort(path: Path) -> tuple[dict, dict]:
    report = strict_json((path / "manifest.json").read_text())
    if report.get("schema_version") != "1" or report.get("kind") != "synthetic_agent_validation_cohort" or report.get("source_partition") != "validation":
        raise ValidationError("expected a frozen synthetic validation cohort")
    required = {"inputs.jsonl", "judge.jsonl", "tool_inputs.jsonl", "tool_judge.jsonl", "agent_config.json", "evaluation_config.json"}
    if set(report["artifact_hashes"]) != required:
        raise ValidationError("cohort artifact set mismatch")
    contents = {}
    for name, expected in report["artifact_hashes"].items():
        raw = (path / name).read_bytes()
        if sha256_bytes(raw) != expected:
            raise ValidationError("cohort artifact hash mismatch")
        contents[name] = [strict_json(line) for line in raw.decode().splitlines()] if name.endswith("jsonl") else strict_json(raw.decode())
    for input_name, judge_name, count_name in (("inputs.jsonl", "judge.jsonl", "forecast_inputs"),
                                               ("tool_inputs.jsonl", "tool_judge.jsonl", "tool_inputs")):
        rows, judge = contents[input_name], contents[judge_name]
        ids = [r["sample_id"] for r in rows]
        if (len(ids) != len(set(ids)) or len(rows) != report["counts"][count_name]
                or ids != [r["sample_id"] for r in judge]):
            raise ValidationError("cohort row identity/count mismatch")
        for row in rows:
            fields(row, {"sample_id", "input"}, "cohort input row")
            input_context(row["input"])
    return report, contents
