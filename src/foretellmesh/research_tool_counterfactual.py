"""Counterfactual information-state curriculum, separate from frozen probes.

Replay cases and answer annotations are never passed to the model. A case's
disclosure, scope, language, role and output-name variants share one event group.
The frozen diagnostic supplies only numerical exclusions, never training rows.
"""
import argparse
from collections import Counter
from datetime import timedelta
from itertools import product
import json
from pathlib import Path
import random
import tempfile

from .agent_baseline_data import input_context
from .agent_runtime import agent_instruction, validate_agent_output
from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .research_tool_data import PARTITIONS, read_research_tool_data, render_task, select_validation, validate_rows
from .schema import ValidationError, fields, iso, probability, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .uncertainty_diagnostic import load_uncertainty_config, read_uncertainty_cohort

VERSION = "synthetic_research_tool_v3"
PROTOCOL = "grounded_json_v3"
CONDITIONS = {
    "bayes": ("complete", "omit_prior", "omit_sensitivity", "omit_both"),
    "mixture": ("complete", "omit_weight", "omit_high", "omit_both"),
    "complement": ("hit_only", "miss_only", "both", "interval_only"),
    "sampling": ("counts_only", "population_given", "omit_successes", "omit_trials"),
}
MEANINGS = {
    "prior": ("event prior probability", "事件先验概率"),
    "sensitivity": ("positive-signal probability given the event", "事件为真时的阳性信号概率"),
    "false_positive_rate": ("positive-signal probability given no event", "事件为假时的阳性信号概率"),
    "posterior_probability": ("event probability after the positive signal", "阳性信号后的事件后验概率"),
    "high_rate": ("success probability in the high scenario", "高情景成功率"),
    "low_rate": ("success probability in the low scenario", "低情景成功率"),
    "weight": ("probability of selecting the high scenario", "选择高情景的概率"),
    "success_probability": ("unconditional success probability", "无条件成功概率"),
    "hit_probability": ("hit probability", "命中概率"),
    "miss_probability": ("miss probability", "未命中概率"),
    "successes": ("observed success count", "已观察到的成功次数"),
    "trials": ("observed trial count", "已观察到的试验次数"),
    "empirical_frequency": ("observed successes divided by trials", "观察样本的成功次数除以试验次数"),
    "population_probability": ("true Bernoulli population probability", "伯努利总体的真实成功概率"),
    "future_outcome": ("unobserved realized event or next-trial outcome", "尚未观察到的事件或下一次试验的实际结果"),
}
RULES = {
    "bayes": ("A positive signal is observed. posterior_probability = prior*sensitivity / (prior*sensitivity + (1-prior)*false_positive_rate).",
              "已观察到阳性信号。posterior_probability = prior*sensitivity / (prior*sensitivity + (1-prior)*false_positive_rate)。"),
    "mixture": ("Exactly one scenario is selected. success_probability = weight*high_rate + (1-weight)*low_rate. The two rates differ.",
                "恰好选择一个情景。success_probability = weight*high_rate + (1-weight)*low_rate。两个情景的成功率不同。"),
    "complement": ("Hit and miss are mutually exclusive and exhaustive: hit_probability + miss_probability = 1. Bounds specify an interval, not an exact probability.",
                   "命中与未命中互斥且穷尽：hit_probability + miss_probability = 1。区间约束不是精确概率。"),
    "sampling": ("Finite IID Bernoulli observations: empirical_frequency = successes/trials. No population prior or exact-identification assumption is supplied. Sample frequency does not identify population_probability.",
                 "有限次独立同分布伯努利观察：empirical_frequency = successes/trials。未提供总体先验或精确识别假设。样本频率不能确定 population_probability。"),
}


def load_counterfactual_config(path: Path) -> dict:
    c = fields(strict_json(path.read_text()), {"schema_version", "dataset_version", "output_protocol", "seed",
        "groups_per_family", "observation_times", "languages", "roles", "scopes", "formats",
        "evaluation_selection", "synthetic_only"}, "counterfactual config")
    fixed = {"schema_version": "1", "dataset_version": VERSION, "output_protocol": PROTOCOL,
             "languages": ["en", "zh"], "roles": ["research", "risk"],
             "scopes": ["calculation", "forecast"], "formats": ["field_names", "aliases"],
             "evaluation_selection": "unchanged_legacy_v1_role_cohort", "synthetic_only": True}
    if any(type(c[k]) is not type(v) or c[k] != v for k, v in fixed.items()):
        raise ValidationError("unsupported counterfactual policy")
    if type(c["seed"]) is not int or not 0 <= c["seed"] < 2**32:
        raise ValidationError("invalid counterfactual seed")
    fields(c["groups_per_family"], set(PARTITIONS), "counterfactual counts")
    fields(c["observation_times"], set(PARTITIONS), "counterfactual times")
    if any(type(n) is not int or not 1 <= n <= 100 for n in c["groups_per_family"].values()):
        raise ValidationError("invalid counterfactual group count")
    times = [timestamp(c["observation_times"][p], p) for p in PARTITIONS]
    if not times[0] < times[1] < times[2]:
        raise ValidationError("counterfactual observation times must increase")
    return c


def validate_parameters(family: str, p: dict) -> None:
    keys = {"bayes": {"prior", "sensitivity", "false_positive_rate"},
            "mixture": {"high_rate", "low_rate", "weight"},
            "complement": {"hit_probability", "lower", "upper"},
            "sampling": {"successes", "trials", "population_probability"}}
    if family not in keys:
        raise ValidationError("unknown counterfactual family")
    fields(p, keys[family], "counterfactual parameters")
    for key in keys[family] - {"successes", "trials"}:
        if not 0 < probability(p[key]) < 1:
            raise ValidationError("counterfactual rates must be interior")
    if family == "mixture" and not p["low_rate"] < p["high_rate"]:
        raise ValidationError("counterfactual scenario rates must differ")
    if family == "complement" and not p["lower"] < p["hit_probability"] < p["upper"]:
        raise ValidationError("invalid probability interval")
    if family == "sampling" and (type(p["successes"]) is not int or type(p["trials"]) is not int
                                 or not 0 < p["successes"] < p["trials"] <= 10000):
        raise ValidationError("invalid finite sample")


def information_state(family: str, p: dict, condition: str) -> tuple[dict, dict, set[str], set[str]]:
    """Identify exact quantities; supplied intervals never become exact values.

    Interior, nondegenerate parameters make these dependency rules sufficient.
    No numerical posterior, predicted outcome or imputed input is required.
    """
    validate_parameters(family, p)
    if condition not in CONDITIONS[family]:
        raise ValidationError("unknown disclosure condition")
    bounds = {}
    if family == "bayes":
        given = dict(p)
        if condition in ("omit_prior", "omit_both"):given.pop("prior")
        if condition in ("omit_sensitivity", "omit_both"):given.pop("sensitivity")
        universe = set(p) | {"posterior_probability"}
        known = set(given)
        if set(p) <= known:known.add("posterior_probability")
    elif family == "mixture":
        given = dict(p)
        if condition in ("omit_weight", "omit_both"):given.pop("weight")
        if condition in ("omit_high", "omit_both"):given.pop("high_rate")
        universe = set(p) | {"success_probability"}
        known = set(given)
        if set(p) <= known:known.add("success_probability")
    elif family == "complement":
        given = {}
        if condition in ("hit_only", "both"):given["hit_probability"] = p["hit_probability"]
        if condition in ("miss_only", "both"):given["miss_probability"] = round(1 - p["hit_probability"], 8)
        if condition == "interval_only":bounds["hit_probability"] = [p["lower"], p["upper"]]
        universe = {"hit_probability", "miss_probability"}
        known = set(universe) if given else set()
    else:
        given = {"successes": p["successes"], "trials": p["trials"]}
        if condition == "population_given":given["population_probability"] = p["population_probability"]
        if condition == "omit_successes":given.pop("successes")
        if condition == "omit_trials":given.pop("trials")
        universe = set(p) | {"empirical_frequency"}
        known = set(given)
        if {"successes", "trials"} <= known:known.add("empirical_frequency")
    return given, bounds, universe | {"future_outcome"}, known


def visible_signatures(family: str, p: dict) -> set[str]:
    # Ignore dates, field aliases and the hidden values of omitted inputs. Two
    # groups with the same visible numerical case cannot hide behind new IDs.
    return {canonical_hash({"family": family, "given": given, "bounds": bounds})
            for given, bounds, _, _ in (information_state(family, p, c) for c in CONDITIONS[family])}


def render_counterfactual(case: dict, partition: str) -> dict:
    fields(case, {"kind", "family", "parameters", "condition", "scope", "format", "language", "role", "observation_time"},
           "counterfactual case")
    if (case["kind"] != "information_state" or partition not in PARTITIONS or case["language"] not in ("en", "zh")
            or case["role"] not in ("research", "risk") or case["scope"] not in ("calculation", "forecast")
            or case["format"] not in ("field_names", "aliases")):
        raise ValidationError("invalid counterfactual case")
    family, p = case["family"], case["parameters"]
    given, bounds, universe, known = information_state(family, p, case["condition"])
    group = "information:" + canonical_hash({"family": family, "parameters": p})[:24]
    requested = universe - {"future_outcome"} if case["scope"] == "calculation" else universe
    if family == "sampling" and case["scope"] == "calculation":requested -= {"population_probability"}
    language = case["language"]
    names = sorted(requested)
    # The bijection changes with the event and scope, independently of which
    # inputs are hidden; paired disclosures retain exactly the same question.
    random.Random(canonical_hash({"group": group, "scope": case["scope"]})).shuffle(names)
    mapping = {name: (name if case["format"] == "field_names" else f"item_{i + 1}") for i, name in enumerate(names)}
    definitions = [{"name": mapping[name], "quantity": name, "meaning": MEANINGS[name][language == "zh"]} for name in names]
    question = ("合成信息审核。只检查下面列出的量是否能由证据精确确定。按角色的 JSON 格式回答；unknowns 只填写无法确定的量对应的 name 字符串，顺序不限，没有则返回 []。不要输出计算数值或额外字段。审核范围及输出名称："
                if language == "zh" else
                "Synthetic information audit. Check only whether the listed quantities are exactly determined by the evidence. Return your role JSON; unknowns must contain only the corresponding name strings for undetermined quantities, in any order, or [] when none. Do not output calculations or extra fields. Scope and output names: ")
    question += json.dumps(definitions, ensure_ascii=False, separators=(",", ":"))
    text = RULES[family][language == "zh"]
    text += (" 未列出的输入没有被提供，也无其他信息可用于恢复。事件或下一次试验的实际结果尚未观察。明确给定："
             if language == "zh" else
             " Unlisted inputs are not supplied; no other information recovers them. The event or next-trial outcome has not been observed. Explicit inputs: ")
    text += json.dumps({"values": given, "bounds": bounds}, sort_keys=True, separators=(",", ":"))
    observed = timestamp(case["observation_time"], "counterfactual observation")
    eid = "e" + canonical_hash({"group": group, "condition": case["condition"]})[:8]
    payload = {"question": question, "observation_time": iso(observed), "market": None, "evidence": [
        {"evidence_id": eid, "text": text, "source": "synthetic://information-state-specification",
         "published_at": iso(observed - timedelta(hours=2)), "available_at": iso(observed - timedelta(hours=1))}]}
    target = {"unknowns": sorted(mapping[name] for name in requested - known), "observation_time": iso(observed)}
    if case["role"] == "research":target.update(evidence_ids=[eid], counter_evidence_ids=[])
    else:target["risks"] = []
    validate_agent_output(case["role"], target, input_context(payload))
    suffix = ":".join(case[k] for k in ("condition", "scope", "format", "language", "role"))
    return {"sample_id": f"{group}:{suffix}", "event_group_id": group, "family": "information_" + family,
            "language": language, "task": case["role"] + "_information", "partition": partition,
            "case": case, "case_sha256": canonical_hash(case),
            "request": {"agent": case["role"], "adapter": None, "instruction": agent_instruction(case["role"], PROTOCOL),
                        "input": payload, "upstream": {}}, "target": target,
            "oracle": {"known_fields": sorted(requested & known), "unknown_fields": sorted(requested - known),
                       "output_names": mapping, "forbidden_evidence_ids": []}}


def sample_parameters(rng: random.Random, family: str) -> dict:
    def rate():return rng.randint(100, 9900) / 10000
    if family == "bayes":return {key: rate() for key in ("prior", "sensitivity", "false_positive_rate")}
    if family == "mixture":
        low, high = sorted(rng.sample(range(100, 9901), 2))
        return {"high_rate": high / 10000, "low_rate": low / 10000, "weight": rate()}
    if family == "complement":
        low, hit, high = sorted(rng.sample(range(100, 9901), 3))
        return {"hit_probability": hit / 10000, "lower": low / 10000, "upper": high / 10000}
    n = rng.randint(20, 9000)
    return {"successes": rng.randint(1, n - 1), "trials": n, "population_probability": rate()}


def exclusion_signatures(probe: dict, legacy: dict) -> set[str]:
    p = probe["complement"]["hit_probability"]
    cases = [("bayes", probe["bayes"]), ("sampling", probe["sampling"]),
             ("complement", {"hit_probability": p, "lower": p / 2, "upper": (p + 1) / 2}),
             ("mixture", {"high_rate": probe["mixture"]["high"], "low_rate": probe["mixture"]["low"], "weight": probe["mixture"]["weight"]})]
    for rows in legacy.values():
        for row in rows:
            case = row["case"]
            if case["kind"] != "tool":continue
            source = case["source_case"]; q = source["parameters"]
            if source["family"] == "mixture":
                cases.append(("mixture", {"high_rate": q["high"] / 100, "low_rate": q["low"] / 100, "weight": q["weight"] / 100}))
            else:
                cases.append(("bayes", {"prior": q["prior"] / 100, "sensitivity": q["sensitivity"] / 100, "false_positive_rate": q["false_positive"] / 100}))
    return set().union(*(visible_signatures(family, values) for family, values in cases))


def generate_parts(config: dict, legacy: dict, probe: dict) -> dict:
    # Revalidate original time/group/variant constraints before inheriting rows.
    validate_rows(legacy, {"observation_times": config["observation_times"]})
    blocked = exclusion_signatures(probe, legacy)
    rng = random.Random(config["seed"])
    parts = {p: [] for p in PARTITIONS}
    for partition in PARTITIONS:
        for old in legacy[partition]:
            row = render_task(old["case"], partition)
            row["case"] = {"kind": "legacy", "source_case": old["case"]}
            row["case_sha256"] = canonical_hash(row["case"])
            row["request"]["instruction"] = agent_instruction(row["request"]["agent"], PROTOCOL)
            parts[partition].append(row)
        for family in CONDITIONS:
            for _ in range(config["groups_per_family"][partition]):
                for attempt in range(10000):
                    params = sample_parameters(rng, family)
                    signatures = visible_signatures(family, params)
                    if not signatures & blocked:break
                else:raise ValidationError("cannot find disjoint counterfactual parameters")
                blocked.update(signatures)
                for condition, scope, fmt, language, role in product(CONDITIONS[family], config["scopes"],
                        config["formats"], config["languages"], config["roles"]):
                    case = {"kind": "information_state", "family": family, "parameters": params,
                            "condition": condition, "scope": scope, "format": fmt, "language": language,
                            "role": role, "observation_time": config["observation_times"][partition]}
                    parts[partition].append(render_counterfactual(case, partition))
        parts[partition].sort(key=lambda row: row["sample_id"])
    return parts


def statistics(parts: dict) -> dict:
    result = {}
    for partition, rows in parts.items():
        new = [r for r in rows if r["case"]["kind"] == "information_state"]
        result[partition] = {"examples": len(rows), "event_groups": len({r["event_group_id"] for r in rows}),
            "tasks": dict(Counter(r["task"] for r in rows)), "new_examples": len(new),
            "new_event_groups": len({r["event_group_id"] for r in new}),
            "empty_unknown_targets": sum(not r["target"]["unknowns"] for r in new),
            "by_condition": dict(Counter(r["family"] + ":" + r["case"]["condition"] for r in new)),
            "by_scope_format": dict(Counter(r["case"]["scope"] + ":" + r["case"]["format"] for r in new))}
    return result


def build_counterfactual_data(legacy_bundle: Path, diagnostic: Path, config_path: Path, output: Path) -> dict:
    if output.exists():raise ValidationError("counterfactual output already exists")
    config = load_counterfactual_config(config_path)
    manifest, legacy, _ = read_research_tool_data(legacy_bundle)
    if manifest["dataset_version"] != "synthetic_research_tool_v1":
        raise ValidationError("counterfactual curriculum requires the original v1 bundle")
    read_uncertainty_cohort(diagnostic)
    probe = load_uncertainty_config(diagnostic / "config.json")
    parts = generate_parts(config, legacy, probe)
    selected = select_validation(legacy["validation"])
    artifacts = {p + ".jsonl": jsonl(rows) for p, rows in parts.items()}
    artifacts.update({"config.json": config_path.read_text(), "excluded_diagnostic_config.json": json_text(probe),
        "legacy_cases.json": json_text({p: [r["case"] for r in rows] for p, rows in legacy.items()}),
        "evaluation_ids.json": json_text(selected)})
    report = {"schema_version": "1", "kind": "synthetic_research_tool_bundle", "dataset_version": VERSION,
        "config_sha256": sha256_file(config_path), "split_version": manifest["split_version"] + ":counterfactual_v3",
        "legacy_manifest_sha256": sha256_file(legacy_bundle / "manifest.json"),
        "excluded_diagnostic_manifest_sha256": sha256_file(diagnostic / "manifest.json"),
        "artifact_hashes": {k: sha256_bytes(v.encode()) for k, v in artifacts.items()}, "counts": statistics(parts),
        "validation_evaluation_examples": len(selected), "code": code_provenance(),
        "limitations": ["Synthetic information-state curriculum; no evidence of historical forecasting improvement.",
            "The frozen diagnostic informs the development direction and numerical exclusions, not training targets; it is not a final test.",
            "All 64 variants of each new numerical scenario share one group; row counts are not independent event counts.",
            "Exact and partially specified interior-probability models only; this is not a general symbolic solver.",
            "Legacy 24-row evaluation selection is unchanged and does not measure the new curriculum; separate held-out generation is required.",
            "Final-test rows are integrity-checked only, never tokenized or used for selection."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".counterfactual-", dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"; stage.mkdir()
        for name, content in artifacts.items():(stage / name).write_text(content)
        (stage / "manifest.json").write_text(json_text(report)); stage.rename(output)
    return report


def read_counterfactual_data(path: Path) -> tuple[dict, dict, dict]:
    manifest = strict_json((path / "manifest.json").read_text())
    names = {p + ".jsonl" for p in PARTITIONS} | {"config.json", "excluded_diagnostic_config.json", "legacy_cases.json", "evaluation_ids.json"}
    if (manifest.get("schema_version") != "1" or manifest.get("kind") != "synthetic_research_tool_bundle"
            or manifest.get("dataset_version") != VERSION or set(manifest["artifact_hashes"]) != names):
        raise ValidationError("invalid counterfactual bundle")
    for name, digest in manifest["artifact_hashes"].items():
        if sha256_file(path / name) != digest:raise ValidationError("counterfactual artifact hash mismatch")
    if manifest["config_sha256"] != sha256_file(path / "config.json"):
        raise ValidationError("counterfactual config hash mismatch")
    config = load_counterfactual_config(path / "config.json")
    probe = load_uncertainty_config(path / "excluded_diagnostic_config.json")
    cases = fields(strict_json((path / "legacy_cases.json").read_text()), set(PARTITIONS), "legacy partitions")
    legacy = {p: [render_task(case, p) for case in rows] for p, rows in cases.items()}
    parts = generate_parts(config, legacy, probe)
    for partition in PARTITIONS:
        actual = [strict_json(line) for line in (path / (partition + ".jsonl")).read_text().splitlines()]
        if actual != parts[partition]:raise ValidationError("counterfactual deterministic replay mismatch")
    selected = strict_json((path / "evaluation_ids.json").read_text())
    if (selected != select_validation(legacy["validation"]) or len(selected) != manifest["validation_evaluation_examples"]
            or statistics(parts) != manifest["counts"]):
        raise ValidationError("counterfactual counts or evaluation selection mismatch")
    return manifest, parts, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy-bundle", "diagnostic-cohort", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    report = build_counterfactual_data(a.legacy_bundle, a.diagnostic_cohort, a.config, a.output)
    print(json_text({"dataset_version": report["dataset_version"], "counts": report["counts"]}))


if __name__ == "__main__":main()
