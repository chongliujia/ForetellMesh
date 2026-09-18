"""Frozen, development-only probes of known quantities versus missing information.

The unknowns array uses a finite vocabulary to permit an exact, conservative
score. This tests a constrained classification contract, not free-text truth.
Neither these probes nor their targets are used for parameter training.
"""
import argparse
from datetime import timedelta
from pathlib import Path
import tempfile

from .agent_baseline_data import input_context
from .agent_runtime import validate_agent_output
from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .schema import ValidationError, fields, iso, probability, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash

VERSION = "synthetic_uncertainty_diagnostic_v1"
CONDITIONS = ("mixture_known", "mixture_missing_weight", "bayes_known", "bayes_missing_prior",
              "complement_parameters", "complement_future", "sampling_empirical", "sampling_population")


def load_uncertainty_config(path: Path) -> dict:
    c = fields(strict_json(path.read_text()), {"schema_version", "dataset_version", "partition", "observation_time",
        "languages", "roles", "mixture", "bayes", "complement", "sampling", "synthetic_only"}, "uncertainty config")
    if (c["schema_version"] != "1" or c["dataset_version"] != VERSION or c["partition"] != "validation"
            or c["languages"] != ["en", "zh"] or c["roles"] != ["research", "risk"] or c["synthetic_only"] is not True):
        raise ValidationError("only fixed synthetic development probes are supported")
    timestamp(c["observation_time"], "probe observation")
    for name, keys in (("mixture", {"high", "low", "weight"}), ("bayes", {"prior", "sensitivity", "false_positive_rate"}),
                       ("complement", {"hit_probability"}), ("sampling", {"successes", "trials", "population_probability"})):
        fields(c[name], keys, name)
        for key in keys - {"successes", "trials"}:
            if not 0 < probability(c[name][key]) < 1:raise ValidationError("probe probabilities must be interior")
    s = c["sampling"]
    if (type(s["successes"]) is not int or type(s["trials"]) is not int or not 0 < s["successes"] < s["trials"]
            or c["mixture"]["low"] >= c["mixture"]["high"]):
        raise ValidationError("invalid probe parameters")
    return c


def render_probe(c: dict, condition: str, language: str, role: str) -> tuple[dict, dict]:
    if condition not in CONDITIONS or language not in c["languages"] or role not in c["roles"]:
        raise ValidationError("unknown probe condition, language or role")
    family = condition.split("_")[0]
    p = c[family]
    group = "uncertainty:" + canonical_hash({"family": family, "parameters": p})[:24]
    sid = f"{group}:{condition}:{language}:{role}"
    zh = language == "zh"
    known = set()
    if family == "mixture":
        definitions = {"high_rate": "高情景的成功率" if zh else "success probability in the high scenario",
                       "low_rate": "低情景的成功率" if zh else "success probability in the low scenario",
                       "weight": "选择高情景的概率" if zh else "probability of selecting the high scenario",
                       "success_probability": "该模型下单次试验的无条件成功概率" if zh else "unconditional success probability of a trial under this model"}
        text = (f'合成两情景模型：high_rate={p["high"]}，low_rate={p["low"]}。只会选择一个情景。' if zh else
                f'Synthetic two-scenario model: high_rate={p["high"]}, low_rate={p["low"]}. Exactly one scenario is selected.')
        known.update(("high_rate", "low_rate"))
        if condition == "mixture_known":
            text += (f'选择高情景的概率 weight={p["weight"]}，否则选择低情景。' if zh else
                     f' The high scenario is selected with weight={p["weight"]}; otherwise the low scenario is selected.')
            known.update(("weight", "success_probability"))
        else:
            text += " 未提供情景选择机制或权重。" if zh else " No scenario-selection mechanism or weight is supplied."
    elif family == "bayes":
        definitions = {"prior": "事件的先验概率" if zh else "prior probability of the event",
                       "sensitivity": "事件为真时阳性信号的条件概率" if zh else "probability of a positive signal given the event",
                       "false_positive_rate": "事件为假时阳性信号的条件概率" if zh else "probability of a positive signal given no event",
                       "posterior_probability": "给定阳性信号后的事件后验概率" if zh else "posterior event probability given the positive signal"}
        text = (f'合成信号模型：sensitivity={p["sensitivity"]}，false_positive_rate={p["false_positive_rate"]}。已观察到阳性信号。' if zh else
                f'Synthetic signal model: sensitivity={p["sensitivity"]}, false_positive_rate={p["false_positive_rate"]}. A positive signal has been observed.')
        known.update(("sensitivity", "false_positive_rate"))
        if condition == "bayes_known":
            text += f' prior={p["prior"]}。' if zh else f' prior={p["prior"]}.'
            known.update(("prior", "posterior_probability"))
        else:
            text += " 未提供先验或基准率。" if zh else " No prior or base rate is supplied."
    elif family == "complement":
        definitions = {"hit_probability": "命中概率" if zh else "hit probability",
                       "miss_probability": "未命中概率" if zh else "miss probability"}
        text = (f'合成模型：hit_probability={p["hit_probability"]}。命中与未命中互斥且穷尽全部可能。' if zh else
                f'Synthetic model: hit_probability={p["hit_probability"]}. Hit and miss are mutually exclusive and exhaustive.')
        known.update(definitions)
    else:
        definitions = {"empirical_frequency": "样本成功频率" if zh else "observed sample success frequency",
                       "population_probability": "总体的真实成功概率" if zh else "true population success probability"}
        text = (f'合成独立同分布伯努利试验：{p["trials"]} 次历史抽样中成功 {p["successes"]} 次。' if zh else
                f'Synthetic independent identically distributed Bernoulli trials: {p["successes"]} successes in {p["trials"]} historical draws.')
        known.add("empirical_frequency")
        if condition == "sampling_population":
            text += (f' 模型另外明确给定真实总体概率 population_probability={p["population_probability"]}。' if zh else
                     f' Separately, the model explicitly specifies the true population_probability={p["population_probability"]}.')
            known.add("population_probability")
        else:
            text += " 未给定总体概率；没有无限样本或精确识别假设。" if zh else " The population probability is not specified; no infinite-sample or exact-identification assumption is supplied."
    if condition != "complement_parameters":
        definitions["future_outcome"] = "下一次试验或未来事件的实际结果" if zh else "realized outcome of the next trial or future event"
        text += " 下一次试验或事件尚未发生，其实际结果尚未观察。" if zh else " The next trial or event has not occurred and its realized outcome has not been observed."
    observed = timestamp(c["observation_time"], "probe observation")
    eid = "e" + canonical_hash({"group": group, "condition": condition})[:8]
    question = ("合成信息状态诊断。按你的角色输出通常的 JSON。unknowns 只能使用下面的字段名，列出无法从给定信息确定的字段；不得改写字段名。没有缺失字段时返回空数组。字段含义：" if zh else
                "Synthetic information-state diagnostic. Return your usual role JSON. In unknowns use only the exact field names below for quantities not determined by the supplied information; do not paraphrase the names. Use [] if no listed field is undetermined. Field meanings: ")
    question += "; ".join(f"{key}: {value}" for key, value in definitions.items())
    payload = {"question": question, "observation_time": iso(observed), "market": None,
               "evidence": [{"evidence_id": eid, "text": text, "source": "synthetic://uncertainty-model",
                             "published_at": iso(observed - timedelta(hours=2)), "available_at": iso(observed - timedelta(hours=1))}]}
    input_context(payload)
    inputs = {"sample_id": sid, "role": role, "input": payload}
    judge = {"sample_id": sid, "event_group_id": group, "condition": condition, "family": family,
             "language": language, "role": role, "allowed_fields": sorted(definitions),
             "known_fields": sorted(known), "unknown_fields": sorted(set(definitions) - known)}
    return inputs, judge


def probe_rows(config: dict) -> tuple[list[dict], list[dict]]:
    pairs = [render_probe(config, condition, language, role) for condition in CONDITIONS
             for language in config["languages"] for role in config["roles"]]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def build_uncertainty_cohort(config_path: Path, output: Path) -> dict:
    if output.exists():raise ValidationError("uncertainty cohort already exists")
    config = load_uncertainty_config(config_path)
    inputs, judges = probe_rows(config)
    artifacts = {"config.json": config_path.read_text(), "inputs.jsonl": jsonl(inputs), "judge.jsonl": jsonl(judges)}
    report = {"schema_version": "1", "kind": VERSION, "partition": "validation",
              "examples": len(inputs), "event_groups": len({r["event_group_id"] for r in judges}),
              "artifact_hashes": {k: sha256_bytes(v.encode()) for k,v in artifacts.items()}, "code": code_provenance(),
              "limitations": ["Synthetic development-only diagnostic; no training or final-test selection.",
                              "Finite-vocabulary classification of unknown fields, not an open-ended factuality metric.",
                              "Counterfactual, language and role variants share event groups."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".uncertainty-", dir=output.parent) as tmp:
        stage = Path(tmp)/"cohort"; stage.mkdir()
        for name, content in artifacts.items():(stage/name).write_text(content)
        (stage/"manifest.json").write_text(json_text(report));stage.rename(output)
    return report


def read_uncertainty_cohort(path: Path) -> tuple[dict, list[dict], list[dict]]:
    report = strict_json((path/"manifest.json").read_text())
    if report.get("kind") != VERSION or report.get("partition") != "validation" or report.get("schema_version") != "1":
        raise ValidationError("invalid uncertainty cohort")
    if set(report["artifact_hashes"]) != {"config.json", "inputs.jsonl", "judge.jsonl"}:
        raise ValidationError("uncertainty artifact set mismatch")
    for name, digest in report["artifact_hashes"].items():
        if sha256_file(path/name) != digest:raise ValidationError("uncertainty hash mismatch")
    config = load_uncertainty_config(path/"config.json")
    inputs, judges = probe_rows(config)
    for name, expected in (("inputs.jsonl", inputs), ("judge.jsonl", judges)):
        actual = [strict_json(line) for line in (path/name).read_text().splitlines()]
        if actual != expected:raise ValidationError("uncertainty replay mismatch")
    if report["examples"] != len(inputs) or report["event_groups"] != len({r["event_group_id"] for r in judges}):
        raise ValidationError("uncertainty counts mismatch")
    return report, inputs, judges


def judge_uncertainty(row: dict, judge: dict, output: dict | None) -> dict:
    if row["sample_id"] != judge["sample_id"] or row["role"] != judge["role"]:
        raise ValidationError("uncertainty judge/input identity mismatch")
    if output is None:return {"schema_valid": False, "vocabulary_valid": False, "correct": False}
    validate_agent_output(row["role"], output, input_context(row["input"]))
    actual, expected = set(output["unknowns"]), set(judge["unknown_fields"])
    outside = actual - set(judge["allowed_fields"])
    return {"schema_valid": True, "vocabulary_valid": not outside, "correct": actual == expected,
            "known_as_unknown": sorted(actual & set(judge["known_fields"])),
            "missing_unknowns": sorted(expected - actual), "out_of_vocabulary": sorted(outside)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    print(json_text(build_uncertainty_cohort(a.config, a.output)))


if __name__ == "__main__":main()
