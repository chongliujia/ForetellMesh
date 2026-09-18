"""Deterministic, bilingual schema warm-up data with explicit mathematical oracles.

All dates describe simulated scenarios, not historical publications. Numerical
answers are provided as tool evidence; the LM learns to report and cite them.
"""

from datetime import timedelta
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import tempfile

from .data import sha256_bytes, strict_json, validate_identities
from .evaluation import code_provenance, json_text
from .schema import ValidationError, fields, iso, parse_record, timestamp


VERSION = "synthetic_forecast_warmup_v1"
SOURCE = "synthetic_forecast_warmup"
FAMILIES = ("complement", "at_least_one", "without_replacement", "mixture", "beta_binomial", "bayes_signal")


def canonical_hash(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                  separators=(",", ":")).encode())


def oracle(family: str, params: dict) -> tuple[Fraction, Fraction | None, str, str]:
    """Return predictive probability, prior mean, confidence, exact calculation."""
    keys = {"complement": {"a", "b"}, "at_least_one": {"a", "b", "n"},
            "without_replacement": {"total", "successes", "draws"},
            "mixture": {"weight", "low", "high"}, "beta_binomial": {"successes", "failures"},
            "bayes_signal": {"prior", "sensitivity", "false_positive"}}
    if family not in keys:
        raise ValidationError("unsupported synthetic family")
    fields(params, keys[family], "oracle parameters")
    if any(type(v) is not int or not 1 <= v <= 10000 for v in params.values()):
        raise ValidationError("oracle parameters must be bounded positive integers")
    base_rate, confidence = None, "high"
    if family in ("complement", "at_least_one"):
        a, b = params["a"], params["b"]
        if not 0 < a < b:
            raise ValidationError("require 0 < a < b")
        p = Fraction(a, b)
        n = 1 if family == "complement" else params["n"]
        if n > 10:
            raise ValidationError("too many trials")
        result = 1 - p if family == "complement" else 1 - (1 - p) ** n
        formula = f"1 - {p}" if family == "complement" else f"1 - (1 - {p})^{n}"
    elif family == "without_replacement":
        total, successes, draws = (params[k] for k in ("total", "successes", "draws"))
        if not draws <= successes < total <= 100:
            raise ValidationError("require draws <= successes < total <= 100")
        result = Fraction(math.comb(successes, draws), math.comb(total, draws))
        formula = f"C({successes},{draws}) / C({total},{draws})"
    elif family == "mixture":
        weight, low, high = (Fraction(params[k], 100) for k in ("weight", "low", "high"))
        if not 0 < weight < 1 or not 0 < low < high < 1:
            raise ValidationError("invalid mixture parameters")
        result = (1 - weight) * low + weight * high
        formula = f"(1 - {weight}) * {low} + {weight} * {high}"
    elif family == "beta_binomial":
        a, b = params["successes"] + 1, params["failures"] + 1
        result, base_rate = Fraction(a, a + b), Fraction(1, 2)
        # Confidence concerns uncertainty in the rate, never whether p > 0.5.
        variance = Fraction(a * b, (a + b) ** 2 * (a + b + 1))
        confidence = "low" if variance >= Fraction(1, 100) else "medium"
        formula = f"Beta(1,1) posterior predictive: {a}/({a}+{b}); posterior variance={variance}"
    else:
        prior, sensitivity, false_positive = (Fraction(params[k], 100) for k in (
            "prior", "sensitivity", "false_positive"))
        if any(not 0 < p < 1 for p in (prior, sensitivity, false_positive)):
            raise ValidationError("invalid signal probabilities")
        base_rate = prior
        result = prior * sensitivity / (prior * sensitivity + (1 - prior) * false_positive)
        formula = f"({prior} * {sensitivity}) / ({prior} * {sensitivity} + (1 - {prior}) * {false_positive})"
    return result, base_rate, confidence, formula


def render_case(case: dict) -> tuple[dict, dict]:
    fields(case, {"version", "family", "parameters", "language", "observation_time"}, "synthetic case")
    if case["version"] != VERSION or case["language"] not in ("en", "zh"):
        raise ValidationError("unsupported synthetic version/language")
    family, p = case["family"], case["parameters"]
    result, base_rate, confidence, formula = oracle(family, p)
    observed = timestamp(case["observation_time"], "synthetic observation_time")
    zh = case["language"] == "zh"
    scenarios = {
        "complement": ("One Bernoulli trial has success probability a/b. Will it fail?",
                       "一次伯努利试验的成功概率为 a/b。这次试验会失败吗？"),
        "at_least_one": ("Each of n independent trials succeeds with probability a/b. Will at least one succeed?",
                         "n 次独立试验中每次成功概率为 a/b。至少一次会成功吗？"),
        "without_replacement": ("Draw draws objects uniformly without replacement from total objects, successes of which are marked. Will all drawn objects be marked?",
                                "从 total 个物体中等概率不放回抽取 draws 个，其中 successes 个有标记。抽出的物体会全部有标记吗？"),
        "mixture": ("A high-rate scenario is selected with weight/100 probability; otherwise a low-rate scenario is selected. Their success rates are high/100 and low/100. Will the trial succeed?",
                    "以 weight/100 的概率选择高成功率情景，否则选择低成功率情景，成功率分别为 high/100 和 low/100。试验会成功吗？"),
        "beta_binomial": ("With a Beta(1,1) prior and successes successes plus failures failures from IID Bernoulli trials with a fixed unknown rate, will the next trial succeed?",
                          "假设成功率固定但未知，先验为 Beta(1,1)，已观察到独立同分布试验 successes 次成功、failures 次失败。下一次试验会成功吗？"),
        "bayes_signal": ("A synthetic state A has prior/100 prior probability. A signal is positive with probability sensitivity/100 in A and false_positive/100 outside A. Given a positive signal, is the state A?",
                         "合成状态 A 的先验概率为 prior/100。在 A 下信号为阳性的概率为 sensitivity/100，在非 A 下为 false_positive/100。观察到阳性信号后，状态是 A 吗？"),
    }
    parameters = json.dumps(p, sort_keys=True, ensure_ascii=False)
    question = ("合成情景：" if zh else "Synthetic scenario: ") + scenarios[family][int(zh)] + "\n" + parameters
    context = ("参数和独立性等条件均为合成设定，不代表真实世界。" if zh else
               "Parameters and independence assumptions are synthetic stipulations, not real-world facts.")
    def evidence(eid, text, source):
        return {"evidence_id": eid, "text": text, "source": source,
                "published_at": iso(observed), "available_at": iso(observed)}
    ev = [evidence("setup", context + "\n" + question, "synthetic://scenario"),
          evidence("calculation", f"Python Fraction oracle: {formula}; exact={result}; probability={float(result):.6f}",
                   "synthetic://python-fraction-oracle/v1")]
    counter = []
    unknowns = ["下一次随机结果尚未观察到。" if zh else "The next random outcome has not been observed."]
    if family == "bayes_signal":
        unknowns = ["潜在状态未被直接观察到。" if zh else "The latent state has not been directly observed."]
    if family == "beta_binomial":
        ev.append(evidence("sampling_limit", "有限样本不能确定真实成功率；平稳性是设定的假设。" if zh else
                           "Finite observations do not determine the true rate; stationarity is an assumption.",
                           "synthetic://sampling-limits"))
        counter = ["sampling_limit"]
        unknowns.append("真实成功率仍有后验不确定性。" if zh else "The underlying rate has posterior uncertainty.")
    group = "synthetic:" + canonical_hash({"family": family, "parameters": p})[:24]
    # The simulated outcome is sampled separately, after defining the probability.
    draw = random.Random(int(canonical_hash({"group": group, "purpose": "outcome"}), 16))
    outcome = int(draw.randrange(result.denominator) < result.numerator)
    record = {
        "sample_id": group + ":" + case["language"], "dataset_source": SOURCE, "dataset_version": VERSION,
        # Language variants have distinct presentation IDs and one semantic group.
        "event_id": group + ":" + case["language"], "event_group_id": group, "question": question,
        "observation_time": iso(observed), "evidence": ev, "market": None,
        "label": {"outcome": outcome, "resolution_time": iso(observed + timedelta(days=1)),
                  "available_at": iso(observed + timedelta(days=1, hours=1))},
    }
    target = {"event": question, "probability": round(float(result), 6), "confidence": confidence,
              "base_rate": None if base_rate is None else round(float(base_rate), 6),
              "key_evidence": ["setup", "calculation"], "counter_evidence": counter,
              "unknowns": unknowns, "observation_time": iso(observed)}
    parse_record(record)
    return record, target


def draw_parameters(rng: random.Random, family: str, index: int) -> dict:
    if family in ("complement", "at_least_one"):
        b = rng.randint(20, 100)
        p = {"a": rng.randint(math.ceil(b / 10), math.floor(9 * b / 10)), "b": b}
        reduced = Fraction(p["a"], p["b"])
        p = {"a": reduced.numerator, "b": reduced.denominator}
        if family == "at_least_one":
            p["n"] = rng.randint(2, 4)
        return p
    if family == "without_replacement":
        total = rng.randint(10, 80)
        return {"total": total, "draws": 2, "successes": rng.randint(2, total - 1)}
    if family == "mixture":
        return {"weight": rng.randint(10, 90), "low": rng.randint(5, 40), "high": rng.randint(60, 95)}
    if family == "beta_binomial":
        limit = 8 if index % 2 == 0 else 80
        return {"successes": rng.randint(1, limit), "failures": rng.randint(1, limit)}
    return {"prior": rng.randint(1, 40), "sensitivity": rng.randint(70, 99), "false_positive": rng.randint(1, 30)}


def generate_synthetic_sft(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError("synthetic output already exists")
    raw = config_path.read_bytes()
    config = strict_json(raw.decode())
    fields(config, {"version", "seed", "groups_per_family", "observation_times", "languages", "families"}, "generator config")
    if config["version"] != VERSION or type(config["seed"]) is not int:
        raise ValidationError("invalid generator version/seed")
    if config["languages"] != ["en", "zh"] or config["families"] != list(FAMILIES):
        raise ValidationError("v1 requires the six families and both languages")
    for key in ("groups_per_family", "observation_times"):
        fields(config[key], {"train", "validation", "test"}, key)
    for n in config["groups_per_family"].values():
        if type(n) is not int or not 1 <= n <= 100:
            raise ValidationError("groups_per_family must be in [1,100]")
    times = [timestamp(config["observation_times"][s], s) for s in ("train", "validation", "test")]
    if not times[0] < times[1] < times[2]:
        raise ValidationError("synthetic split times must increase")
    rng, seen, records, targets, proofs = random.Random(config["seed"]), set(), [], [], {}
    for split in ("train", "validation", "test"):
        for family in FAMILIES:
            for index in range(config["groups_per_family"][split]):
                for _ in range(10000):
                    params = draw_parameters(rng, family, index)
                    identity = canonical_hash({"family": family, "parameters": params})
                    if identity not in seen:
                        seen.add(identity)
                        break
                else:
                    raise ValidationError("could not generate distinct parameter groups")
                for language in config["languages"]:
                    case = {"version": VERSION, "family": family, "parameters": params,
                            "language": language, "observation_time": config["observation_times"][split]}
                    record, target = render_case(case)
                    name = canonical_hash(case) + ".json"
                    proof_bytes = json_text(case).encode()
                    proofs[name] = proof_bytes
                    observed = record["observation_time"]
                    targets.append({"schema_version": "1", "sample_id": record["sample_id"],
                                    "input_sha256": canonical_hash(parse_record(record).forecast_input.to_payload()),
                                    "target": target, "provenance": {
                                        "kind": "synthetic_oracle", "author": "foretellmesh.synthetic_sft.render_case/v1",
                                        "issued_at": observed, "available_at": observed,
                                        "information_cutoff": observed, "question_available_at": observed,
                                        "artifact": "proofs/" + name, "artifact_sha256": sha256_bytes(proof_bytes),
                                        "reviewer": "deterministic_oracle_replay", "review_notes": "Synthetic scenario; simulated dates, exact Fraction calculation."}})
                    records.append(record)
    validate_identities([parse_record(r) for r in records])
    records.sort(key=lambda row: row["sample_id"])
    targets.sort(key=lambda row: row["sample_id"])
    def jsonl(rows):
        return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    record_text, target_text = jsonl(records), jsonl(targets)
    report = {"schema_version": "1", "kind": "synthetic_schema_warmup", "record_count": len(records),
              "event_group_count": len(seen), "config_sha256": sha256_bytes(raw),
              "records_sha256": sha256_bytes(record_text.encode()), "targets_sha256": sha256_bytes(target_text.encode()),
              "code": code_provenance(), "historical_data": False,
              "limitations": "Simulated dates and outcomes; shared templates across splits; not a real forecasting benchmark."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".synthetic-sft-", dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        (stage / "proofs").mkdir(parents=True)
        for name, content in proofs.items():
            (stage / "proofs" / name).write_bytes(content)
        (stage / "records.jsonl").write_text(record_text, encoding="utf-8")
        (stage / "targets.jsonl").write_text(target_text, encoding="utf-8")
        (stage / "config.json").write_bytes(raw)
        (stage / "manifest.json").write_text(json_text(report), encoding="utf-8")
        stage.rename(output)
    return report
