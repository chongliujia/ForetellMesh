"""Deterministic baselines, matched market comparisons, and run artifacts."""

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import subprocess

from . import __version__
from .config import ExperimentConfig, load_config
from .data import load_records, sha256_bytes
from .metrics import score_predictions
from .schema import ForecastRecord, ValidationError, iso
from .splits import SplitResult, split_records


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"


def code_provenance() -> dict:
    package = Path(__file__).resolve().parent
    chunks = []
    for path in sorted(package.glob("*.py")):
        chunks.extend((path.name.encode("utf-8"), b"\0", path.read_bytes(), b"\0"))
    commit = None
    dirty = None
    root = package.parent.parent
    if (root / ".git").exists():
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
            dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True,
                stderr=subprocess.DEVNULL, timeout=5,
            ).strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return {"package_version": __version__, "source_sha256": sha256_bytes(b"".join(chunks)),
            "git_commit": commit, "git_dirty": dirty, "python_version": platform.python_version()}


def evaluate_partition(
    records: list[ForecastRecord], base_rate: float | None, config: ExperimentConfig,
) -> tuple[dict, list[dict]]:
    outcomes = [record.label.outcome for record in records]
    probabilities = {
        "constant_0_5": [0.5] * len(records),
        "empirical_base_rate": [base_rate] * len(records),
        "market": [record.forecast_input.market.probability
                   if record.forecast_input.market is not None else None for record in records],
    }
    probabilities = {name: probabilities[name] for name in config.baselines}
    if "empirical_base_rate" in probabilities and base_rate is None:
        raise ValidationError("empirical baseline requires a training fit")
    kwargs = {"ece_bins": config.ece_bins, "log_loss_epsilon": config.log_loss_epsilon}
    scores = {name: score_predictions(values, outcomes, **kwargs)
              for name, values in probabilities.items()}
    matched = [index for index, p in enumerate(probabilities.get("market", [])) if p is not None]
    paired_scores = {
        name: score_predictions([values[index] for index in matched],
                                [outcomes[index] for index in matched], **kwargs)
        for name, values in probabilities.items()
    }
    deltas = {}
    for name in probabilities:
        if name == "market" or "market" not in probabilities:
            continue
        deltas[name] = {
            metric: paired_scores[name][metric] - paired_scores["market"][metric]
            if matched else None for metric in ("brier", "log_loss", "ece")
        }
    predictions = [
        {"sample_id": record.sample_id, "dataset_source": record.dataset_source,
         "dataset_version": record.dataset_version, "event_id": record.event_id,
         "event_group_id": record.event_group_id,
         "observation_time": iso(record.forecast_input.observation_time),
         "resolution_time": iso(record.label.resolution_time),
         "outcome": record.label.outcome,
         "probabilities": {name: values[index] for name, values in probabilities.items()}}
        for index, record in enumerate(records)
    ]
    return {
        "sample_count": len(records),
        "event_group_count": len({record.event_group_id for record in records}),
        "baselines": scores,
        "market_matched": {"enabled": "market" in probabilities,
                           "sample_count": len(matched), "baselines": paired_scores,
                           "baseline_minus_market": deltas},
    }, predictions


def evaluate(dataset: Path, config_path: Path) -> tuple[dict, list[dict], SplitResult]:
    config, config_hash = load_config(config_path)
    records, dataset_hash = load_records(dataset)
    split_result = split_records(records, config)
    train = split_result.partitions["train"]
    fit_empirical = "empirical_base_rate" in config.baselines
    if fit_empirical and not train:
        raise ValidationError("no eligible training records remain for empirical base rate")
    if not split_result.partitions["test"]:
        raise ValidationError("no eligible test records remain")
    # The fit uses the same per-snapshot weighting as evaluation. Exact duplicate
    # question/time snapshots have already been removed by the split pipeline.
    base_rate = math.fsum(record.label.outcome for record in train) / len(train) if fit_empirical else None
    reports, predictions = {}, []
    for split in ("validation", "test"):
        reports[split], rows = evaluate_partition(split_result.partitions[split], base_rate, config)
        predictions.extend({**row, "split": split} for row in rows)
    counts = Counter(record.dataset_source for record in records)
    report = {
        "schema_version": "1",
        "run_name": config.run_name,
        "synthetic_only": all(config.sources[record.dataset_source].synthetic for record in records),
        "experiment": {
            "kind": "deterministic_baseline_evaluation", "model": None,
            "config_sha256": config_hash, "dataset_sha256": dataset_hash,
            "split_version": config.split_version,
            "split_manifest_sha256": sha256_bytes(json_text(split_result.manifest()).encode("utf-8")),
            "validation_start": iso(config.validation_start), "test_start": iso(config.test_start),
            "evaluation_as_of": iso(config.evaluation_as_of),
            "ece_bins": config.ece_bins, "log_loss_epsilon": config.log_loss_epsilon,
            "baselines": list(config.baselines),
            "sources": {name: {**asdict(config.sources[name]), "record_count": count}
                        for name, count in sorted(counts.items())},
            "code": code_provenance(),
        },
        "audit": {
            "structural_and_timestamp_checks": "passed",
            "deduplication": "normalized exact question + observation timestamp",
            "semantic_event_grouping": "provided by data curator; not automatically verified",
            "split_policy": "chronological; drop whole groups crossing split boundaries",
            "score_weighting": "equal weight per retained observation snapshot",
            "input_count": len(records),
            "retained_counts": {name: len(rows) for name, rows in split_result.partitions.items()},
            "excluded_count": len(split_result.exclusions),
            "exclusion_reasons": dict(sorted(Counter(
                item["reason"] for item in split_result.exclusions
            ).items())),
        },
        "empirical_base_rate": None if not fit_empirical else {
            "probability": base_rate, "training_sample_count": len(train),
            "fit_split": "train", "fit_label_cutoff": iso(config.validation_start),
        },
        "evaluation": reports,
    }
    return report, predictions, split_result


def markdown_report(report: dict) -> str:
    lines = [f"# {report['run_name']}", ""]
    if report["synthetic_only"]:
        lines.extend(["**Synthetic demonstration only; these scores are not forecasting evidence.**", ""])
    fitted = report["empirical_base_rate"]
    fit_description = (f"Train-only empirical base rate: {fitted['probability']:.6f}."
                       if fitted is not None else "Empirical base-rate fitting is disabled by configuration.")
    lines.extend([
        f"Input records: {report['audit']['input_count']}; excluded: {report['audit']['excluded_count']}.",
        fit_description, "",
        "| Split | Baseline | Scored / eligible | Coverage | Brier | Log loss | ECE |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])

    def formatted(value):
        return "N/A" if value is None else f"{value:.6f}"

    for split, result in report["evaluation"].items():
        for name, metrics in result["baselines"].items():
            lines.append(
                f"| {split} | {name} | {metrics['prediction_count']} / {metrics['eligible_count']} | "
                + " | ".join(formatted(metrics[key]) for key in ("coverage", "brier", "log_loss", "ece"))
                + " |"
            )
    lines.extend(["", "Market comparisons below use identical samples. Negative deltas favor the baseline.",
                  "", "| Split | Baseline | Matched samples | Brier Δ | Log loss Δ | ECE Δ |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for split, result in report["evaluation"].items():
        matched = result["market_matched"]
        for name, deltas in matched["baseline_minus_market"].items():
            lines.append(f"| {split} | {name} | {matched['sample_count']} | "
                         + " | ".join(formatted(deltas[key]) for key in ("brier", "log_loss", "ece"))
                         + " |")
    lines.extend(["", "Exclusion reasons: " + json.dumps(report["audit"]["exclusion_reasons"], sort_keys=True),
                  "", "Timestamp checks cover declared metadata. Semantic equivalence and historical source "
                  "snapshots require curator review. ECE uses fixed equal-width probability bins; "
                  "small samples are not evidence of calibration.",
                  "", "See report.json for calibration curves, hashes, source versions, and split policy.", ""])
    return "\n".join(lines)
