"""Version-controlled experiment and source policies."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .data import sha256_bytes, strict_json
from .schema import ValidationError, fields, nonempty, probability, timestamp


@dataclass(frozen=True)
class SourcePolicy:
    version: str
    role: str
    synthetic: bool


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    split_version: str
    validation_start: datetime
    test_start: datetime
    evaluation_as_of: datetime
    ece_bins: int
    log_loss_epsilon: float
    sources: dict[str, SourcePolicy]
    baselines: tuple[str, ...] = ("constant_0_5", "empirical_base_rate", "market")


def load_config(path: Path) -> tuple[ExperimentConfig, str]:
    content = path.read_bytes()
    try:
        raw = strict_json(content.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValidationError("config must be UTF-8") from exc
    required = {"run_name", "split_version", "validation_start", "test_start",
                "evaluation_as_of", "ece_bins", "log_loss_epsilon", "sources"}
    if isinstance(raw, dict) and "baselines" in raw:
        required.add("baselines")
    fields(raw, required, "config")
    baselines = raw.get("baselines", ["constant_0_5", "empirical_base_rate", "market"])
    if not isinstance(baselines, list) or not baselines or any(
        not isinstance(name, str) or name not in ("constant_0_5", "empirical_base_rate", "market")
        for name in baselines
    ):
        raise ValidationError("baselines: expected a nonempty list of supported baseline names")
    if len(set(baselines)) != len(baselines):
        raise ValidationError("baselines must not contain duplicates")
    validation_start = timestamp(raw["validation_start"], "validation_start")
    test_start = timestamp(raw["test_start"], "test_start")
    evaluation_as_of = timestamp(raw["evaluation_as_of"], "evaluation_as_of")
    if not validation_start < test_start <= evaluation_as_of:
        raise ValidationError("require validation_start < test_start <= evaluation_as_of")
    bins = raw["ece_bins"]
    if type(bins) is not int or not 1 <= bins <= 1000:
        raise ValidationError("ece_bins: expected integer in [1, 1000]")
    epsilon = probability(raw["log_loss_epsilon"], "log_loss_epsilon")
    if not 0 < epsilon < 0.5:
        raise ValidationError("log_loss_epsilon: require 0 < epsilon < 0.5")
    if not isinstance(raw["sources"], dict) or not raw["sources"]:
        raise ValidationError("sources: expected a nonempty object")
    sources = {}
    for name, item in raw["sources"].items():
        nonempty(name, "source name")
        fields(item, {"version", "role", "synthetic"}, f"source {name}")
        if item["role"] not in ("train_eval", "eval_only"):
            raise ValidationError(f"source {name}: role must be train_eval or eval_only")
        if name == "forecastbench" and item["role"] != "eval_only":
            raise ValidationError("ForecastBench is reserved for evaluation")
        if type(item["synthetic"]) is not bool:
            raise ValidationError(f"source {name}: synthetic must be boolean")
        sources[name] = SourcePolicy(nonempty(item["version"], "source version"),
                                     item["role"], item["synthetic"])
    return ExperimentConfig(
        nonempty(raw["run_name"], "run_name"), nonempty(raw["split_version"], "split_version"),
        validation_start, test_start, evaluation_as_of, bins, epsilon, sources, tuple(baselines),
    ), sha256_bytes(content)
