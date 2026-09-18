"""Strict data contracts. Labels never belong to the model-visible input."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any


class ValidationError(ValueError):
    """A record or experiment violates the data contract."""


def fields(value: Any, required: set[str], context: str) -> dict:
    if not isinstance(value, dict):
        raise ValidationError(f"{context}: expected an object")
    missing, extra = required - value.keys(), value.keys() - required
    if missing or extra:
        raise ValidationError(
            f"{context}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}"
        )
    return value


def nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context}: expected a nonempty string")
    return value


def timestamp(value: Any, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(nonempty(value, context))
    except ValueError as exc:
        raise ValidationError(f"{context}: expected an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValidationError(f"{context}: timezone is required")
    try:
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"{context}: timestamp outside supported UTC range") from exc


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def probability(value: Any, context: str = "probability") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{context}: expected a number in [0, 1]")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValidationError(f"{context}: expected a finite number") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValidationError(f"{context}: expected a finite number in [0, 1]")
    return result


def binary_outcome(value: Any) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValidationError("outcome: expected integer 0 or 1")
    return value


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str
    source: str
    published_at: datetime
    available_at: datetime


@dataclass(frozen=True)
class MarketSnapshot:
    probability: float
    observed_at: datetime
    available_at: datetime


@dataclass(frozen=True)
class ForecastInput:
    question: str
    observation_time: datetime
    evidence: tuple[Evidence, ...]
    market: MarketSnapshot | None

    def to_payload(self) -> dict:
        """Explicit allowlist: no labels, split, dataset identity, or resolution times."""
        return {
            "question": self.question,
            "observation_time": iso(self.observation_time),
            "evidence": [
                {**asdict(item), "published_at": iso(item.published_at),
                 "available_at": iso(item.available_at)}
                for item in self.evidence
            ],
            "market": None if self.market is None else {
                "probability": self.market.probability,
                "observed_at": iso(self.market.observed_at),
                "available_at": iso(self.market.available_at),
            },
        }


@dataclass(frozen=True)
class Label:
    outcome: int
    resolution_time: datetime
    available_at: datetime


@dataclass(frozen=True)
class ForecastRecord:
    sample_id: str
    dataset_source: str
    dataset_version: str
    event_id: str
    event_group_id: str
    forecast_input: ForecastInput
    label: Label | None


def parse_record(raw: Any) -> ForecastRecord:
    fields(raw, {"sample_id", "dataset_source", "dataset_version", "event_id",
                 "event_group_id", "question", "observation_time", "evidence",
                 "market", "label"}, "record")
    metadata = {key: nonempty(raw[key], key) for key in (
        "sample_id", "dataset_source", "dataset_version", "event_id", "event_group_id"
    )}
    observation = timestamp(raw["observation_time"], "observation_time")
    if not isinstance(raw["evidence"], list):
        raise ValidationError("evidence: expected an array")
    evidence = []
    seen_ids = set()
    for item in raw["evidence"]:
        fields(item, {"evidence_id", "text", "source", "published_at", "available_at"},
               "evidence")
        entry = Evidence(
            *(nonempty(item[key], f"evidence.{key}") for key in
              ("evidence_id", "text", "source")),
            timestamp(item["published_at"], "evidence.published_at"),
            timestamp(item["available_at"], "evidence.available_at"),
        )
        if not entry.published_at <= entry.available_at <= observation:
            raise ValidationError("evidence: require published_at <= available_at <= observation_time")
        if entry.evidence_id in seen_ids:
            raise ValidationError("evidence: duplicate evidence_id")
        seen_ids.add(entry.evidence_id)
        evidence.append(entry)

    market = None
    if raw["market"] is not None:
        item = fields(raw["market"], {"probability", "observed_at", "available_at"}, "market")
        market = MarketSnapshot(
            probability(item["probability"], "market.probability"),
            timestamp(item["observed_at"], "market.observed_at"),
            timestamp(item["available_at"], "market.available_at"),
        )
        if not market.observed_at <= market.available_at <= observation:
            raise ValidationError("market: require observed_at <= available_at <= observation_time")

    label = None
    if raw["label"] is not None:
        item = fields(raw["label"], {"outcome", "resolution_time", "available_at"}, "label")
        label = Label(
            binary_outcome(item["outcome"]),
            timestamp(item["resolution_time"], "label.resolution_time"),
            timestamp(item["available_at"], "label.available_at"),
        )
        if not observation < label.resolution_time <= label.available_at:
            raise ValidationError("label: require observation_time < resolution_time <= available_at")
    return ForecastRecord(
        **metadata,
        forecast_input=ForecastInput(nonempty(raw["question"], "question"), observation,
                                     tuple(evidence), market),
        label=label,
    )
