"""Conservative chronological splitting with whole-group exclusions."""

from collections import defaultdict
from dataclasses import dataclass

from .config import ExperimentConfig
from .data import question_key, validate_identities
from .schema import ForecastRecord, ValidationError


@dataclass(frozen=True)
class SplitResult:
    partitions: dict[str, list[ForecastRecord]]
    exclusions: list[dict[str, str]]

    def manifest(self) -> list[dict[str, str]]:
        entries = [
            {"sample_id": record.sample_id, "event_group_id": record.event_group_id,
             "split": split}
            for split, records in self.partitions.items() for record in records
        ]
        entries.extend({**entry, "split": "excluded"} for entry in self.exclusions)
        return sorted(entries, key=lambda item: item["sample_id"])


def split_records(records: list[ForecastRecord], config: ExperimentConfig) -> SplitResult:
    validate_identities(records)
    groups = defaultdict(list)
    for record in records:
        source = config.sources.get(record.dataset_source)
        if source is None or source.version != record.dataset_version:
            raise ValidationError(f"unregistered dataset source/version: {record.sample_id}")
        groups[record.event_group_id].append(record)
    partitions = {"train": [], "validation": [], "test": []}
    exclusions = []

    def candidate(record: ForecastRecord) -> str:
        time = record.forecast_input.observation_time
        if time < config.validation_start:
            return "train"
        return "validation" if time < config.test_start else "test"

    for group_id, members in sorted(groups.items()):
        windows = {candidate(record) for record in members}
        reason = None
        if len(windows) > 1:
            reason = "group_crosses_time_boundary"
        elif "train" in windows and any(
            config.sources[record.dataset_source].role == "eval_only" for record in members
        ):
            reason = "heldout_source_group_in_train_window"
        for record in members:
            split = candidate(record)
            record_reason = reason
            if record_reason is None:
                cutoff = {"train": config.validation_start, "validation": config.test_start,
                          "test": config.evaluation_as_of}[split]
                if record.forecast_input.observation_time > config.evaluation_as_of:
                    record_reason = "observation_after_evaluation_cutoff"
                elif record.label is None:
                    record_reason = "unresolved"
                elif record.label.available_at > cutoff:
                    record_reason = "label_unavailable_at_split_cutoff"
            if record_reason:
                exclusions.append({"sample_id": record.sample_id, "event_group_id": group_id,
                                   "reason": record_reason})
            else:
                partitions[split].append(record)
    for split, records_in_split in partitions.items():
        # Identical question/time snapshots count once. Prefer held-out source
        # provenance, then source/version/sample identity for stable selection.
        seen = set()
        retained = []
        ordered = sorted(records_in_split, key=lambda item: (
            config.sources[item.dataset_source].role != "eval_only",
            item.dataset_source, item.dataset_version, item.sample_id,
        ))
        for record in ordered:
            key = (question_key(record.forecast_input.question),
                   record.forecast_input.observation_time)
            if key in seen:
                exclusions.append({"sample_id": record.sample_id,
                                   "event_group_id": record.event_group_id,
                                   "reason": "duplicate_question_observation"})
            else:
                seen.add(key)
                retained.append(record)
        partitions[split] = sorted(retained, key=lambda item: item.sample_id)
    return SplitResult(partitions, sorted(exclusions, key=lambda item: item["sample_id"]))
