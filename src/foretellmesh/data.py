"""Read unified JSONL and reject inconsistent identities before splitting."""

import hashlib
import json
from pathlib import Path
import unicodedata

from .schema import ForecastRecord, ValidationError, parse_record


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValidationError(f"nonstandard JSON constant: {value}")


def strict_json(text: str):
    try:
        return json.loads(text, object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise ValidationError(f"invalid JSON: {exc}") from exc


def question_key(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def validate_identities(records: list[ForecastRecord]) -> None:
    sample_ids, snapshots = set(), set()
    event_groups, question_groups, event_labels, question_labels = {}, {}, {}, {}
    for record in records:
        if record.sample_id in sample_ids:
            raise ValidationError(f"duplicate sample_id: {record.sample_id}")
        sample_ids.add(record.sample_id)
        # A source event retains its identity across dataset revisions.
        event = (record.dataset_source, record.event_id)
        snapshot = (*event, record.forecast_input.observation_time)
        if snapshot in snapshots:
            raise ValidationError(f"duplicate event observation: {record.sample_id}")
        snapshots.add(snapshot)
        key = question_key(record.forecast_input.question)
        for lookup, identity, kind in (
            (event_groups, event, "event"), (question_groups, key, "equivalent question")
        ):
            if identity in lookup and lookup[identity] != record.event_group_id:
                raise ValidationError(f"{kind} assigned to different event groups: {record.sample_id}")
            lookup[identity] = record.event_group_id
        if record.label is not None:
            label = (record.label.outcome, record.label.resolution_time)
            for lookup, identity in ((event_labels, event), (question_labels, key)):
                if identity in lookup and lookup[identity] != label:
                    raise ValidationError(f"conflicting labels for event/question: {record.sample_id}")
                lookup[identity] = label


def load_records(path: Path) -> tuple[list[ForecastRecord], str]:
    content = path.read_bytes()
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValidationError("dataset must be UTF-8") from exc
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(parse_record(strict_json(line)))
        except ValidationError as exc:
            raise ValidationError(f"{path.name}:{number}: {exc}") from exc
    if not records:
        raise ValidationError("dataset contains no records")
    validate_identities(records)
    return sorted(records, key=lambda item: item.sample_id), sha256_bytes(content)
