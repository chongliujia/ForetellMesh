"""Auditable source import and explicit provenance annotations for promotion."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

from .adapters import Candidate, forecastbench_candidates, kalshi_snapshot_candidates, prophet_candidates
from .data import sha256_bytes, sha256_file, strict_json, validate_identities
from .evaluation import code_provenance, json_text
from .schema import ValidationError, fields, iso, nonempty, parse_record, timestamp
from .sources import load_source, verify_artifact


def audit_candidates(candidates: list[Candidate]) -> None:
    seen = set()
    groups = defaultdict(list)
    for candidate in candidates:
        if candidate.sample_id in seen:
            raise ValidationError(f"duplicate source sample: {candidate.sample_id}")
        seen.add(candidate.sample_id)
        groups[candidate.event_id].append(candidate)
    for members in groups.values():
        if len({item.outcome for item in members if item.outcome is not None}) > 1:
            for item in members:
                item.blockers.append("conflicting_event_outcomes")


def annotation_template(source: str, inputs: dict, candidates: list[Candidate]) -> dict:
    return {
        "schema_version": "1", "source": source, "input_hashes": inputs,
        "entries": {
            item.sample_id: {"decision": "pending", "event_group_id": item.suggested_event_group_id,
                             "observation_time": item.observation_time, "question_available_at": None,
                             "resolution_time": None, "label_available_at": None, "provenance": None}
            for item in candidates
        },
    }


def promote_candidates(
    candidates: list[Candidate], source: str, version: str, inputs: dict, annotations: dict | None,
) -> tuple[list[dict], list[dict]]:
    entries = {}
    if annotations is not None:
        fields(annotations, {"schema_version", "source", "input_hashes", "entries"}, "annotations")
        if annotations["schema_version"] != "1" or annotations["source"] != source:
            raise ValidationError("annotation schema/source mismatch")
        if annotations["input_hashes"] != inputs:
            raise ValidationError("annotation input hashes do not match the source files")
        entries = annotations["entries"]
        if not isinstance(entries, dict) or entries.keys() - {item.sample_id for item in candidates}:
            raise ValidationError("annotations contain unknown sample IDs or invalid entries")
    accepted, disposition = [], []
    # Suggested native groups are the minimum grouping granularity. An annotation
    # may merge related groups across sources, but must not split a native group.
    group_assignments = {}
    for candidate in candidates:
        annotation = entries.get(candidate.sample_id)
        if annotation is not None:
            fields(annotation, {"decision", "event_group_id", "observation_time", "question_available_at",
                                "resolution_time", "label_available_at", "provenance"}, "annotation entry")
            if annotation["decision"] not in ("pending", "include", "exclude"):
                raise ValidationError("annotation decision must be pending/include/exclude")
        decision = "pending" if annotation is None else annotation["decision"]
        if decision == "exclude":
            nonempty(annotation["provenance"], "exclusion reason/provenance")
            disposition.append({"sample_id": candidate.sample_id, "status": "excluded",
                                "reasons": [annotation["provenance"]]})
            continue
        if decision == "pending":
            disposition.append({"sample_id": candidate.sample_id, "status": "quarantined",
                                "reasons": sorted(set(candidate.blockers + ["provenance_annotation_required"]))})
            continue
        if candidate.blockers:
            raise ValidationError(f"cannot include {candidate.sample_id}: {candidate.blockers}")
        nonempty(annotation["provenance"], "annotation provenance")
        group = nonempty(annotation["event_group_id"], "event_group_id")
        native_group = candidate.suggested_event_group_id
        if native_group in group_assignments and group_assignments[native_group] != group:
            raise ValidationError("annotations must not split a native event group")
        group_assignments[native_group] = group
        observed = timestamp(annotation["observation_time"], "annotated observation_time")
        available = timestamp(annotation["question_available_at"], "question_available_at")
        if available > observed:
            raise ValidationError("question snapshot was unavailable at observation_time")
        if candidate.observation_time is not None:
            if observed != timestamp(candidate.observation_time, "source observation_time"):
                raise ValidationError("annotation cannot move the source observation_time")
        if source == "forecastbench":
            if observed.date().isoformat() != candidate.context["forecast_due_date"]:
                raise ValidationError("ForecastBench observation must be on its forecast_due_date (UTC)")
            if observed < timestamp(candidate.context["freeze_datetime"], "freeze_datetime"):
                raise ValidationError("ForecastBench observation precedes its frozen snapshot")
        resolution = timestamp(annotation["resolution_time"], "annotated resolution_time")
        label_available = timestamp(annotation["label_available_at"], "label_available_at")
        date_hint = candidate.context.get("resolution_date_hint")
        if date_hint is not None and resolution.date().isoformat() != date_hint:
            raise ValidationError("annotated UTC resolution date disagrees with source resolution_date")
        floor = candidate.context.get("label_available_at_floor")
        if floor is not None and label_available < timestamp(floor, "label availability floor"):
            raise ValidationError("label availability precedes the archived finalized snapshot")
        raw = {
            "sample_id": candidate.sample_id, "dataset_source": source, "dataset_version": version,
            "event_id": candidate.event_id, "event_group_id": group, "question": candidate.question,
            "observation_time": iso(observed), "evidence": [], "market": candidate.market,
            "label": {"outcome": candidate.outcome, "resolution_time": iso(resolution),
                      "available_at": iso(label_available)},
        }
        if raw["market"] is not None:
            # A documented snapshot-publication time is a conservative bound on
            # when its recorded quote was available. Never backdate this proof.
            market = raw["market"]
            raw["market"] = {**market, "available_at": iso(max(
                available, timestamp(market["available_at"], "market.available_at")
            ))}
        parse_record(raw)
        accepted.append(raw)
        disposition.append({"sample_id": candidate.sample_id, "status": "accepted", "reasons": []})
    validate_identities([parse_record(raw) for raw in accepted])
    return accepted, disposition


def ingest_source(
    *, source_name: str, registry: Path, data: Path, output: Path,
    resolutions: Path | None = None, annotations_path: Path | None = None,
) -> dict:
    if output.exists():
        raise ValidationError(f"output already exists: {output}")
    source = load_source(registry, source_name)
    inputs = {"data": sha256_file(data)}
    if resolutions is not None:
        inputs["resolutions"] = sha256_file(resolutions)
    if source_name == "prophet_arena":
        if resolutions is not None:
            raise ValidationError("Prophet Arena labels are already in the CSV")
        verify_artifact(data, source, "data")
        candidates, statistics = prophet_candidates(data)
        version = source.revision
    elif source_name == "forecastbench":
        if resolutions is None:
            raise ValidationError("ForecastBench requires --resolutions")
        verify_artifact(data, source, "questions")
        verify_artifact(resolutions, source, "resolutions")
        candidates, statistics = forecastbench_candidates(data, resolutions)
        version = source.revision
    elif source_name == "prediction_market_analysis":
        if resolutions is not None:
            raise ValidationError("PMA expects a local Kalshi metadata Parquet shard")
        candidates, statistics = kalshi_snapshot_candidates(data)
        # A code revision is NOT a version of an independently hosted archive.
        version = "sha256:" + inputs["data"]
    else:
        raise ValidationError(f"no adapter for {source_name}")
    if not candidates:
        raise ValidationError("source produced no candidates")
    candidates.sort(key=lambda item: item.sample_id)
    audit_candidates(candidates)
    annotation_bytes = None if annotations_path is None else annotations_path.read_bytes()
    annotations = None if annotation_bytes is None else strict_json(annotation_bytes.decode("utf-8"))
    accepted, disposition = promote_candidates(candidates, source_name, version, inputs, annotations)
    for key, path in (("data", data), ("resolutions", resolutions)):
        if path is not None and sha256_file(path) != inputs[key]:
            raise ValidationError("source file changed during import")
    candidate_text = "".join(json.dumps(item.payload(), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                             for item in candidates)
    records_text = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                           for item in accepted)
    status_counts = Counter(item["status"] for item in disposition)
    report = {
        "schema_version": "1", "source": source_name, "dataset_version": version,
        "upstream_revision": source.revision, "role": source.role,
        "license": source.metadata["license"], "homepage": source.metadata["homepage"],
        "input_hashes": inputs, "statistics": statistics, "candidate_count": len(candidates),
        "accepted_count": len(accepted), "quarantined_count": status_counts["quarantined"],
        "excluded_count": status_counts["excluded"],
        "blocker_counts": dict(sorted(Counter(reason for item in candidates for reason in set(item.blockers)).items())),
        "note_counts": dict(sorted(Counter(reason for item in candidates for reason in set(item.notes)).items())),
        "annotation_sha256": None if annotation_bytes is None else sha256_bytes(annotation_bytes),
        "candidates_sha256": sha256_bytes(candidate_text.encode("utf-8")),
        "records_sha256": sha256_bytes(records_text.encode("utf-8")),
        "code": code_provenance(),
        "limitations": ["No automatic semantic cross-dataset grouping certification",
                        "Timestamp annotations are provenance assertions, not independently verified facts",
                        "Untimestamped summaries/background are omitted from model inputs",
                        "This is a data audit, not a forecasting benchmark result"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ingest-", dir=output.parent) as staging:
        stage = Path(staging) / "import"
        stage.mkdir()
        (stage / "audit.json").write_text(json_text(report), encoding="utf-8")
        (stage / "source.json").write_text(json_text(source.metadata), encoding="utf-8")
        (stage / "candidates.jsonl").write_text(candidate_text, encoding="utf-8")
        (stage / "records.jsonl").write_text(records_text, encoding="utf-8")
        (stage / "disposition.json").write_text(json_text(disposition), encoding="utf-8")
        (stage / "annotations.template.json").write_text(
            json_text(annotation_template(source_name, inputs, candidates)), encoding="utf-8")
        if annotation_bytes is not None:
            (stage / "annotations.applied.json").write_bytes(annotation_bytes)
        markdown = (
            f"# {source_name}: source audit\n\n"
            f"Dataset version: `{version}`\n\n"
            f"Native rows: {statistics['input_rows']}; candidates: {len(candidates)}; "
            f"accepted: {len(accepted)}; quarantined: {status_counts['quarantined']}; "
            f"excluded: {status_counts['excluded']}.\n\n"
            "Only records.jsonl is eligible for the evaluation loader. Candidates and annotation templates "
            "are audit artifacts, not model inputs.\n\n"
            "No missing provenance timestamps have been filled from close times or download times. "
            "See audit.json, disposition.json and annotations.template.json for the outstanding checks.\n"
        )
        (stage / "audit.md").write_text(markdown, encoding="utf-8")
        stage.rename(output)
    return report
