"""Verify a small ForecastBench/Manifold cohort against archived primary sources.

Current API data is used exclusively for evaluator-side resolution metadata.
Questions and forecast features come only from the pre-cutoff Git snapshot.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

from .adapters import forecastbench_candidates
from .data import sha256_bytes, strict_json
from .evaluation import code_provenance, json_text
from .ingestion import audit_candidates, ingest_source
from .schema import ValidationError, fields, iso, nonempty, timestamp
from .sources import load_source, verify_artifact


HF_API = "https://huggingface.co/api/datasets/forecastingresearch/forecastbench-datasets/commits/"
HF_DATA = "https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets/resolve/"
MANIFOLD_API = "https://api.manifold.markets/v0/market/"


def resolution_projection(raw: dict) -> dict:
    """Ignore all current prices, descriptions, usernames, and other live fields."""
    if not isinstance(raw, dict):
        raise ValidationError("Manifold resolution response must be an object")
    if raw.get("outcomeType") != "BINARY" or raw.get("isResolved") is not True:
        raise ValidationError("Manifold contract is not resolved binary")
    if raw.get("resolution") not in ("YES", "NO"):
        raise ValidationError("unsupported Manifold resolution")
    if type(raw.get("resolutionTime")) is not int or raw["resolutionTime"] <= 0:
        raise ValidationError("Manifold resolutionTime must be positive integer milliseconds")
    nonempty(raw.get("id"), "Manifold market id")
    return {key: raw[key] for key in ("id", "outcomeType", "isResolved", "resolution", "resolutionTime")}


def resolution_datetime(projection: dict) -> datetime:
    try:
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=projection["resolutionTime"])
    except OverflowError as exc:
        raise ValidationError("Manifold resolutionTime outside supported range") from exc


def load_plan(path: Path) -> tuple[dict, str]:
    content = path.read_bytes()
    plan = strict_json(content.decode("utf-8"))
    fields(plan, {"schema_version", "source_revision", "question_release_revision",
                  "question_publication_time", "label_publication_time", "observation_time",
                  "cohort", "event_groups", "grouping_notes", "artifacts"}, "provenance plan")
    if plan["schema_version"] != "1" or plan["cohort"] != "all_manifold_questions_in_round":
        raise ValidationError("unsupported provenance plan/cohort")
    for key in ("source_revision", "question_release_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(plan[key])):
            raise ValidationError("provenance revisions must be immutable commits")
    published = timestamp(plan["question_publication_time"], "question_publication_time")
    observed = timestamp(plan["observation_time"], "observation_time")
    labeled = timestamp(plan["label_publication_time"], "label_publication_time")
    if not published <= observed < labeled:
        raise ValidationError("require question publication <= observation < label publication")
    if not isinstance(plan["event_groups"], dict) or not isinstance(plan["artifacts"], dict):
        raise ValidationError("event_groups and artifacts must be objects")
    for group in plan["event_groups"].values():
        nonempty(group, "reviewed event group")
    nonempty(plan["grouping_notes"], "grouping_notes")
    purposes = []
    market_ids = []
    for name, artifact in plan["artifacts"].items():
        if Path(name).name != name or name in ("", ".", "..", "download_manifest.json", "plan.json"):
            raise ValidationError("proof artifact names must be plain filenames")
        fields(artifact, {"url", "sha256", "max_bytes", "purpose", "market_id"}, "proof artifact")
        purpose = artifact["purpose"]
        purposes.append(purpose)
        if not re.fullmatch(r"[0-9a-f]{64}", str(artifact["sha256"])):
            raise ValidationError("proof artifact requires SHA-256")
        if type(artifact["max_bytes"]) is not int or not 1 <= artifact["max_bytes"] <= 4_000_000:
            raise ValidationError("invalid proof artifact size limit")
        url = nonempty(artifact["url"], "proof artifact URL")
        if purpose == "question_snapshot":
            prefix = HF_DATA + plan["question_release_revision"] + "/datasets/question_sets/"
            if not url.startswith(prefix) or urlparse(url).query:
                raise ValidationError("question proof must reference the official pinned question snapshot")
        elif purpose in ("question_commit", "label_commit"):
            revision = plan["question_release_revision"] if purpose == "question_commit" else plan["source_revision"]
            if url != HF_API + revision + "?limit=1":
                raise ValidationError("commit proof URL does not match its declared revision")
        elif purpose == "market_resolution":
            market_id = nonempty(artifact["market_id"], "proof market id")
            if not re.fullmatch(r"[A-Za-z0-9]+", market_id) or url != MANIFOLD_API + market_id:
                raise ValidationError("invalid Manifold resolution proof URL")
            market_ids.append(market_id)
        else:
            raise ValidationError("unknown proof artifact purpose")
        if purpose != "market_resolution" and artifact["market_id"] is not None:
            raise ValidationError("non-market proof must not declare a market id")
    if any(purposes.count(purpose) != 1 for purpose in ("question_snapshot", "question_commit", "label_commit")):
        raise ValidationError("plan requires one question snapshot and one proof of each commit")
    if len(market_ids) != len(set(market_ids)) or set(market_ids) != set(plan["event_groups"]):
        raise ValidationError("resolution proofs and reviewed event groups must match exactly")
    return plan, sha256_bytes(content)


def validate_proof(content: bytes, artifact: dict) -> dict | list:
    if len(content) > artifact["max_bytes"]:
        raise ValidationError("proof artifact exceeds size limit")
    raw = strict_json(content.decode("utf-8"))
    if artifact["purpose"] == "market_resolution":
        raw = resolution_projection(raw)
        if raw["id"] != artifact["market_id"]:
            raise ValidationError("Manifold proof market id mismatch")
        # Hash only resolution fields. Unrelated mutable live fields cannot change
        # this evaluator proof or leak into a historical forecasting input.
        digest = sha256_bytes(json_text(raw).encode("utf-8"))
    else:
        digest = sha256_bytes(content)
    if digest != artifact["sha256"]:
        raise ValidationError("proof SHA-256 mismatch; do not silently accept changed source facts")
    return raw


def download_provenance(plan_path: Path, output: Path) -> dict:
    plan, plan_hash = load_plan(plan_path)
    if output.exists():
        raise ValidationError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".proof-fetch-", dir=output.parent) as temporary:
        stage = Path(temporary) / "proofs"
        stage.mkdir()
        downloads = {}
        for name, artifact in plan["artifacts"].items():
            with urlopen(artifact["url"], timeout=30) as response:
                content = response.read(artifact["max_bytes"] + 1)
            validate_proof(content, artifact)
            (stage / name).write_bytes(content)
            downloads[name] = {"url": artifact["url"], "raw_sha256": sha256_bytes(content),
                               "verified_sha256": artifact["sha256"],
                               "hash_scope": "resolution_fields" if artifact["purpose"] == "market_resolution" else "raw_bytes"}
        manifest = {"plan_sha256": plan_hash, "artifacts": downloads}
        (stage / "download_manifest.json").write_text(json_text(manifest), encoding="utf-8")
        (stage / "plan.json").write_text(json_text(plan), encoding="utf-8")
        stage.rename(output)
    return manifest


def verify_forecastbench_subset(
    *, plan_path: Path, registry: Path, data: Path, resolutions: Path, proofs: Path, output: Path,
) -> dict:
    if output.exists():
        raise ValidationError(f"output already exists: {output}")
    plan, plan_hash = load_plan(plan_path)
    source = load_source(registry, "forecastbench")
    if source.revision != plan["source_revision"] or source.role != "eval_only":
        raise ValidationError("provenance plan requires the pinned held-out ForecastBench source")
    question_info = verify_artifact(data, source, "questions")
    label_info = verify_artifact(resolutions, source, "resolutions")
    inputs = {"data": question_info["sha256"], "resolutions": label_info["sha256"]}
    verified, market_proofs, artifacts = {}, {}, {}
    for filename, artifact in plan["artifacts"].items():
        path = proofs / filename
        if path.stat().st_size > artifact["max_bytes"]:
            raise ValidationError("proof artifact exceeds size limit")
        content = path.read_bytes()
        parsed = validate_proof(content, artifact)
        purpose = artifact["purpose"]
        artifacts[filename] = {"url": artifact["url"], "verified_sha256": artifact["sha256"]}
        if purpose == "market_resolution":
            market_proofs[artifact["market_id"]] = parsed
        else:
            verified[purpose] = parsed
        if purpose == "question_snapshot" and sha256_bytes(content) != inputs["data"]:
            raise ValidationError("current question file differs from the pre-cutoff historical snapshot")
    for purpose, revision_key, time_key in (
        ("question_commit", "question_release_revision", "question_publication_time"),
        ("label_commit", "source_revision", "label_publication_time"),
    ):
        commits = verified[purpose]
        if not isinstance(commits, list) or len(commits) != 1 or not isinstance(commits[0], dict):
            raise ValidationError("expected a single official commit metadata record")
        commit = commits[0]
        if commit.get("id") != plan[revision_key] or timestamp(commit.get("date"), "commit date") != timestamp(plan[time_key], time_key):
            raise ValidationError("official commit metadata disagrees with declared provenance")
    candidates, statistics = forecastbench_candidates(data, resolutions)
    audit_candidates(candidates)
    cohort = sorted((item for item in candidates if item.context["source"] == "manifold"),
                    key=lambda item: item.sample_id)
    resolved_ids = {item.context["native_id"] for item in cohort if item.context["raw_resolved"] is True}
    if resolved_ids != set(market_proofs):
        raise ValidationError("plan must cover every resolved Manifold question in the pinned round")
    observed = timestamp(plan["observation_time"], "observation_time")
    label_available = timestamp(plan["label_publication_time"], "label_publication_time")
    annotations = {"schema_version": "1", "source": "forecastbench", "input_hashes": inputs, "entries": {}}
    checks = []
    for candidate in cohort:
        market_id = candidate.context["native_id"]
        reasons = list(candidate.blockers)
        resolved_at = None
        if market_id in market_proofs:
            result = market_proofs[market_id]
            resolved_at = resolution_datetime(result)
            if int(result["resolution"] == "YES") != candidate.outcome:
                reasons.append("platform_outcome_disagrees_with_benchmark")
            if resolved_at.date().isoformat() != candidate.context["resolution_date_hint"]:
                reasons.append("platform_resolution_date_disagrees_with_benchmark")
            if not observed < resolved_at <= label_available:
                reasons.append("platform_resolution_outside_verified_time_bounds")
        if not reasons:
            provenance = {
                "plan_sha256": plan_hash,
                "question_revision": plan["question_release_revision"],
                "label_revision": plan["source_revision"], "resolution_api": MANIFOLD_API + market_id,
                "label_availability_semantics": "proven_available_by_pinned_commit_not_first_publication",
            }
            annotations["entries"][candidate.sample_id] = {
                "decision": "include", "event_group_id": plan["event_groups"][market_id],
                "observation_time": iso(observed),
                "question_available_at": plan["question_publication_time"],
                "resolution_time": iso(resolved_at), "label_available_at": iso(label_available),
                "provenance": json_text(provenance).strip(),
            }
        checks.append({"sample_id": candidate.sample_id, "native_id": market_id,
                       "status": "verified" if not reasons else "quarantined", "reasons": reasons,
                       "benchmark_resolution_date": candidate.context["resolution_date_hint"],
                       "platform_resolution_time": None if resolved_at is None else iso(resolved_at)})
    report = {
        "purpose": "provenance_smoke_test_only", "plan_sha256": plan_hash,
        "input_hashes": inputs, "source_role": "eval_only", "statistics": statistics,
        "question_snapshot_identical_to_historical_release": True,
        "question_publication_time": plan["question_publication_time"],
        "observation_time": plan["observation_time"], "label_available_by": plan["label_publication_time"],
        "cohort_count": len(cohort), "resolved_in_benchmark_count": len(resolved_ids),
        "verified_count": len(annotations["entries"]), "checks": checks, "artifacts": artifacts,
        "grouping_notes": plan["grouping_notes"], "code": code_provenance(),
        "limitations": ["Selection uses source and label availability, not forecast scores",
                         "Tiny resolved subset is not a representative forecasting benchmark",
                         "Group review covers this cohort, not all cross-dataset semantic duplicates",
                         "Current platform responses supply labels only; never model-visible context",
                         "Publication timing relies on official repository commit metadata"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".provenance-", dir=output.parent) as temporary:
        stage = Path(temporary) / "verified"
        stage.mkdir()
        annotations_path = stage / "annotations.json"
        annotations_path.write_text(json_text(annotations), encoding="utf-8")
        imported = ingest_source(source_name="forecastbench", registry=registry, data=data, resolutions=resolutions,
                                 annotations_path=annotations_path, output=stage / "import")
        if imported["accepted_count"] != report["verified_count"]:
            raise ValidationError("verification/import count mismatch")
        (stage / "verification.json").write_text(json_text(report), encoding="utf-8")
        (stage / "plan.json").write_text(json_text(plan), encoding="utf-8")
        stage.rename(output)
    return report
