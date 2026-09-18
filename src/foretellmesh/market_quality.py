"""Auditable market staging report; deliberately does not issue training releases."""
from collections import Counter
from pathlib import Path
import re
import tempfile

from .data import question_key, sha256_bytes, strict_json, validate_identities
from .evaluation import code_provenance, json_text
from .schema import ValidationError, fields, nonempty, parse_record
from .sft_data import jsonl
from .synthetic_sft import canonical_hash


def word_set(text):
    return set(re.findall(r"\w+", question_key(text)))


def audit_market_quality(datasets: list[Path], groups_path: Path, heldout_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError("quality output already exists")
    group_bytes, heldout_bytes = groups_path.read_bytes(), heldout_path.read_bytes()
    group_plan, index = strict_json(group_bytes.decode()), strict_json(heldout_bytes.decode())
    fields(group_plan, {"schema_version", "version", "groups"}, "event group plan")
    if group_plan["schema_version"] != "1" or not isinstance(group_plan["groups"], list):
        raise ValidationError("invalid group plan")
    if (index.get("source") != "forecastbench" or index.get("schema_version") != "1"
            or not index.get("entries") or canonical_hash(index["entries"]) != index.get("entries_sha256")):
        raise ValidationError("invalid held-out index")
    lookup, canonical_groups = {}, set()
    for group in group_plan["groups"]:
        fields(group, {"event_group_id", "native_event_groups", "rationale"}, "group")
        canonical = nonempty(group["event_group_id"], "canonical group")
        nonempty(group["rationale"], "group rationale")
        if canonical in canonical_groups or not isinstance(group["native_event_groups"], list) or not group["native_event_groups"]:
            raise ValidationError("duplicate/empty canonical group")
        canonical_groups.add(canonical)
        for native in group["native_event_groups"]:
            nonempty(native, "native event group")
            if native in lookup:
                raise ValidationError("native event assigned to multiple groups")
            lookup[native] = canonical
    heldout_questions = {question_key(e["question"]) for e in index["entries"]}
    heldout_ids = {e["event_id"] for e in index["entries"]}
    # Lexical matches are review candidates only. A non-match is never clearance.
    index_tokens = [(e, word_set(e["question"])) for e in index["entries"]]
    records, rows, inputs = [], [], []
    for root in datasets:
        manifest_bytes = (root / "manifest.json").read_bytes()
        manifest = strict_json(manifest_bytes.decode())
        if manifest.get("kind") != "prospective_prediction_market_dataset" or manifest.get("schema_version") != "1":
            raise ValidationError("unsupported market bundle")
        source = manifest.get("dataset_source")
        if source not in ("foretellmesh_kalshi", "foretellmesh_polymarket"):
            raise ValidationError("unsupported market source")
        contents = {}
        for name, digest in manifest["artifact_hashes"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValidationError("unsafe bundle path")
            raw = path.read_bytes()
            if sha256_bytes(raw) != digest:
                raise ValidationError("bundle artifact hash mismatch")
            contents[name] = raw.decode()
        raw_records = [strict_json(x) for x in contents["records.jsonl"].splitlines() if x.strip()]
        metadata = [strict_json(x) for x in contents["metadata.jsonl"].splitlines() if x.strip()]
        by_id = {m["sample_id"]: m for m in metadata}
        if (len(raw_records) != manifest["record_count"] or len(by_id) != len(metadata)
                or set(by_id) != {r["sample_id"] for r in raw_records}
                or dict(Counter(m["status"] for m in metadata)) != manifest["counts"]):
            raise ValidationError("bundle metadata/count mismatch")
        for raw in raw_records:
            record = parse_record(raw)
            if record.dataset_source != source or record.dataset_version != manifest["dataset_version"]:
                raise ValidationError("bundle source/version mismatch")
            meta = by_id[raw["sample_id"]]
            if (raw["label"] is not None) != (meta["status"] == "resolved"):
                raise ValidationError("label disposition mismatch")
            native = raw["event_group_id"]
            canonical = lookup.get(native, native)
            records.append({**raw, "event_group_id": canonical})
            issues = ["release_calendar_review_pending", "semantic_benchmark_review_pending",
                      "chronological_split_not_frozen", "sft_supervision_missing"]
            if native not in lookup:
                issues.append("cross_platform_group_unmapped")
            if not raw["evidence"]:
                issues.append("external_evidence_missing")
            if not raw["label"]:
                issues.append("verified_outcome_missing")
            if not raw["market"]:
                issues.append("usable_market_baseline_missing")
            quality = meta.get("quote_quality")
            if quality is None:
                issues.append("quote_freshness_depth_unverified")
            else:
                issues.extend(quality["issues"])
            if source == "foretellmesh_polymarket":
                issues.append("resolution_context_review_pending")
            exact = question_key(raw["question"]) in heldout_questions or raw["event_id"] in heldout_ids
            if exact:
                issues.append("exact_heldout_overlap")
            tokens = word_set(raw["question"])
            scored = sorted(((len(tokens & other) / len(tokens | other) if tokens | other else 0, e)
                             for e, other in index_tokens), key=lambda x: (-x[0], x[1]["event_id"]))
            candidates = [{"event_id": e["event_id"], "question": e["question"], "jaccard": round(score, 6)}
                          for score, e in scored[:3] if score >= .15]
            rows.append({"sample_id": raw["sample_id"], "dataset_source": source, "native_event_group_id": native,
                         "event_group_id": canonical, "status": meta["status"], "quote_quality": quality,
                         "exact_heldout_overlap": exact, "heldout_review_candidates": candidates,
                         "blockers": sorted(set(issues)), "ready_for_sft": False, "ready_for_benchmark": False})
        inputs.append({"dataset_source": source, "manifest_sha256": sha256_bytes(manifest_bytes),
                       "artifact_hashes": manifest["artifact_hashes"], "capture_requests_sha256": manifest["capture_requests_sha256"]})
    validate_identities([parse_record(r) for r in records])
    records.sort(key=lambda r: (r["observation_time"], r["sample_id"]))
    rows.sort(key=lambda r: r["sample_id"])
    artifacts = {"grouped_staging_records.jsonl": jsonl(records), "review_queue.jsonl": jsonl(rows),
                 "group_plan.json": json_text(group_plan)}
    counts = {"records": len(records), "sources": dict(Counter(r["dataset_source"] for r in records)),
              "native_groups": len({r["native_event_group_id"] for r in rows}),
              "split_groups": len({r["event_group_id"] for r in records}),
              "market_quotes": sum(r["market"] is not None for r in records),
              "quotes_passing_freshness_spread_depth": sum(
                  r["quote_quality"] is not None and not r["quote_quality"]["issues"] for r in rows),
              "verified_outcomes": sum(r["label"] is not None for r in records),
              "with_external_evidence": sum(bool(r["evidence"]) for r in records),
              "exact_heldout_overlaps": sum(r["exact_heldout_overlap"] for r in rows),
              "ready_for_sft": 0, "ready_for_benchmark": 0,
              "blockers": dict(Counter(issue for r in rows for issue in r["blockers"]))}
    report = {"schema_version": "1", "kind": "prediction_market_quality_audit", "inputs": inputs,
              "group_plan_sha256": sha256_bytes(group_bytes), "heldout_index_sha256": sha256_bytes(heldout_bytes),
              "heldout_entry_count": len(index["entries"]), "counts": counts, "code": code_provenance(),
              "release_policy": "staging_only; all semantic, evidence, supervision and split reviews remain required",
              "artifact_hashes": {k: sha256_bytes(v.encode()) for k, v in artifacts.items()}}
    artifacts["report.json"] = json_text(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".market-quality-", dir=output.parent) as tmp:
        stage = Path(tmp) / "quality"
        stage.mkdir()
        for name, content in artifacts.items():
            (stage / name).write_text(content)
        stage.rename(output)
    return report
