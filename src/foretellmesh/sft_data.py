"""Source-bound SFT targets, held-out exclusion, and model-visible text export."""

from collections import Counter
import json
from pathlib import Path
import re
import tempfile

from .adapters import forecastbench_candidates
from .config import load_config
from .data import load_records, question_key, sha256_bytes, strict_json
from .evaluation import code_provenance, json_text
from .ingestion import ingest_source
from .schema import ValidationError, fields, iso, nonempty, probability, timestamp
from .sources import load_source, verify_artifact
from .splits import split_records
from .synthetic_sft import SOURCE, VERSION, canonical_hash, render_case


PROMPT_VERSION = "forecast_json_sft_v1"
INSTRUCTION = (
    "Forecast the event using only the supplied information available by observation_time. "
    "Return one JSON object with exactly: event, probability, confidence, base_rate, "
    "key_evidence, counter_evidence, unknowns, observation_time. "
    "Copy question into event and copy observation_time. probability must be a number in [0,1]. "
    "confidence is low, medium, or high and describes certainty about the probability estimate, "
    "not whether the event is likely. base_rate is a supplied prior/base rate or null. "
    "key_evidence and counter_evidence are disjoint arrays of supplied evidence_id values. "
    "unknowns is an array of short strings. Use supplied tool calculations for arithmetic. "
    "Market prices are a baseline, not ground truth. Do not invent facts or sources. "
    "Use the question's language for unknowns."
)


def validate_response(value: dict, record) -> dict:
    fields(value, {"event", "probability", "confidence", "base_rate", "key_evidence",
                   "counter_evidence", "unknowns", "observation_time"}, "forecast response")
    if value["event"] != record.forecast_input.question:
        raise ValidationError("forecast event must exactly match the input question")
    if timestamp(value["observation_time"], "response observation_time") != record.forecast_input.observation_time:
        raise ValidationError("response observation_time differs from input")
    p = probability(value["probability"])
    base = None if value["base_rate"] is None else probability(value["base_rate"], "base_rate")
    if value["confidence"] not in ("low", "medium", "high"):
        raise ValidationError("invalid confidence")
    known = {item.evidence_id for item in record.forecast_input.evidence}
    for name in ("key_evidence", "counter_evidence", "unknowns"):
        items = value[name]
        if not isinstance(items, list) or any(not isinstance(s, str) or not s.strip() for s in items):
            raise ValidationError(f"{name} must contain nonempty strings")
        if len(items) != len(set(items)):
            raise ValidationError(f"duplicate {name}")
        if name != "unknowns" and set(items) - known:
            raise ValidationError("forecast cites unknown evidence")
    if set(value["key_evidence"]) & set(value["counter_evidence"]):
        raise ValidationError("evidence and counter-evidence references must be disjoint")
    return {**value, "probability": p, "base_rate": base,
            "observation_time": iso(record.forecast_input.observation_time)}


def model_text(record, response: dict) -> dict:
    response = validate_response(response, record)
    payload = record.forecast_input.to_payload()
    return {"prompt": "Task:\n" + INSTRUCTION + "\nInput JSON:\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\nForecast JSON:\n",
        "completion": json.dumps(response, ensure_ascii=False, sort_keys=True, allow_nan=False)}


def encode_sft_pair(tokenizer, pair: dict, max_length: int) -> dict:
    fields(pair, {"prompt", "completion"}, "SFT text pair")
    prefix = tokenizer.encode(nonempty(pair["prompt"], "prompt"), add_special_tokens=False)
    completion = tokenizer.encode(nonempty(pair["completion"], "completion"), add_special_tokens=False)
    if tokenizer.eos_token_id is None or not prefix or not completion:
        raise ValidationError("nonempty prompt/completion and EOS are required")
    completion.append(tokenizer.eos_token_id)
    if len(prefix) + len(completion) > max_length:
        raise ValidationError(f"SFT example exceeds {max_length} tokens; truncation is forbidden")
    return {"input_ids": prefix + completion, "attention_mask": [1] * (len(prefix) + len(completion)),
            "labels": [-100] * len(prefix) + completion}


def validate_target(row: dict, record, source_policy, proof_root: Path) -> dict:
    fields(row, {"schema_version", "sample_id", "input_sha256", "target", "provenance"}, "SFT target")
    if row["schema_version"] != "1" or row["sample_id"] != record.sample_id:
        raise ValidationError("SFT target identity/schema mismatch")
    if row["input_sha256"] != canonical_hash(record.forecast_input.to_payload()):
        raise ValidationError("SFT target input hash mismatch")
    response = validate_response(row["target"], record)
    proof = fields(row["provenance"], {"kind", "author", "issued_at", "available_at", "information_cutoff",
                                      "question_available_at", "artifact", "artifact_sha256", "reviewer", "review_notes"},
                   "target provenance")
    for key in ("author", "reviewer", "review_notes"):
        nonempty(proof[key], key)
    t = record.forecast_input.observation_time
    issued = timestamp(proof["issued_at"], "forecast issued_at")
    cutoff = timestamp(proof["information_cutoff"], "forecast information_cutoff")
    available = timestamp(proof["available_at"], "target available_at")
    question_time = timestamp(proof["question_available_at"], "question_available_at")
    if not question_time <= cutoff <= issued == t or available < issued:
        raise ValidationError("invalid target/question temporal provenance")
    if any(e.available_at > cutoff for e in record.forecast_input.evidence):
        raise ValidationError("target predates visible evidence")
    if record.forecast_input.market and record.forecast_input.market.available_at > cutoff:
        raise ValidationError("target predates market input")
    relative = Path(nonempty(proof["artifact"], "proof artifact"))
    root = proof_root.resolve()
    artifact = (root / relative).resolve()
    if relative.is_absolute() or not artifact.is_relative_to(root) or not artifact.is_file():
        raise ValidationError("proof artifact must be a local file within the target directory")
    content = artifact.read_bytes()
    if sha256_bytes(content) != proof["artifact_sha256"]:
        raise ValidationError("target proof artifact hash mismatch")
    if proof["kind"] == "synthetic_oracle":
        if not source_policy.synthetic or record.dataset_source != SOURCE or record.dataset_version != VERSION:
            raise ValidationError("synthetic oracle cannot supervise real records")
        expected, answer = render_case(strict_json(content.decode("utf-8")))
        from .schema import parse_record
        reference = parse_record(expected)
        if (record.forecast_input != reference.forecast_input or record.sample_id != reference.sample_id
                or record.event_id != reference.event_id or record.event_group_id != reference.event_group_id
                or response != answer):
            raise ValidationError("synthetic oracle replay disagrees with input or target")
    elif proof["kind"] == "historical_forecast":
        if source_policy.synthetic:
            raise ValidationError("historical forecast cannot be labeled synthetic")
        archive = fields(strict_json(content.decode("utf-8")), {
            "schema_version", "input_sha256", "target", "issued_at", "available_at", "information_cutoff",
            "question_available_at", "author", "source_uri"}, "normalized historical forecast archive")
        if archive["schema_version"] != "1" or archive["input_sha256"] != row["input_sha256"]:
            raise ValidationError("historical archive input/schema mismatch")
        if archive["target"] != row["target"] or any(archive[k] != proof[k] for k in (
                "issued_at", "available_at", "information_cutoff", "question_available_at", "author")):
            raise ValidationError("historical target disagrees with archived forecast")
        nonempty(archive["source_uri"], "historical source_uri")
        # For real records, review is a curator assertion tied to archived bytes.
        # An artifact hash cannot establish that a timestamp/content claim is true.
    else:
        raise ValidationError("unsupported supervision: outcome conversion and market copying are not SFT targets")
    return response


def jsonl(rows) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)


def make_forecastbench_index(registry: Path, questions: Path, resolutions: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError("held-out index already exists")
    source = load_source(registry, "forecastbench")
    verify_artifact(questions, source, "questions")
    verify_artifact(resolutions, source, "resolutions")
    candidates, _ = forecastbench_candidates(questions, resolutions)
    # Bind the parsed index to the same immutable inputs checked above.
    question_proof = verify_artifact(questions, source, "questions")
    resolution_proof = verify_artifact(resolutions, source, "resolutions")
    entries = [{"event_id": c.event_id, "event_group_id": c.suggested_event_group_id,
                "question": c.question} for c in candidates]
    result = {"schema_version": "1", "source": "forecastbench", "dataset_version": source.revision,
              "input_hashes": {"questions": question_proof["sha256"], "resolutions": resolution_proof["sha256"]},
              "entries": entries, "entries_sha256": canonical_hash(entries),
              "coverage": "all native questions and expanded horizons, including unresolved candidates"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json_text(result))
    return result


def build_sft(data: Path, targets: Path, config_path: Path, heldout_index: Path,
              output: Path, benchmark_review: Path | None = None) -> dict:
    if output.exists():
        raise ValidationError("SFT output already exists")
    config, config_hash = load_config(config_path)
    config_bytes = config_path.read_bytes()
    if sha256_bytes(config_bytes) != config_hash:
        raise ValidationError("SFT config changed while reading")
    records, data_hash = load_records(data)
    target_bytes = targets.read_bytes()
    by_id = {}
    known_ids = {r.sample_id for r in records}
    for line in target_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = strict_json(line)
        if not isinstance(row, dict):
            raise ValidationError("SFT target must be an object")
        sid = nonempty(row.get("sample_id"), "target sample_id")
        if sid in by_id or sid not in known_ids:
            raise ValidationError("duplicate or unknown SFT target ID")
        by_id[sid] = row
    index_bytes = heldout_index.read_bytes()
    index = strict_json(index_bytes.decode("utf-8"))
    fields(index, {"schema_version", "source", "dataset_version", "input_hashes", "entries", "entries_sha256", "coverage"},
           "held-out index")
    if (index["schema_version"] != "1" or index["source"] != "forecastbench" or not index["entries"]
            or canonical_hash(index["entries"]) != index["entries_sha256"]):
        raise ValidationError("invalid held-out index")
    for entry in index["entries"]:
        fields(entry, {"event_id", "event_group_id", "question"}, "held-out entry")
        for key, value in entry.items():
            nonempty(value, key)
    index_hash = sha256_bytes(index_bytes)
    reviews, review_hash, review_bytes = {}, None, None
    if benchmark_review is not None:
        raw = benchmark_review.read_bytes()
        review = fields(strict_json(raw.decode()), {"heldout_index_sha256", "groups"}, "benchmark review")
        if review["heldout_index_sha256"] != index_hash or not isinstance(review["groups"], dict):
            raise ValidationError("benchmark review does not match frozen index")
        reviews, review_hash = review["groups"], sha256_bytes(raw)
        review_bytes = raw
        if set(reviews) - {r.event_group_id for r in records}:
            raise ValidationError("benchmark review contains unknown event groups")
        for item in reviews.values():
            fields(item, {"decision", "reviewer", "notes"}, "event-group benchmark review")
            if item["decision"] not in ("clear", "overlap", "pending"):
                raise ValidationError("invalid benchmark review decision")
            nonempty(item["reviewer"], "reviewer")
            nonempty(item["notes"], "review notes")
    splits = split_records(records, config)
    heldout_ids = {e["event_id"] for e in index["entries"]}
    heldout_groups = {e["event_group_id"] for e in index["entries"]}
    heldout_questions = {question_key(e["question"]) for e in index["entries"]}
    def words(text):
        return set(re.findall(r"\w+", question_key(text)))
    token_sets = [words(q) for q in heldout_questions]
    blocked_groups = {}
    for record in records:
        group = record.event_group_id
        source = config.sources[record.dataset_source]
        key = question_key(record.forecast_input.question)
        if source.role == "eval_only" or record.dataset_source == "forecastbench":
            blocked_groups[group] = "heldout_source_group"
        elif group in heldout_groups or record.event_id in heldout_ids or key in heldout_questions:
            blocked_groups[group] = "heldout_event_or_exact_question_overlap"
        else:
            current = words(key)
            if any(len(current & other) / len(current | other) >= 0.9 for other in token_sets if current or other):
                blocked_groups[group] = "possible_heldout_near_duplicate"
        if reviews.get(group, {}).get("decision") == "overlap":
            blocked_groups[group] = "reviewed_heldout_overlap"
    model_rows = {name: [] for name in ("train", "validation", "test")}
    metadata, dispositions, proof_copies = [], list(splits.exclusions), {}
    for split, members in splits.partitions.items():
        deadline = {"train": config.validation_start, "validation": config.test_start, "test": config.evaluation_as_of}[split]
        for record in members:
            reason = blocked_groups.get(record.event_group_id)
            source = config.sources[record.dataset_source]
            if reason is None and not source.synthetic and reviews.get(record.event_group_id, {}).get("decision") != "clear":
                reason = "semantic_benchmark_review_required"
            row = by_id.get(record.sample_id)
            if reason is None and row is None:
                reason = "supervision_missing"
            if reason is not None:
                dispositions.append({"sample_id": record.sample_id, "event_group_id": record.event_group_id, "reason": reason})
                continue
            response = validate_target(row, record, source, targets.parent)
            if timestamp(row["provenance"]["available_at"], "target availability") > deadline:
                dispositions.append({"sample_id": record.sample_id, "event_group_id": record.event_group_id,
                                     "reason": "target_unavailable_at_split_cutoff"})
                continue
            model_rows[split].append(model_text(record, response))
            proof_bytes = (targets.parent / row["provenance"]["artifact"]).read_bytes()
            proof_hash = sha256_bytes(proof_bytes)
            if proof_hash != row["provenance"]["artifact_sha256"]:
                raise ValidationError("proof changed during SFT build")
            proof_name = "proofs/" + proof_hash + ".json"
            proof_copies[proof_name] = proof_bytes
            metadata.append({"sample_id": record.sample_id, "dataset_source": record.dataset_source,
                             "dataset_version": record.dataset_version, "event_id": record.event_id,
                             "event_group_id": record.event_group_id, "split": split,
                             "row_index": len(model_rows[split]) - 1, "input_sha256": row["input_sha256"],
                             "target_sha256": canonical_hash(response),
                             "provenance": {**row["provenance"], "artifact": proof_name},
                             "source_artifact": row["provenance"]["artifact"],
                             "observation_time": iso(record.forecast_input.observation_time),
                             "resolution_time": iso(record.label.resolution_time),
                             "label_available_at": iso(record.label.available_at), "outcome": record.label.outcome})
    artifacts = {f"{split}.jsonl": jsonl(rows) for split, rows in model_rows.items()}
    artifacts["metadata.jsonl"] = jsonl(metadata)
    artifacts["disposition.json"] = json_text(sorted(dispositions, key=lambda row: row["sample_id"]))
    review_template = {"heldout_index_sha256": index_hash, "groups": {
        group: {"decision": "pending", "reviewer": "", "notes": ""} for group in sorted({
            r.event_group_id for r in records if not config.sources[r.dataset_source].synthetic})}}
    artifacts["benchmark_review.template.json"] = json_text(review_template)
    report = {"schema_version": "1", "kind": "forecast_sft_dataset", "run_name": config.run_name,
              "prompt_version": PROMPT_VERSION, "split_version": config.split_version,
              "synthetic_only": all(config.sources[r.dataset_source].synthetic for r in records),
              "input_count": len(records), "counts": {s: len(v) for s, v in model_rows.items()},
              "ready_for_sft": bool(model_rows["train"] and model_rows["validation"]),
              "quarantined_or_excluded": len(dispositions),
              "reason_counts": dict(Counter(r["reason"] for r in dispositions)),
              "data_sha256": data_hash, "targets_sha256": sha256_bytes(target_bytes),
              "config_sha256": config_hash, "heldout_index_sha256": index_hash,
              "heldout_entry_count": len(index["entries"]), "benchmark_review_sha256": review_hash,
              "artifact_hashes": {**{name: sha256_bytes(content.encode()) for name, content in artifacts.items()},
                                  **{name: sha256_bytes(content) for name, content in proof_copies.items()}},
              "code": code_provenance(),
              "limitations": ["Lexical matching cannot certify semantic deduplication; real groups require curator review.",
                              "Real provenance annotations require external verification; hashes only bind supplied artifacts.",
                              "Synthetic warm-up is not evidence of real forecasting quality."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sft-build-", dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        for name, content in artifacts.items():
            (stage / name).write_text(content, encoding="utf-8")
        for name, content in proof_copies.items():
            (stage / name).parent.mkdir(exist_ok=True)
            (stage / name).write_bytes(content)
        (stage / "config.json").write_bytes(config_bytes)
        (stage / "heldout_index.json").write_bytes(index_bytes)
        if review_bytes is not None:
            (stage / "benchmark_review.applied.json").write_bytes(review_bytes)
        (stage / "manifest.json").write_text(json_text(report), encoding="utf-8")
        stage.rename(output)
    return report


def audit_prophet_sft(data: Path, registry: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError("Prophet SFT audit output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prophet-sft-", dir=output.parent) as tmp:
        stage = Path(tmp) / "audit"
        source_report = ingest_source(source_name="prophet_arena", registry=registry, data=data, output=stage)
        candidates = [strict_json(line) for line in (stage / "candidates.jsonl").read_text().splitlines()]
        reasons = ["question_snapshot_proof_missing", "historical_forecast_target_missing",
                   "exact_resolution_time_missing", "label_availability_proof_missing", "semantic_benchmark_review_required"]
        queue = [{"sample_id": c["sample_id"], "event_id": c["event_id"],
                  "event_group_id": c["suggested_event_group_id"], "native_locator": c["locator"],
                  "observation_time": c["observation_time"], "status": "quarantined",
                  "missing": reasons + c["blockers"], "target": None} for c in candidates]
        report = {"schema_version": "1", "kind": "prophet_sft_readiness_audit",
                  "dataset_version": source_report["dataset_version"], "input_hashes": source_report["input_hashes"],
                  "submission_count": source_report["statistics"]["input_rows"], "candidate_count": len(queue),
                  "ready_count": 0, "missing_counts": dict(Counter(k for row in queue for k in row["missing"])),
                  "untimestamped_source_entries": source_report["statistics"]["omitted_source_entries"],
                  "target_policy": "Neither market prices nor realized 0/1 outcomes are historical forecast targets.",
                  "code": code_provenance()}
        (stage / "sft_review_queue.jsonl").write_text(jsonl(queue), encoding="utf-8")
        (stage / "sft_readiness.json").write_text(json_text(report), encoding="utf-8")
        stage.rename(output)
    return report
