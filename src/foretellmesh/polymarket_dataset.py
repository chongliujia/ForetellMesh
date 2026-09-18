"""Public Polymarket observations, immutable replay and conservative resolution labels.

Gamma identities/rules, CLOB token mapping and books, and Data API resolution
states are independently archived. No trading/account endpoint is supported.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen

from .adapters import identifier
from .data import sha256_bytes, strict_json, validate_identities
from .evaluation import code_provenance, json_text
from .market_dataset import decimal_value, now
from .schema import ValidationError, fields, iso, nonempty, parse_record, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"
SOURCE = "foretellmesh_polymarket"
HEX = r"0x[0-9a-fA-F]{64}"
URL_PATTERN = re.compile(r"(?:https://gamma-api\.polymarket\.com/(?:events|markets)/[0-9]+|"
                         r"https://clob\.polymarket\.com/clob-markets/" + HEX + r"|"
                         r"https://clob\.polymarket\.com/book\?token_id=[0-9]+|"
                         r"https://data-api\.polymarket\.com/v2/resolutions\?condition=" + HEX + r")")


def public_get(url):
    if not URL_PATTERN.fullmatch(url):
        raise ValidationError("unsupported public Polymarket GET")
    with urlopen(Request(url, headers={"User-Agent": "ForetellMesh/0.1 public-data-research"}), timeout=25) as response:
        if response.url != url:
            raise ValidationError("unexpected API redirect")
        raw, date = response.read(8_000_001), response.headers.get("Date")
    if len(raw) > 8_000_000:
        raise ValidationError("response size limit")
    return raw, date


class Archive:
    def __init__(self, stage):
        self.stage, self.entries = stage, []
        (stage / "raw").mkdir()

    def get(self, url):
        if not URL_PATTERN.fullmatch(url):
            raise ValidationError("unsupported archive URL")
        start = now()
        raw, date = public_get(url)
        end = now()
        obj = strict_json(raw.decode())
        if not isinstance(obj, dict) or timestamp(start, "start") > timestamp(end, "end"):
            raise ValidationError("invalid response or clock")
        ref = {"url": url, "file": f"raw/{len(self.entries):04d}.json", "sha256": sha256_bytes(raw),
               "started_at": start, "completed_at": end, "server_date": date}
        (self.stage / ref["file"]).write_bytes(raw)
        self.entries.append(ref)
        return obj


def read_archive(root, kind):
    manifest = strict_json((root / "manifest.json").read_text())
    if (manifest.get("schema_version") != "1" or manifest.get("kind") != kind
            or canonical_hash(manifest["requests"]) != manifest["requests_sha256"]):
        raise ValidationError("invalid Polymarket archive manifest")
    responses = {}
    for ref in manifest["requests"]:
        path = (root / ref["file"]).resolve()
        if (not path.is_relative_to(root.resolve()) or not URL_PATTERN.fullmatch(ref["url"])
                or ref["url"] in responses):
            raise ValidationError("invalid/duplicate archive reference")
        raw = path.read_bytes()
        if sha256_bytes(raw) != ref["sha256"]:
            raise ValidationError("archive response hash mismatch")
        if timestamp(ref["started_at"], "start") > timestamp(ref["completed_at"], "end"):
            raise ValidationError("invalid request interval")
        obj = strict_json(raw.decode())
        if not isinstance(obj, dict):
            raise ValidationError("response must be an object")
        responses[ref["url"]] = (ref, obj)
    return manifest, responses


def config_value(value):
    fields(value, {"schema_version", "dataset_name", "event_ids", "max_markets",
                   "max_quote_age_seconds", "max_snapshot_seconds", "max_spread", "min_best_size"}, "Polymarket config")
    ids = value["event_ids"]
    if (value["schema_version"] != "1" or not isinstance(ids, list) or not 1 <= len(ids) <= 10
            or any(not isinstance(x, str) or not re.fullmatch(r"[0-9]+", x) for x in ids)
            or len(ids) != len(set(ids))):
        raise ValidationError("invalid event cohort")
    nonempty(value["dataset_name"], "dataset name")
    for key, limit in (("max_markets", 200), ("max_quote_age_seconds", 3600), ("max_snapshot_seconds", 600)):
        if type(value[key]) is not int or not 1 <= value[key] <= limit:
            raise ValidationError("invalid capture bound")
    if not 0 < decimal_value(value["max_spread"], "spread") < 1 or decimal_value(value["min_best_size"], "size") <= 0:
        raise ValidationError("invalid quote bounds")
    return value


def mapping(market):
    outcomes = strict_json(market["outcomes"]) if isinstance(market["outcomes"], str) else market["outcomes"]
    tokens = strict_json(market["clobTokenIds"]) if isinstance(market["clobTokenIds"], str) else market["clobTokenIds"]
    if (not isinstance(outcomes, list) or len(outcomes) != 2
            or any(not isinstance(x, str) for x in outcomes) or set(outcomes) != {"Yes", "No"}
            or not isinstance(tokens, list) or len(tokens) != 2
            or any(not isinstance(x, str) or not re.fullmatch(r"[0-9]+", x) for x in tokens) or len(set(tokens)) != 2
            or not re.fullmatch(HEX, market.get("conditionId", ""))
            or not re.fullmatch(r"[0-9]+", str(market.get("id", "")))):
        raise ValidationError("invalid binary outcome/token mapping")
    return dict(zip(outcomes, tokens)), outcomes.index("Yes")


def semantic_hash(market):
    tokens, yes_index = mapping(market)
    return canonical_hash({"tokens": tokens, "yes_index": yes_index, **{k: market.get(k) for k in (
        "id", "conditionId", "questionID", "question", "description", "resolutionSource", "endDate",
        "negRisk", "negRiskOther", "negRiskMarketID", "negRiskRequestID")}})


def resolution_row(obj, condition):
    rows = obj.get("data")
    if not isinstance(rows, list) or len(rows) > 1:
        raise ValidationError("ambiguous resolution response")
    if not rows:
        return None
    row = rows[0]
    if not isinstance(row, dict) or row.get("condition_id") != condition:
        raise ValidationError("resolution condition mismatch")
    nonempty(row.get("status"), "resolution status")
    return row


def book_quote(book, ref, t, condition, yes_token, config):
    if book.get("market") != condition or book.get("asset_id") != yes_token:
        raise ValidationError("book condition/token mismatch")
    raw_time = book.get("timestamp")
    if not isinstance(raw_time, str) or not re.fullmatch(r"[0-9]{13}", raw_time):
        raise ValidationError("expected millisecond book timestamp")
    observed = datetime.fromtimestamp(int(raw_time) / 1000, timezone.utc)
    received = timestamp(ref["completed_at"], "book received")
    if observed > received:
        raise ValidationError("future book timestamp")
    sides = []
    for side in ("bids", "asks"):
        if not isinstance(book.get(side), list):
            raise ValidationError("missing book side")
        levels = []
        for level in book[side]:
            price, size = decimal_value(level["price"], "price"), decimal_value(level["size"], "size")
            if not 0 <= price <= 1 or size <= 0:
                raise ValidationError("invalid book level")
            levels.append((price, size))
        sides.append(levels)
    quality = {"quote_age_seconds": (t - observed).total_seconds(), "best_bid": None, "best_ask": None,
               "bid_size": None, "ask_size": None, "spread": None, "issues": []}
    if not all(sides):
        quality["issues"].append("missing_book_side")
    else:
        # API response ordering is not relied on.
        bid, ask = max(x[0] for x in sides[0]), min(x[0] for x in sides[1])
        bid_size = sum(x[1] for x in sides[0] if x[0] == bid)
        ask_size = sum(x[1] for x in sides[1] if x[0] == ask)
        if bid > ask:
            raise ValidationError("crossed order book")
        quality.update(best_bid=float(bid), best_ask=float(ask), bid_size=float(bid_size),
                       ask_size=float(ask_size), spread=float(ask - bid))
        if not 0 < bid <= ask < 1:
            quality["issues"].append("boundary_quote")
        if ask - bid > decimal_value(config["max_spread"], "max spread"):
            quality["issues"].append("wide_spread")
        if min(bid_size, ask_size) < decimal_value(config["min_best_size"], "min size"):
            quality["issues"].append("thin_best_level")
    if quality["quote_age_seconds"] > config["max_quote_age_seconds"]:
        quality["issues"].append("stale_book")
    quote_data = None if quality["issues"] else {"probability": float((bid + ask) / 2),
                                               "observed_at": iso(observed), "available_at": ref["completed_at"]}
    return quote_data, quality


def observe(event, market, clob, book, resolution, refs, version, config):
    t = max(timestamp(r["completed_at"], "received") for r in refs.values())
    start = min(timestamp(refs[k]["started_at"], "started") for k in ("market", "clob", "book", "resolution"))
    if (t - start).total_seconds() > config["max_snapshot_seconds"]:
        raise ValidationError("snapshot capture too slow")
    tokens, yes_index = mapping(market)
    if (market.get("active") is not True or market.get("closed") is not False
            or market.get("acceptingOrders") is not True or market.get("enableOrderBook") is not True
            or market.get("negRiskOther") is True or event.get("closed") is not False):
        raise ValidationError("not an open supported market")
    if not timestamp(market["startDate"], "startDate") <= t < timestamp(market["endDate"], "endDate"):
        raise ValidationError("outside market interval")
    for obj in (event, market):
        for key in ("createdAt", "updatedAt"):
            if obj.get(key) and timestamp(obj[key], key) > t:
                raise ValidationError("future metadata")
    if not any(str(m.get("id")) == str(market["id"]) and m.get("conditionId") == market["conditionId"] for m in event["markets"]):
        raise ValidationError("event market mismatch")
    clob_tokens = clob.get("t", [])
    if (not isinstance(clob_tokens, list) or len(clob_tokens) != 2
            or [(x.get("o"), x.get("t")) for x in clob_tokens] != list(tokens.items())
            or clob.get("c") != market["conditionId"] or clob.get("ao") is not True):
        raise ValidationError("CLOB token/condition/status mismatch")
    state = resolution_row(resolution, market["conditionId"])
    if state and state["status"] not in ("initialized", "posed", "active"):
        raise ValidationError("resolution already proposed or unknown lifecycle")
    quote_data, quality = book_quote(book, refs["book"], t, market["conditionId"], tokens["Yes"], config)
    record = {"sample_id": identifier(SOURCE, str(market["id"]), iso(t)), "dataset_source": SOURCE,
              "dataset_version": version, "event_id": identifier("polymarket", str(market["id"])),
              "event_group_id": identifier("polymarket", str(event["id"])),
              "question": nonempty(market["question"], "question") + "\nResolution rules: " + nonempty(market["description"], "rules"),
              "observation_time": iso(t), "evidence": [], "market": quote_data, "label": None}
    parse_record(record)
    return {"record": record, "market_id": str(market["id"]), "condition_id": market["conditionId"],
            "parent_event_id": str(event["id"]), "tokens": tokens, "yes_index": yes_index,
            "semantic_sha256": semantic_hash(market), "refs": refs, "quote_quality": quality,
            "initial_resolution": state, "scheduled_close": market["endDate"],
            "resolution_context_review": "pending"}


def replay_capture(root):
    manifest, responses = read_archive(root, "polymarket_prospective_capture")
    config = config_value(manifest["config"])
    observations, exclusions = [], []
    seen = set()
    for event_id in config["event_ids"]:
        event_ref, event = responses[GAMMA + "/events/" + event_id]
        if str(event.get("id")) != event_id:
            raise ValidationError("event id mismatch")
        for discovery in event["markets"]:
            mid = str(discovery["id"])
            if mid in seen:
                raise ValidationError("duplicate market in event cohort")
            seen.add(mid)
            try:
                mref, market = responses[GAMMA + "/markets/" + mid]
                if str(market["id"]) != mid:
                    raise ValidationError("market id mismatch")
                tokens, _ = mapping(market)
                cref, clob = responses[CLOB + "/clob-markets/" + market["conditionId"]]
                bref, book = responses[CLOB + "/book?token_id=" + tokens["Yes"]]
                rref, resolution = responses[DATA + "/v2/resolutions?condition=" + market["conditionId"]]
                observations.append(observe(event, market, clob, book, resolution,
                    {"event": event_ref, "market": mref, "clob": cref, "book": bref, "resolution": rref},
                    "sha256:" + manifest["requests_sha256"], config))
            except (ValidationError, KeyError) as exc:
                exclusions.append({"event_id": event_id, "market_id": mid, "reason": str(exc)})
    if len(seen) > config["max_markets"]:
        raise ValidationError("cohort exceeds market budget")
    validate_identities([parse_record(o["record"]) for o in observations])
    return manifest, observations, exclusions


def collect_polymarket(config_path, output):
    if output.exists():
        raise ValidationError("capture output already exists")
    config = config_value(strict_json(config_path.read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".polymarket-", dir=output.parent) as tmp:
        stage = Path(tmp) / "capture"
        stage.mkdir()
        archive = Archive(stage)
        events = [archive.get(GAMMA + "/events/" + eid) for eid in config["event_ids"]]
        if sum(len(e["markets"]) for e in events) > config["max_markets"]:
            raise ValidationError("whole event cohort exceeds budget")
        for event in events:
            for discovery in event["markets"]:
                market = archive.get(GAMMA + "/markets/" + str(discovery["id"]))
                try:
                    tokens, _ = mapping(market)
                except (ValidationError, KeyError):
                    continue  # Replayed into exclusions, never silently dropped.
                archive.get(CLOB + "/clob-markets/" + market["conditionId"])
                archive.get(CLOB + "/book?token_id=" + tokens["Yes"])
                archive.get(DATA + "/v2/resolutions?condition=" + market["conditionId"])
        manifest = {"schema_version": "1", "kind": "polymarket_prospective_capture", "config": config,
                    "requests": archive.entries, "requests_sha256": canonical_hash(archive.entries), "code": code_provenance()}
        (stage / "manifest.json").write_text(json_text(manifest))
        _, observations, exclusions = replay_capture(stage)
        (stage / "observations.jsonl").write_text(jsonl(observations))
        (stage / "exclusions.json").write_text(json_text(exclusions))
        stage.rename(output)
    return {"observations": len(observations), "events": len({x["parent_event_id"] for x in observations}),
            "market_quotes": sum(x["record"]["market"] is not None for x in observations), "exclusions": exclusions}


def poll_polymarket_labels(capture, output):
    if output.exists():
        raise ValidationError("poll output already exists")
    manifest, observations, _ = replay_capture(capture)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".polymarket-poll-", dir=output.parent) as tmp:
        stage = Path(tmp) / "poll"
        stage.mkdir()
        archive = Archive(stage)
        for obs in observations:
            archive.get(GAMMA + "/markets/" + obs["market_id"])
            archive.get(DATA + "/v2/resolutions?condition=" + obs["condition_id"])
        report = {"schema_version": "1", "kind": "polymarket_label_poll",
                  "capture_requests_sha256": manifest["requests_sha256"], "requests": archive.entries,
                  "requests_sha256": canonical_hash(archive.entries), "code": code_provenance()}
        (stage / "manifest.json").write_text(json_text(report))
        stage.rename(output)
    return {"polled_markets": len(observations)}


def settlement(obs, market, state, available):
    for key in ("createdAt", "updatedAt"):
        if market.get(key) and timestamp(market[key], key) > timestamp(available, "available"):
            return None, "future_settlement_metadata"
    if semantic_hash(market) != obs["semantic_sha256"]:
        return None, "contract_changed_since_observation"
    if not state:
        return None, "resolution_state_missing"
    if state["status"] != "resolved":
        return None, "pending_resolution"
    if any(state.get(k) is not False for k in ("extended_review", "was_disputed")) or state.get("was_arbitrated") is True:
        return None, "disputed_or_unverified_resolution"
    # Gamma questionID need not equal the UMA backing question ID (neg-risk).
    # Bind condition ID and compare the archived resolution context instead.
    initial = obs["initial_resolution"]
    if (not initial or any(state.get(k) != initial.get(k) for k in ("question_id", "new_version_q"))):
        return None, "resolution_context_changed"
    payouts = state.get("payouts")
    if (not isinstance(payouts, list) or len(payouts) != 2 or any(type(x) is not int for x in payouts)
            or sorted(payouts) != [0, 1_000_000]):
        return None, "binary_payout_proof_missing"
    if (not state.get("resolved_at") or type(state.get("resolved_block")) is not int
            or state["resolved_block"] <= 0 or state.get("resolution_source") not in ("reported", "derived")):
        return None, "settlement_time_or_provenance_missing"
    resolved = timestamp(state["resolved_at"], "resolved_at")
    if not timestamp(obs["record"]["observation_time"], "observation") < resolved <= timestamp(available, "available"):
        return None, "invalid_settlement_timeline"
    return {"outcome": int(payouts[obs["yes_index"]] == 1_000_000), "resolution_time": iso(resolved),
            "available_at": available}, "resolved"


def build_polymarket_dataset(capture, polls, output):
    if output.exists():
        raise ValidationError("dataset output already exists")
    manifest, observations, exclusions = replay_capture(capture)
    history, poll_hashes = defaultdict(list), []
    for root in polls:
        poll, responses = read_archive(root, "polymarket_label_poll")
        if poll["capture_requests_sha256"] != manifest["requests_sha256"]:
            raise ValidationError("poll belongs to a different capture")
        poll_hashes.append(poll["requests_sha256"])
        for obs in observations:
            mref, market = responses[GAMMA + "/markets/" + obs["market_id"]]
            rref, resolution = responses[DATA + "/v2/resolutions?condition=" + obs["condition_id"]]
            available = max((mref["completed_at"], rref["completed_at"]), key=lambda x: timestamp(x, "poll"))
            if min(timestamp(mref["started_at"], "poll"), timestamp(rref["started_at"], "poll")) <= timestamp(obs["record"]["observation_time"], "observation"):
                raise ValidationError("label polling must follow observation")
            state = resolution_row(resolution, obs["condition_id"])
            history[obs["market_id"]].append((available, market, state, {"market": mref, "resolution": rref}))
    records, metadata = [], []
    for obs in observations:
        labels, status, proof = [], "pending_resolution", None
        for available, market, state, refs in sorted(history[obs["market_id"]], key=lambda x: timestamp(x[0], "poll")):
            label, status = settlement(obs, market, state, available)
            if label:
                labels.append((label, refs))
            elif labels:
                status = "settlement_reopened_or_unverifiable"
                break
            elif status != "pending_resolution":
                break
        else:
            if labels:
                if len({(x[0]["outcome"], x[0]["resolution_time"]) for x in labels}) != 1:
                    status = "conflicting_settlements"
                else:
                    status, proof = "resolved", labels[0][1]
        record = {**obs["record"], "label": labels[0][0] if status == "resolved" else None}
        parse_record(record)
        records.append(record)
        metadata.append({**{k: v for k, v in obs.items() if k != "record"}, "sample_id": record["sample_id"],
                         "status": status, "label_ref": proof, "semantic_group_review": "pending",
                         "benchmark_overlap_review": "pending"})
    validate_identities([parse_record(r) for r in records])
    artifacts = {"records.jsonl": jsonl(records), "metadata.jsonl": jsonl(metadata), "exclusions.json": json_text(exclusions)}
    report = {"schema_version": "1", "kind": "prospective_prediction_market_dataset", "dataset_source": SOURCE,
              "dataset_version": "sha256:" + manifest["requests_sha256"], "capture_requests_sha256": manifest["requests_sha256"],
              "poll_requests_sha256": sorted(poll_hashes), "record_count": len(records),
              "counts": dict(Counter(m["status"] for m in metadata)), "ready_for_sft": False, "ready_for_benchmark": False,
              "artifact_hashes": {k: sha256_bytes(v.encode()) for k, v in artifacts.items()}, "code": code_provenance()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".polymarket-build-", dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        for name, content in artifacts.items():
            (stage / name).write_text(content)
        (stage / "manifest.json").write_text(json_text(report))
        stage.rename(output)
    return report
