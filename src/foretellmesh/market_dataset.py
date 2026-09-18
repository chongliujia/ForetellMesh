"""Prospective Kalshi snapshots and append-only outcome polling, using public GETs.

Raw responses remain immutable. Derivation replays those responses offline; no
current metadata is backdated into a historical question or evidence snapshot.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import tempfile
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .adapters import identifier
from .data import sha256_bytes, strict_json, validate_identities
from .evaluation import code_provenance, json_text
from .schema import ValidationError, fields, iso, nonempty, parse_record, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash


API = "https://external-api.kalshi.com/trade-api/v2"
SOURCE = "foretellmesh_kalshi"
SEMANTICS = ("ticker", "event_ticker", "market_type", "title", "yes_sub_title", "no_sub_title",
             "rules_primary", "rules_secondary", "strike_type", "floor_strike", "cap_strike",
             "functional_strike", "custom_strike", "notional_value_dollars")


def now() -> str:
    return iso(datetime.now(timezone.utc))


def load_collection_config(path: Path) -> dict:
    value = fields(strict_json(path.read_text()), {
        "schema_version", "dataset_name", "series_tickers", "events_per_series", "max_markets",
        "horizon_days", "page_size", "max_pages_per_series"}, "collection config")
    if value["schema_version"] != "1":
        raise ValidationError("unsupported collection version")
    nonempty(value["dataset_name"], "dataset_name")
    series = value["series_tickers"]
    if (not isinstance(series, list) or not 1 <= len(series) <= 10
            or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z0-9-]+", s) for s in series)
            or len(set(series)) != len(series)):
        raise ValidationError("expected 1-10 unique series tickers")
    for key, upper in (("events_per_series", 10), ("max_markets", 200), ("horizon_days", 365),
                       ("page_size", 1000), ("max_pages_per_series", 10)):
        if type(value[key]) is not int or not 1 <= value[key] <= upper:
            raise ValidationError(f"invalid collection bound: {key}")
    return value


def public_get(url: str) -> tuple[bytes, str | None]:
    if not url.startswith(API + "/"):
        raise ValidationError("only Kalshi public API GETs are supported")
    request = Request(url, headers={"User-Agent": "ForetellMesh/0.1 public-data-research"}, method="GET")
    with urlopen(request, timeout=25) as response:
        raw = response.read(8_000_001)
        date = response.headers.get("Date")
    if len(raw) > 8_000_000:
        raise ValidationError("public response exceeds size limit")
    return raw, date


class Archive:
    def __init__(self, stage: Path):
        self.stage, self.entries = stage, []
        (stage / "raw").mkdir()

    def get(self, kind: str, path: str, params: dict | None = None) -> dict:
        url = API + path + ("?" + urlencode(params) if params else "")
        started = now()
        raw, server_date = public_get(url)
        completed = now()
        obj = strict_json(raw.decode("utf-8"))
        if not isinstance(obj, dict) or timestamp(completed, "completed") < timestamp(started, "started"):
            raise ValidationError("invalid response or backwards local clock")
        filename = f"raw/{len(self.entries):04d}.json"
        (self.stage / filename).write_bytes(raw)
        self.entries.append({"kind": kind, "url": url, "file": filename,
                             "sha256": sha256_bytes(raw), "started_at": started,
                             "completed_at": completed, "server_date": server_date})
        return obj


def read_archive(root: Path, kind: str) -> tuple[dict, list[tuple[dict, dict]]]:
    manifest = strict_json((root / "manifest.json").read_text())
    if manifest["schema_version"] != "1" or manifest["kind"] != kind:
        raise ValidationError("incorrect market archive kind")
    results = []
    if canonical_hash(manifest["requests"]) != manifest["requests_sha256"]:
        raise ValidationError("request manifest hash mismatch")
    for item in manifest["requests"]:
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or not item["url"].startswith(API + "/"):
            raise ValidationError("invalid market archive reference")
        raw = path.read_bytes()
        if sha256_bytes(raw) != item["sha256"]:
            raise ValidationError("market archive hash mismatch")
        if timestamp(item["started_at"], "started") > timestamp(item["completed_at"], "completed"):
            raise ValidationError("invalid request interval")
        results.append((item, strict_json(raw.decode())))
    return manifest, results


def semantic_hash(market: dict) -> str:
    return canonical_hash({key: market.get(key) for key in SEMANTICS})


def decimal_value(value, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValidationError(f"invalid {name}")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValidationError(f"invalid {name}") from exc
    if not number.is_finite():
        raise ValidationError(f"invalid {name}")
    return number


def observed_record(market: dict, event: dict, market_time: str, event_time: str, version: str) -> dict:
    t = max(timestamp(market_time, "market snapshot"), timestamp(event_time, "event snapshot"))
    if (market.get("status") not in ("active", "open") or market.get("result") not in (None, "")
            or market.get("settlement_ts") or market.get("is_provisional") is True):
        raise ValidationError("not an unresolved active market")
    if market.get("market_type") != "binary" or decimal_value(market.get("notional_value_dollars"), "notional") != 1:
        raise ValidationError("only binary one-dollar contracts are supported")
    if not timestamp(market["open_time"], "open") <= t < timestamp(market["close_time"], "scheduled close"):
        raise ValidationError("snapshot is outside the open trading interval")
    for key in ("created_time", "updated_time"):
        if market.get(key) and timestamp(market[key], key) > t:
            raise ValidationError("market metadata timestamp is in the future")
    ticker = nonempty(market["ticker"], "ticker")
    parent = nonempty(market["event_ticker"], "event_ticker")
    if event.get("event_ticker") != parent:
        raise ValidationError("event/market identity mismatch")
    # The archived rule text belongs to this observation; its original publication
    # date is not asserted. It is not a retrospective news evidence item.
    question = "\n".join([
        nonempty(event["title"], "event title"), market.get("title") or "",
        "Yes outcome: " + nonempty(market["yes_sub_title"], "yes outcome"),
        "Resolution rules: " + nonempty(market["rules_primary"], "rules_primary"),
        market.get("rules_secondary") or "",
    ]).strip()
    quote_data = None
    bid, ask = market.get("yes_bid_dollars"), market.get("yes_ask_dollars")
    if bid is not None and ask is not None:
        bid, ask = decimal_value(bid, "bid"), decimal_value(ask, "ask")
        if not 0 <= bid <= ask <= 1:
            raise ValidationError("invalid/crossed dollar quotes")
        # 0/1 can denote missing sides. Do not silently turn them into a 0.5 quote.
        if 0 < bid <= ask < 1:
            quote_data = {"probability": float((bid + ask) / 2),
                          "observed_at": market_time, "available_at": market_time}
    record = {"sample_id": identifier(SOURCE, ticker, iso(t)), "dataset_source": SOURCE,
              "dataset_version": version, "event_id": identifier("kalshi", ticker),
              "event_group_id": identifier("kalshi", parent), "question": question,
              "observation_time": iso(t), "evidence": [], "market": quote_data, "label": None}
    parse_record(record)
    return record


def replay_capture(root: Path) -> tuple[dict, list[dict]]:
    manifest, responses = read_archive(root, "kalshi_prospective_capture")
    markets, events = {}, {}
    for ref, obj in responses:
        if ref["kind"] == "markets":
            for row in obj["markets"]:
                ticker = row["ticker"]
                if ticker in markets:
                    raise ValidationError("duplicate ticker across discovery pages")
                markets[ticker] = (ref, row)
        elif ref["kind"] == "event":
            events[obj["event"]["event_ticker"]] = (ref, obj["event"])
    observations = []
    for ticker in manifest["selected_tickers"]:
        ref, market = markets[ticker]
        event_ref, event = events[market["event_ticker"]]
        record = observed_record(market, event, ref["completed_at"], event_ref["completed_at"],
                                 "sha256:" + manifest["requests_sha256"])
        observations.append({"record": record, "ticker": ticker, "semantic_sha256": semantic_hash(market),
                             "market_ref": ref, "event_ref": event_ref, "category": event.get("category"),
                             "series_ticker": event.get("series_ticker"), "scheduled_close": market["close_time"]})
    validate_identities([parse_record(row["record"]) for row in observations])
    return manifest, observations


def collect_markets(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError("capture output already exists")
    config = load_collection_config(config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".market-capture-", dir=output.parent) as tmp:
        stage = Path(tmp) / "capture"
        stage.mkdir()
        archive = Archive(stage)
        started = now()
        start = timestamp(started, "run start")
        selected, exclusions, pages = [], [], {}
        for series in config["series_tickers"]:
            archive.get("series", "/series/" + quote(series, safe=""))
            cursor, seen_cursors, by_event = "", set(), defaultdict(list)
            for page in range(config["max_pages_per_series"]):
                params = {"series_ticker": series, "status": "open", "limit": config["page_size"],
                          "min_close_ts": int(start.timestamp()),
                          "max_close_ts": int((start + timedelta(days=config["horizon_days"])).timestamp())}
                if cursor:
                    params["cursor"] = cursor
                obj = archive.get("markets", "/markets", params)
                for market in obj["markets"]:
                    by_event[market["event_ticker"]].append(market)
                cursor = obj.get("cursor") or ""
                if not cursor:
                    break
                if cursor in seen_cursors:
                    raise ValidationError("repeated pagination cursor")
                seen_cursors.add(cursor)
            pages[series] = {"pages": page + 1, "truncated": bool(cursor), "discovered_markets": sum(map(len, by_event.values()))}
            if cursor:
                # We cannot certify a whole native event with a truncated listing.
                exclusions.append({"series": series, "reason": "page_limit_reached"})
                continue
            groups = sorted(by_event.values(), key=lambda rows: (min(timestamp(r["close_time"], "close") for r in rows), rows[0]["event_ticker"]))
            accepted_events = 0
            for members in groups:
                if accepted_events == config["events_per_series"]:
                    break
                if len(selected) + len(members) > config["max_markets"]:
                    exclusions.append({"event": members[0]["event_ticker"], "reason": "whole_event_exceeds_budget"})
                    continue
                event = archive.get("event", "/events/" + quote(members[0]["event_ticker"], safe=""))["event"]
                event_time = archive.entries[-1]["completed_at"]
                try:
                    if event.get("series_ticker") != series:
                        raise ValidationError("unexpected event series")
                    for market in members:
                        # Final replay uses the exact market response timestamp.
                        observed_record(market, event, event_time, event_time, "preflight")
                except (ValidationError, KeyError) as exc:
                    exclusions.append({"event": event["event_ticker"], "reason": str(exc)})
                    continue
                selected.extend(m["ticker"] for m in members)
                accepted_events += 1
        manifest = {"schema_version": "1", "kind": "kalshi_prospective_capture", "config": config,
                    "started_at": started, "completed_at": now(), "requests": archive.entries,
                    "requests_sha256": canonical_hash(archive.entries), "selected_tickers": sorted(selected),
                    "pages": pages, "exclusions": exclusions, "code": code_provenance()}
        (stage / "manifest.json").write_text(json_text(manifest), encoding="utf-8")
        _, observations = replay_capture(stage)
        (stage / "observations.jsonl").write_text(jsonl(observations), encoding="utf-8")
        stage.rename(output)
    return {"observations": len(observations), "events": len({o["record"]["event_group_id"] for o in observations}),
            "market_quotes": sum(o["record"]["market"] is not None for o in observations), "exclusions": exclusions}


def poll_market_labels(capture: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError("label poll output already exists")
    manifest, observations = replay_capture(capture)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".market-labels-", dir=output.parent) as tmp:
        stage = Path(tmp) / "labels"
        stage.mkdir()
        archive = Archive(stage)
        for row in observations:
            archive.get("market", "/markets/" + quote(row["ticker"], safe=""))
        report = {"schema_version": "1", "kind": "kalshi_label_poll",
                  "capture_requests_sha256": manifest["requests_sha256"], "requests": archive.entries,
                  "requests_sha256": canonical_hash(archive.entries), "code": code_provenance()}
        (stage / "manifest.json").write_text(json_text(report), encoding="utf-8")
        stage.rename(output)
    return report


def build_market_dataset(capture: Path, polls: list[Path], output: Path) -> dict:
    if output.exists():
        raise ValidationError("market dataset output already exists")
    manifest, observations = replay_capture(capture)
    by_ticker, poll_hashes = defaultdict(list), []
    for root in polls:
        poll, responses = read_archive(root, "kalshi_label_poll")
        if poll["capture_requests_sha256"] != manifest["requests_sha256"]:
            raise ValidationError("label poll belongs to a different capture")
        poll_hashes.append(poll["requests_sha256"])
        for ref, obj in responses:
            market = obj["market"]
            if market["ticker"] not in manifest["selected_tickers"]:
                raise ValidationError("label poll includes unknown ticker")
            by_ticker[market["ticker"]].append((ref, market))
    records, metadata, disposition = [], [], []
    for observation in observations:
        record = observation["record"]
        t = timestamp(record["observation_time"], "observation")
        labels, reason, proof = [], "pending_resolution", None
        history = sorted(by_ticker[observation["ticker"]], key=lambda x: timestamp(x[0]["completed_at"], "poll time"))
        for ref, market in history:
            available = timestamp(ref["completed_at"], "label observed")
            if available <= t:
                raise ValidationError("label poll must follow its observation")
            if semantic_hash(market) != observation["semantic_sha256"]:
                reason = "contract_changed_since_observation"
                break
            if market.get("status") not in ("settled", "finalized"):
                if labels:
                    reason = "settlement_reopened"
                    break
                continue
            if market.get("result") not in ("yes", "no") or market.get("is_provisional") is True:
                reason = "non_binary_or_provisional_settlement"
                break
            if not market.get("settlement_ts"):
                reason = "settlement_timestamp_missing"
                break
            resolution = timestamp(market["settlement_ts"], "settlement timestamp")
            if not t < resolution <= available:
                reason = "invalid_settlement_timeline"
                break
            outcome = int(market["result"] == "yes")
            if (market.get("settlement_value_dollars") is not None
                    and decimal_value(market["settlement_value_dollars"], "settlement value") != outcome):
                reason = "settlement_value_conflict"
                break
            labels.append(({"outcome": outcome, "resolution_time": iso(resolution), "available_at": iso(available)}, ref))
        else:
            if labels:
                if len({(x[0]["outcome"], x[0]["resolution_time"]) for x in labels}) > 1:
                    reason = "conflicting_settlements"
                else:
                    label, proof = min(labels, key=lambda x: timestamp(x[0]["available_at"], "label availability"))
                    record = {**record, "label": label}
                    reason = "resolved"
        # Even failed settlement review leaves the original unlabeled observation
        # intact. Nothing after T is ever copied into the model-visible payload.
        parse_record(record)
        records.append(record)
        metadata.append({**{k: v for k, v in observation.items() if k != "record"},
                         "sample_id": record["sample_id"], "status": reason,
                         "label_ref": proof if reason == "resolved" else None,
                         "semantic_group_review": "pending", "benchmark_overlap_review": "pending"})
        disposition.append({"sample_id": record["sample_id"], "status": reason})
    validate_identities([parse_record(row) for row in records])
    report = {"schema_version": "1", "kind": "prospective_prediction_market_dataset",
              "dataset_source": SOURCE, "dataset_version": "sha256:" + manifest["requests_sha256"],
              "capture_requests_sha256": manifest["requests_sha256"], "poll_requests_sha256": sorted(poll_hashes),
              "counts": dict(Counter(r["status"] for r in disposition)), "record_count": len(records),
              "ready_for_sft": False, "ready_for_benchmark": False, "code": code_provenance(),
              "limitations": ["Market/rule snapshots only; external news evidence has not been collected.",
                              "No SFT teacher targets; outcomes are labels for proper scoring, not answer probabilities.",
                              "Semantic event grouping, benchmark overlap and chronological splits require a release review.",
                              "Repeated captures form quote history; no retrospective trade reconstruction."]}
    artifacts = {"records.jsonl": jsonl(records), "metadata.jsonl": jsonl(metadata), "disposition.json": json_text(disposition)}
    report["artifact_hashes"] = {k: sha256_bytes(v.encode()) for k, v in artifacts.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".market-build-", dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        for name, content in artifacts.items():
            (stage / name).write_text(content, encoding="utf-8")
        (stage / "manifest.json").write_text(json_text(report), encoding="utf-8")
        stage.rename(output)
    return report
