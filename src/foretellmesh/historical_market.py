"""Bounded historical FOMC replay with explicit gaps and isolated outcome ledgers.

Historical fetch times are never relabeled as publication times. Only dated
primary-source releases and decoded initialization receipts establish historical
text provenance. Current market descriptions alone cannot certify an old input.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
import re
import tempfile
from urllib.parse import urlencode

from .adapters import identifier
from .data import sha256_bytes, strict_json, validate_identities
from .evaluation import code_provenance, json_text
from .historical_sources import Archive, allowed_url, fed_statement, initialized_question, normalize, read_archive
from .market_dataset import API as KALSHI, decimal_value, now
from .polymarket_dataset import GAMMA, CLOB, DATA, mapping
from .schema import ValidationError, fields, iso, nonempty, parse_record, probability, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash

SOURCE_PREFIX = "foretellmesh_historical_"


def load_plan(value: dict) -> dict:
    fields(value, {"schema_version", "dataset_name", "max_markets_per_platform_event", "max_price_pages",
                   "price_window_hours", "max_price_age_seconds", "polymarket_bucket_seconds", "kalshi_period_minutes",
                   "observation_days_before_release", "cohort_policy", "events"}, "historical collection plan")
    if value["schema_version"] != "1":
        raise ValidationError("unsupported historical plan")
    for key in ("dataset_name", "cohort_policy"):
        nonempty(value[key], key)
    for key, upper in (("max_markets_per_platform_event", 100), ("max_price_pages", 10),
                       ("price_window_hours", 24), ("max_price_age_seconds", 86400)):
        if type(value[key]) is not int or not 1 <= value[key] <= upper:
            raise ValidationError("invalid historical collection bound")
    if type(value["polymarket_bucket_seconds"]) is not int or value["polymarket_bucket_seconds"] not in (300, 1800, 10800, 43200):
        raise ValidationError("invalid historical Polymarket price grain")
    if type(value["kalshi_period_minutes"]) is not int or value["kalshi_period_minutes"] not in (1, 60, 1440):
        raise ValidationError("invalid Kalshi price grain")
    offsets = value["observation_days_before_release"]
    if (not isinstance(offsets, list) or not 1 <= len(offsets) <= 5
            or any(type(x) is not int or not 1 <= x <= 30 for x in offsets) or len(set(offsets)) != len(offsets)):
        raise ValidationError("invalid observation offsets")
    events = value["events"]
    if not isinstance(events, list) or not 1 <= len(events) <= 20:
        raise ValidationError("invalid historical cohort")
    seen = set()
    for event in events:
        fields(event, {"event_group_id", "polymarket_event_id", "kalshi_event_ticker", "release_time", "result_url", "evidence"}, "historical event")
        if (not re.fullmatch(r"macro:us:fomc:\d{4}-\d{2}", event["event_group_id"])
                or not re.fullmatch(r"\d+", event["polymarket_event_id"])
                or not re.fullmatch(r"KXFED-\d{2}[A-Z]{3}", event["kalshi_event_ticker"])):
            raise ValidationError("unsupported historical FOMC identity")
        for kind in ("event_group_id", "polymarket_event_id", "kalshi_event_ticker"):
            key = (kind, event[kind])
            if key in seen:
                raise ValidationError("duplicate event cohort identity")
            seen.add(key)
        release = timestamp(event["release_time"], "release")
        if not allowed_url(event["result_url"]) or "federalreserve.gov/" not in event["result_url"]:
            raise ValidationError("invalid official outcome source")
        if not isinstance(event["evidence"], list) or not 1 <= len(event["evidence"]) <= 10:
            raise ValidationError("invalid evidence source plan")
        for item in event["evidence"]:
            fields(item, {"url", "published_at"}, "historical evidence plan")
            if not allowed_url(item["url"]) or "federalreserve.gov/" not in item["url"]:
                raise ValidationError("unsupported historical evidence source")
            if not timestamp(item["published_at"], "publication") < release:
                raise ValidationError("outcome-time evidence declared as forecast context")
    return value


def observation_times(event: dict, plan: dict) -> list[datetime]:
    release = timestamp(event["release_time"], "release")
    return sorted(release - timedelta(days=d) for d in plan["observation_days_before_release"])


def poly_price_url(token: str, t: datetime, plan: dict, cursor: str | None = None) -> str:
    params = {"token_id": token, "start": int((t - timedelta(hours=plan["price_window_hours"])).timestamp()),
              "end": int(t.timestamp()) + 1, "bucket_seconds": plan["polymarket_bucket_seconds"], "limit": 1000}
    if cursor:
        params["cursor"] = cursor
    return DATA + "/v2/prices-history?" + urlencode(params)


def kalshi_price_url(ticker: str, t: datetime, plan: dict, historical: bool) -> str:
    path = "/historical/markets/" if historical else "/series/KXFED/markets/"
    return KALSHI + path + ticker + "/candlesticks?" + urlencode({
        "start_ts": int((t - timedelta(hours=plan["price_window_hours"])).timestamp()),
        "end_ts": int(t.timestamp()), "period_interval": plan["kalshi_period_minutes"]})


def capture_historical_markets(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError("historical capture output already exists")
    plan = load_plan(strict_json(config_path.read_text()))
    if any(timestamp(e["release_time"], "release") >= timestamp(now(), "now") for e in plan["events"]):
        raise ValidationError("historical cohort includes a future event")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".historical-capture-", dir=output.parent) as tmp:
        stage = Path(tmp) / "capture"
        stage.mkdir()
        archive = Archive(stage)
        cutoff = archive.get(KALSHI + "/historical/cutoff")
        archive.get(KALSHI + "/series/KXFED")
        for path in ("contract_terms/FED.pdf", "regulatory/product-certifications/FED.pdf"):
            archive.get("https://assets.kalshi.com/" + path, json_body=False)
        for event in plan["events"]:
            archive.get(event["result_url"], json_body=False)
            for item in event["evidence"]:
                archive.get(item["url"], json_body=False)
            pevent = archive.get(GAMMA + "/events/" + event["polymarket_event_id"])
            kobj = archive.get(KALSHI + "/events/" + event["kalshi_event_ticker"])
            resolutions = archive.get(DATA + "/v2/resolutions?event_id=" + event["polymarket_event_id"])
            pm = [] if pevent is None else pevent.get("markets", [])
            km = [] if kobj is None else kobj.get("markets", [])
            if max(len(pm), len(km)) > plan["max_markets_per_platform_event"]:
                raise ValidationError("whole historical event exceeds market budget")
            resolution_by_condition = {r["condition_id"]: r for r in (resolutions or {}).get("data", [])}
            for market in pm:
                try:
                    tokens, _ = mapping(market)
                except (ValidationError, KeyError, TypeError):
                    continue
                archive.get(CLOB + "/clob-markets/" + market["conditionId"])
                state = resolution_by_condition.get(market["conditionId"], {})
                tx = state.get("transaction_hash", "")
                if re.fullmatch(r"0x[0-9a-fA-F]{64}", tx):
                    archive.get("https://polygonscan.com/tx/" + tx, json_body=False)
                for t in observation_times(event, plan):
                    cursor, seen = None, set()
                    for _ in range(plan["max_price_pages"]):
                        obj = archive.get(poly_price_url(tokens["Yes"], t, plan, cursor))
                        if obj is None or not obj.get("pagination", {}).get("has_more"):
                            break
                        cursor = obj["pagination"].get("next_cursor")
                        if not isinstance(cursor, str) or not cursor or cursor in seen:
                            raise ValidationError("invalid historical price cursor")
                        seen.add(cursor)
            for market in km:
                settled = market.get("settlement_ts")
                historical = bool(cutoff and settled and timestamp(settled, "settled") < timestamp(cutoff["market_settled_ts"], "historical cutoff"))
                for t in observation_times(event, plan):
                    archive.get(kalshi_price_url(market["ticker"], t, plan, historical))
        report = {"schema_version": "1", "kind": "historical_market_capture", "config": plan,
                  "config_sha256": canonical_hash(plan), "requests": archive.requests,
                  "requests_sha256": canonical_hash(archive.requests), "code": code_provenance()}
        (stage / "manifest.json").write_text(json_text(report))
        stage.rename(output)
    return {"requests": len(report["requests"]), "successful_requests": sum(r["status"] == 200 for r in report["requests"]),
            "failed_requests": [{"url": r["url"], "status": r["status"]} for r in report["requests"] if r["status"] != 200]}


def poly_history_quote(pages: list[dict], t: datetime, plan: dict) -> tuple[dict | None, dict]:
    quality = {"kind": "historical_aggregated_price", "excluded_zero_width_points": 0,
               "excluded_future_or_straddling_points": 0, "candidate_points": 0, "issues": []}
    if not pages or pages[-1].get("pagination", {}).get("has_more") is not False:
        quality["issues"].append("price_history_incomplete")
        return None, quality
    candidates, seen = [], {}
    for page in pages:
        if not isinstance(page.get("data"), list):
            raise ValidationError("invalid price page")
        for item in page["data"]:
            point, width = item.get("timestamp"), item.get("resolution_seconds")
            if type(point) is not int or point <= 0 or type(width) is not int or width < 0:
                raise ValidationError("invalid historical price interval")
            p = probability(item.get("price"), "historical price")
            key = (point, width)
            if key in seen and seen[key] != p:
                raise ValidationError("conflicting historical price points")
            if key in seen:
                continue
            seen[key] = p
            # Width 0 conflates real exact ticks and synthetic settled-price
            # points. With no discriminator, fail closed on ALL zero-width rows.
            if width == 0:
                quality["excluded_zero_width_points"] += 1
                continue
            end = datetime.fromtimestamp(point + width, timezone.utc)
            if end > t:
                quality["excluded_future_or_straddling_points"] += 1
                continue
            if point < (t - timedelta(hours=plan["price_window_hours"])).timestamp():
                continue
            candidates.append((end, point, width, p))
    quality["candidate_points"] = len(candidates)
    if not candidates:
        quality["issues"].append("no_proven_pre_cutoff_price")
        return None, quality
    end, point, width, p = max(candidates)
    age = (t - end).total_seconds()
    quality.update(bucket_start=point, bucket_seconds=width, bucket_end=iso(end), age_seconds=age)
    if age > plan["max_price_age_seconds"]:
        quality["issues"].append("stale_historical_price")
    if not 0 < p < 1:
        quality["issues"].append("boundary_historical_price")
    return (None if quality["issues"] else {"probability": p, "observed_at": iso(end), "available_at": iso(end)}), quality


def kalshi_history_quote(obj: dict | None, t: datetime, plan: dict, *, historical: bool = False) -> tuple[dict | None, dict]:
    quality = {"kind": "historical_candle_bid_ask_midpoint", "issues": []}
    if historical:quality['source_schema'] = 'historical_fixed_point_close'
    if obj is None:
        quality["issues"].append("price_history_missing")
        return None, quality
    valid = []
    seen = set()
    for row in obj["candlesticks"]:
        end_ts = row["end_period_ts"]
        if type(end_ts) is not int or end_ts in seen:
            raise ValidationError("invalid/duplicate candle time")
        seen.add(end_ts)
        end = datetime.fromtimestamp(end_ts, timezone.utc)
        if end > t or end < t - timedelta(hours=plan["price_window_hours"]):
            continue
        key = 'close' if historical else 'close_dollars'
        bid, ask = row.get("yes_bid", {}).get(key), row.get("yes_ask", {}).get(key)
        if bid is None or ask is None:
            continue
        if historical and any(not isinstance(x, str) or not re.fullmatch(r'(?:0|1)\.\d{4}', x) for x in (bid, ask)):
            raise ValidationError('invalid historical fixed-point dollar quote')
        bid, ask = decimal_value(bid, "historical bid"), decimal_value(ask, "historical ask")
        if not 0 <= bid <= ask <= 1:
            raise ValidationError("invalid historical candle quotes")
        valid.append((end, bid, ask))
    if not valid:
        quality["issues"].append("missing_historical_quote")
        return None, quality
    end, bid, ask = max(valid)
    quality.update(candle_end=iso(end), bid=float(bid), ask=float(ask), spread=float(ask - bid), age_seconds=(t - end).total_seconds())
    if not 0 < bid <= ask < 1:
        quality["issues"].append("boundary_historical_price")
    if (t - end).total_seconds() > plan["max_price_age_seconds"]:
        quality["issues"].append("stale_historical_price")
    if ask - bid > decimal_value("0.10", "spread"):
        quality["issues"].append("wide_historical_spread")
    return (None if quality["issues"] else {"probability": float((bid + ask) / 2), "observed_at": iso(end), "available_at": iso(end)}), quality


def fed_upper_bound(text: str) -> Fraction:
    """Read the Committee's decision, not a dissenting member's preferred rate."""
    num = r"\d+(?:-\d+/\d+|\.\d+)?"
    change = r"\d+(?:/\d+|\.\d+)? percentage points?"
    rate_range = (r"target range for the federal funds rate (?:at|by " + change
                  + r" to) (" + num + r") to (" + num + r") percent")
    # The Fed also publishes mixed fractions with a nonbreaking hyphen.
    # Preserve archived source text; normalize only these equivalent glyphs.
    normalized = normalize(text).replace('\u2011', '-').replace('\u2010', '-')
    decisions = re.findall(r"\b[Tt]he Committee decided to (?:maintain|lower|raise) the " + rate_range, normalized)
    # Keep support for callers supplying only the range clause. Full official
    # statements may also quote a dissent: December 2024 contains two ranges.
    has_decision = re.search(r"\b[Tt]he Committee decided to\b", normalized)
    matches = decisions if has_decision else re.findall(rate_range, normalized)
    if len(matches) != 1:
        raise ValidationError("cannot unambiguously extract official target range")
    def value(s):
        if "-" in s:
            whole, frac = s.split("-", 1)
            return Fraction(whole) + Fraction(frac)
        return Fraction(s)
    low, high = map(value, matches[0])
    if not 0 <= low < high <= 20:
        raise ValidationError("invalid official target range")
    return high


def expected_poly_outcome(question: str, change_bps: Fraction) -> int:
    if question.startswith("Will there be no change in Fed interest rates"):
        return int(change_bps == 0)
    match = re.match(r"Will the Fed (decrease|increase) interest rates by (25|50\+) bps", question)
    if not match:
        raise ValidationError("unsupported FOMC rate-change contract")
    signed = change_bps if match[1] == "increase" else -change_bps
    return int(signed == 25 if match[2] == "25" else signed >= 50)


def historical_poly_label(market: dict, state: dict | None, available: str, t: datetime,
                          expected: int) -> tuple[dict | None, dict]:
    ledger = {"outcome": None, "resolution_time": None, "available_at": available, "status": "resolution_unverified"}
    if not state or state.get("condition_id") != market["conditionId"]:
        return None, ledger
    if market.get("closed") is not True or state.get("status") != "resolved":
        ledger["status"] = "not_finally_resolved"
        return None, ledger
    if state.get("extended_review") is not False or state.get("was_disputed") is not False:
        ledger["status"] = "disputed_or_unverified_resolution"
        return None, ledger
    _, yes_index = mapping(market)
    payout = state.get("payouts")
    if payout is not None:
        if not (isinstance(payout, list) and len(payout) == 2
                and all(type(x) is int for x in payout) and sorted(payout) == [0, 1_000_000]):
            ledger["status"] = "non_binary_or_missing_payout"
            return None, ledger
        outcome = int(payout[yes_index] == 1_000_000)
        if state.get("price") is not None and state["price"] != str(outcome * 10**18):
            ledger["status"] = "conflicting_platform_resolution_fields"
            return None, ledger
    elif state.get("price") in ("0", "1000000000000000000"):
        # UMA YES_OR_NO final oracle price is useful outcome evidence, but its
        # last_update_timestamp and transaction_hash are NOT a settlement proof.
        outcome = int(state["price"] != "0")
    else:
        ledger["status"] = "non_binary_or_missing_payout"
        return None, ledger
    ledger.update(outcome=outcome, status="outcome_cross_checked_time_unverified", official_outcome=expected)
    if outcome != expected:
        ledger["status"] = "official_result_disagrees_with_platform"
        return None, ledger
    if (not isinstance(payout, list) or not state.get("resolved_at") or type(state.get("resolved_block")) is not int
            or state["resolved_block"] <= 0 or state.get("resolution_source") not in ("reported", "derived")):
        return None, ledger
    resolved = timestamp(state["resolved_at"], "resolution")
    if not t < resolved <= timestamp(available, "label availability"):
        ledger["status"] = "invalid_settlement_timeline"
        return None, ledger
    label = {"outcome": outcome, "resolution_time": iso(resolved), "available_at": available}
    ledger.update(**label, status="verified_settlement")
    return label, ledger


def historical_kalshi_label(market: dict, available: str, t: datetime, expected: int) -> tuple[dict | None, dict]:
    ledger = {"outcome": None, "resolution_time": None, "available_at": available, "status": "resolution_unverified"}
    if (market.get("status") not in ("settled", "finalized") or market.get("result") not in ("yes", "no")
            or market.get("is_provisional") is True or not market.get("settlement_ts")):
        return None, ledger
    outcome = int(market["result"] == "yes")
    ledger.update(outcome=outcome, official_outcome=expected)
    if outcome != expected or (market.get("settlement_value_dollars") is not None and decimal_value(market["settlement_value_dollars"], "payout") != outcome):
        ledger["status"] = "official_result_disagrees_with_platform"
        return None, ledger
    resolved = timestamp(market["settlement_ts"], "settlement")
    if not t < resolved <= timestamp(available, "availability"):
        ledger["status"] = "invalid_settlement_timeline"
        return None, ledger
    label = {"outcome": outcome, "resolution_time": iso(resolved), "available_at": available}
    ledger.update(**label, status="verified_settlement")
    return label, ledger


def build_historical_markets(capture: Path, heldout_index: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError("historical replay output already exists")
    manifest, responses = read_archive(capture)
    plan = load_plan(manifest["config"])
    index_bytes = heldout_index.read_bytes()
    index = strict_json(index_bytes.decode())
    if (index.get("source") != "forecastbench" or not index.get("entries")
            or canonical_hash(index["entries"]) != index.get("entries_sha256")):
        raise ValidationError("invalid held-out index")
    heldout_questions = {normalize(e["question"]).casefold() for e in index["entries"]}
    heldout_ids = {e["event_id"] for e in index["entries"]}
    def get(url, *, binary=False):
        ref, raw = responses[url]
        if ref["status"] != 200:
            raise ValidationError(f"source fetch failed ({ref['status']}): {url}")
        return ref, raw if binary else strict_json(raw.decode())
    def safe_get(url):
        if url not in responses or responses[url][0]["status"] != 200:
            return None
        return get(url)[1]
    cutoff = safe_get(KALSHI + "/historical/cutoff")
    candidates, ledgers, evidence_archive, exclusions, model_inputs = [], [], {}, [], []
    version = "sha256:" + canonical_hash({"requests": manifest["requests_sha256"], "config": manifest["config_sha256"]})
    for event in plan["events"]:
        rref, raw_result = get(event["result_url"], binary=True)
        official_result = fed_statement(raw_result, event["result_url"], event["release_time"])
        final_rate = fed_upper_bound(official_result["text"])
        releases = []
        for item in event["evidence"]:
            eref, raw_evidence = get(item["url"], binary=True)
            release = fed_statement(raw_evidence, item["url"], item["published_at"])
            release["retrieval_ref"] = eref
            releases.append(release)
            evidence_archive[release["sha256"]] = release
        releases.sort(key=lambda r: timestamp(r["published_at"], "publication"))
        rate_before = fed_upper_bound(releases[-1]["text"])
        change_bps = (final_rate - rate_before) * 100
        # The pilot does not support nonstandard 12.5bp rounding or emergency
        # meetings. Reject rather than silently impose a different contract.
        if change_bps.denominator != 1 or change_bps % 25:
            raise ValidationError("unsupported FOMC change increment")
        for platform in ("polymarket", "kalshi"):
            eurl = GAMMA + "/events/" + event["polymarket_event_id"] if platform == "polymarket" else KALSHI + "/events/" + event["kalshi_event_ticker"]
            try:
                mref, obj = get(eurl)
                if platform == "polymarket" and str(obj.get("id")) != event["polymarket_event_id"]:
                    raise ValidationError("Polymarket event identity mismatch")
                if platform == "kalshi" and obj.get("event", {}).get("event_ticker") != event["kalshi_event_ticker"]:
                    raise ValidationError("Kalshi event identity mismatch")
                markets = obj["markets"]
                if not markets:
                    raise ValidationError("empty native event; older markets need historical discovery")
                if len(markets) > plan["max_markets_per_platform_event"]:
                    raise ValidationError("event exceeds configured market budget")
            except (ValidationError, KeyError) as exc:
                exclusions.append({"event_group_id": event["event_group_id"], "platform": platform, "reason": str(exc)})
                continue
            states, state_ref = {}, None
            if platform == "polymarket":
                state_ref, state_obj = get(DATA + "/v2/resolutions?event_id=" + event["polymarket_event_id"])
                for state in state_obj["data"]:
                    if state["condition_id"] in states:
                        raise ValidationError("duplicate resolution condition")
                    states[state["condition_id"]] = state
            seen = set()
            for market in markets:
                mid = str(market["id"]) if platform == "polymarket" else market["ticker"]
                if mid in seen:
                    raise ValidationError("duplicate native market")
                seen.add(mid)
                for t in observation_times(event, plan):
                    reasons, proof, state = [], None, None
                    sid = identifier(SOURCE_PREFIX + platform, mid, iso(t))
                    try:
                        if not t < timestamp(official_result["published_at"], "first public outcome"):
                            raise ValidationError("observation after public outcome")
                        evidence = [{"evidence_id": "fed:" + r["sha256"][:16], "text": r["text"], "source": r["source"],
                                     "published_at": r["published_at"], "available_at": r["published_at"]}
                                    for r in releases if timestamp(r["published_at"], "evidence time") <= t]
                        if not evidence:
                            reasons.append("pre_cutoff_evidence_missing")
                        if platform == "polymarket":
                            tokens, _ = mapping(market)
                            state = states.get(market["conditionId"])
                            expected = expected_poly_outcome(market["question"], change_bps)
                            clob = safe_get(CLOB + "/clob-markets/" + market["conditionId"])
                            if (not clob or clob.get("c") != market["conditionId"]
                                    or [(x.get("o"), x.get("t")) for x in clob.get("t", [])] != list(tokens.items())):
                                reasons.append("outcome_token_mapping_unverified")
                            if timestamp(market["startDate"], "market open") > t:
                                raise ValidationError("market not yet open at cutoff")
                            original_question = market["question"] + "\nResolution rules: " + market["description"]
                            question = original_question
                            if state and re.fullmatch(r"0x[0-9a-fA-F]{64}", state.get("transaction_hash", "")):
                                try:
                                    pref, praw = get("https://polygonscan.com/tx/" + state["transaction_hash"], binary=True)
                                    proof = initialized_question(praw, tx_hash=state["transaction_hash"],
                                        request_id=market["negRiskRequestID"], adapter=market["resolvedBy"])
                                    if state.get("question_id") != proof["question_id"]:
                                        raise ValidationError("resolution and initialization request disagree")
                                    if timestamp(proof["published_at"], "question publication") > t:
                                        raise ValidationError("question initialized after cutoff")
                                    prefix = "q: title: " + market["question"] + ", description: " + market["description"]
                                    if not normalize(proof["ancillary_data"]).startswith(normalize(prefix)):
                                        raise ValidationError("current rules differ from archived initialization")
                                    proof["ref"] = pref
                                    # Only text actually present in the historical log is eligible.
                                    question = proof["ancillary_data"].split(", creator: ", 1)[0]
                                except (ValidationError, KeyError) as exc:
                                    proof = None
                                    reasons.append("question_archive_unverified: " + str(exc))
                            else:
                                reasons.append("question_archive_missing")
                            if state and state.get("new_version_q") is not False:
                                reasons.append("rule_update_history_review_required")
                            pages, cursor, cursors = [], None, set()
                            for _ in range(plan["max_price_pages"]):
                                page = safe_get(poly_price_url(tokens["Yes"], t, plan, cursor))
                                if page is None:
                                    break
                                pages.append(page)
                                if page.get("pagination", {}).get("has_more") is False:
                                    break
                                cursor = page.get("pagination", {}).get("next_cursor")
                                if not isinstance(cursor, str) or not cursor or cursor in cursors:
                                    raise ValidationError("invalid replay price cursor")
                                cursors.add(cursor)
                            quote, quality = poly_history_quote(pages, t, plan)
                            available = max([mref["completed_at"], state_ref["completed_at"], rref["completed_at"]], key=lambda x: timestamp(x, "received"))
                            label, ledger = historical_poly_label(market, state, available, t, expected)
                        else:
                            if (market.get("event_ticker") != event["kalshi_event_ticker"] or market.get("market_type") != "binary"
                                    or decimal_value(market.get("notional_value_dollars"), "notional") != 1):
                                raise ValidationError("unsupported Kalshi contract identity")
                            if not timestamp(market["open_time"], "open") <= t < timestamp(market["close_time"], "close"):
                                raise ValidationError("market not open at cutoff")
                            match = re.fullmatch(re.escape(event["kalshi_event_ticker"]) + r"-T(\d+(?:\.\d+)?)", mid)
                            if not match or market.get("strike_type") != "greater" or Fraction(str(market["floor_strike"])) != Fraction(match[1]):
                                raise ValidationError("unsupported Kalshi threshold contract")
                            expected = int(final_rate > Fraction(match[1]))
                            original_question = market["title"] + "\nYes outcome: " + market["yes_sub_title"] + "\nResolution rules: " + market["rules_primary"] + "\n" + (market.get("rules_secondary") or "")
                            question = original_question
                            # The 2021 certification and currently served rule PDF
                            # differ in expiry/trading provisions. Neither proves
                            # the full 2026 wording was available at historical T.
                            reasons.append("historical_market_rules_version_missing")
                            historical = bool(cutoff and market.get("settlement_ts") and timestamp(market["settlement_ts"], "settled") < timestamp(cutoff["market_settled_ts"], "cutoff"))
                            quote, quality = kalshi_history_quote(safe_get(kalshi_price_url(mid, t, plan, historical)), t, plan, historical=historical)
                            available = max([mref["completed_at"], rref["completed_at"]], key=lambda x: timestamp(x, "received"))
                            label, ledger = historical_kalshi_label(market, available, t, expected)
                        record = {"sample_id": sid, "dataset_source": SOURCE_PREFIX + platform, "dataset_version": version,
                                  "event_id": identifier(platform, mid), "event_group_id": event["event_group_id"],
                                  "question": question, "observation_time": iso(t), "evidence": evidence, "market": quote, "label": label}
                        parse_record(record)
                        if quote is None:
                            reasons.append("historical_price_unusable")
                        if label is None:
                            reasons.append("exact_settlement_proof_missing")
                        if ledger["status"] == "official_result_disagrees_with_platform":
                            reasons.append(ledger["status"])
                        native_title = market["question"] if platform == "polymarket" else market["title"]
                        if (record["event_id"] in heldout_ids
                                or normalize(original_question).casefold() in heldout_questions
                                or normalize(native_title).casefold() in heldout_questions):
                            reasons.append("exact_heldout_overlap")
                        reasons.extend(["semantic_benchmark_review_required", "historical_teacher_target_missing"])
                        candidates.append({"record": record, "original_question": original_question, "blockers": sorted(set(reasons)),
                                           "question_proof": proof, "quote_quality": quality, "market_ref": mref,
                                           "result_ref": rref, "first_public_result_time": official_result["published_at"],
                                           "resolution_ref": state_ref, "ready_for_sft": False, "ready_for_benchmark": False})
                        ledgers.append({"sample_id": sid, "event_group_id": event["event_group_id"], "platform": platform, **ledger})
                        # Archive-proven original questions can be inspected with
                        # pre-cutoff context. Current-rule candidates stay outside
                        # this view. This export is NOT a training release.
                        if proof is not None and evidence and quote is not None:
                            model_inputs.append({"sample_id": sid, "input": parse_record(record).forecast_input.to_payload(),
                                                 "status": "review_only_initial_rules; changes_and_supervision_pending"})
                    except (ValidationError, KeyError) as exc:
                        exclusions.append({"sample_id": sid, "event_group_id": event["event_group_id"], "platform": platform, "reason": str(exc)})
    validate_identities([parse_record(x["record"]) for x in candidates])
    candidates.sort(key=lambda x: x["record"]["sample_id"])
    ledgers.sort(key=lambda x: x["sample_id"])
    model_inputs.sort(key=lambda x: x["sample_id"])
    artifacts = {"candidates.jsonl": jsonl(candidates), "outcomes.jsonl": jsonl(ledgers),
                 "review_inputs.jsonl": jsonl(model_inputs), "evidence.jsonl": jsonl(list(evidence_archive.values())),
                 "exclusions.json": json_text(exclusions), "plan.json": json_text(plan)}
    report = {"schema_version": "1", "kind": "historical_prediction_market_staging", "dataset_version": version,
              "capture_requests_sha256": manifest["requests_sha256"], "config_sha256": manifest["config_sha256"],
              "heldout_index_sha256": sha256_bytes(index_bytes), "code": code_provenance(),
              "counts": {"candidates": len(candidates), "market_contracts": len({x["record"]["event_id"] for x in candidates}),
                         "event_groups": len({x["record"]["event_group_id"] for x in candidates}),
                         "platforms": dict(Counter(x["record"]["dataset_source"] for x in candidates)),
                         "historical_quotes": sum(x["record"]["market"] is not None for x in candidates),
                         "archive_proven_initial_questions": sum(x["question_proof"] is not None for x in candidates),
                         "exact_settlement_labels": sum(x["record"]["label"] is not None for x in candidates),
                         "outcomes_cross_checked": sum(x["status"] in ("verified_settlement", "outcome_cross_checked_time_unverified") for x in ledgers),
                         "excluded_zero_width_price_points": sum(x["quote_quality"].get("excluded_zero_width_points", 0) for x in candidates),
                         "evidence_documents": len(evidence_archive), "review_only_inputs": len(model_inputs), "exclusions": len(exclusions),
                         "blockers": dict(Counter(r for x in candidates for r in x["blockers"])), "ready_for_sft": 0, "ready_for_benchmark": 0},
              "limitations": ["Development cohort, not a random or representative sample.",
                              "Market discovery currently requires a populated native event response; older archived-only events are excluded explicitly.",
                              "Historical publication assertions rely on official dated releases and an explorer's receipt archive.",
                              "Current Kalshi rules are not certified as the historical version.",
                              "Polymarket initial text does not prove completeness of later rule clarifications.",
                              "Zero-width price points include legitimate ticks and synthetic settlement points; all excluded conservatively.",
                              "Final outcome is an evaluator label, never an SFT probability target. No teacher answers generated.",
                              "ForecastBench semantic overlap, historical knowledge contamination and event/time splits remain to review."],
              "artifact_hashes": {k: sha256_bytes(v.encode()) for k, v in artifacts.items()}}
    artifacts["report.json"] = json_text(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".historical-build-", dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        for name, content in artifacts.items():
            (stage / name).write_text(content)
        stage.rename(output)
    return report
