"""Native-source parsing into audit candidates, never directly into training data."""

import ast
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import math
from pathlib import Path
from urllib.parse import quote

from .data import sha256_file, strict_json
from .schema import ValidationError, iso, nonempty, probability, timestamp


@dataclass
class Candidate:
    sample_id: str
    event_id: str
    suggested_event_group_id: str
    question: str
    observation_time: str | None
    outcome: int | None
    market: dict | None
    locator: str
    context: dict = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        return asdict(self)


def identifier(*parts: str) -> str:
    return ":".join(quote(nonempty(part, "source identifier"), safe="") for part in parts)


def native_binary(value) -> int | None:
    if type(value) in (int, float) and value in (0, 1):
        return int(value)
    return None


def numeric(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValidationError(f"{context}: expected a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"{context}: invalid number") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{context}: nonfinite number")
    return result


def midpoint_cents(data: dict, observed_at: str) -> dict | None:
    if data.get("yes_bid") is None or data.get("yes_ask") is None:
        return None
    bid, ask = numeric(data["yes_bid"], "yes_bid"), numeric(data["yes_ask"], "yes_ask")
    if not 0 <= bid <= ask <= 100:
        raise ValidationError("invalid or crossed bid/ask in cents")
    return {"probability": (bid + ask) / 200, "observed_at": observed_at,
            "available_at": observed_at}


def prophet_candidates(path: Path) -> tuple[list[Candidate], dict]:
    candidates, categories, native_events, submissions = [], Counter(), set(), set()
    omitted_evidence = 0
    required = {"submission_id", "event_ticker", "title", "snapshot_time", "close_time",
                "market_data", "market_outcome", "category", "markets", "sources"}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValidationError("invalid/duplicate CSV header")
        if not required <= set(reader.fieldnames):
            raise ValidationError(f"Prophet CSV missing columns: {sorted(required - set(reader.fieldnames))}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValidationError(f"Prophet CSV malformed row: {line}")
            submission = nonempty(row["submission_id"], "submission_id")
            if submission in submissions:
                raise ValidationError(f"duplicate Prophet submission_id: {submission}")
            submissions.add(submission)
            native_event = nonempty(row["event_ticker"], "event_ticker")
            native_events.add(native_event)
            categories[row["category"]] += 1
            observed = iso(timestamp(row["snapshot_time"], "snapshot_time"))
            # The actual CSV stores this one column as a Python literal, not JSON.
            try:
                markets = ast.literal_eval(row["markets"])
            except (ValueError, SyntaxError, RecursionError) as exc:
                raise ValidationError(f"invalid market list at row {line}") from exc
            if not isinstance(markets, list) or not markets or any(not isinstance(x, str) or not x for x in markets):
                raise ValidationError(f"invalid market list at row {line}")
            if len(markets) != len(set(markets)):
                raise ValidationError(f"duplicate market name at row {line}")
            outcomes, quotes, sources = (strict_json(row[key]) for key in ("market_outcome", "market_data", "sources"))
            if not isinstance(outcomes, dict) or set(outcomes) != set(markets):
                raise ValidationError(f"market/outcome keys disagree at row {line}")
            if not isinstance(quotes, dict) or not isinstance(sources, list):
                raise ValidationError(f"invalid market_data/sources at row {line}")
            omitted_evidence += len(sources)
            for market in markets:
                blockers, notes = [], ["source_summaries_omitted_without_timestamps",
                                       "augmented_title_and_rules_omitted_without_provenance"]
                outcome = native_binary(outcomes[market])
                if outcome is None:
                    blockers.append("non_binary_outcome")
                quote_data = None
                try:
                    if market in quotes:
                        if not isinstance(quotes[market], dict):
                            raise ValidationError("invalid quote object")
                        quote_data = midpoint_cents(quotes[market], observed)
                except ValidationError:
                    notes.append("invalid_market_quote_omitted")
                if quote_data is None:
                    notes.append("market_snapshot_missing")
                candidates.append(Candidate(
                    identifier("prophet_arena", submission, market),
                    identifier("kalshi", native_event, market), identifier("kalshi", native_event),
                    nonempty(row["title"], "title") + f"\nContract: {market}",
                    observed, outcome, quote_data, f"csv_row:{line}",
                    {"submission_id": submission, "event_ticker": native_event, "market_name": market,
                     "category": row["category"], "close_time_untrusted_for_resolution": row["close_time"],
                     "omitted_source_count": len(sources)}, blockers, notes,
                ))
    return candidates, {"input_rows": len(submissions), "native_event_count": len(native_events),
                        "categories": dict(sorted(categories.items())),
                        "omitted_source_entries": omitted_evidence}


def forecastbench_candidates(questions_path: Path, resolutions_path: Path) -> tuple[list[Candidate], dict]:
    questions = strict_json(questions_path.read_text(encoding="utf-8"))
    resolutions = strict_json(resolutions_path.read_text(encoding="utf-8"))
    for item, key in ((questions, "questions"), (resolutions, "resolutions")):
        if not isinstance(item, dict) or not isinstance(item.get(key), list):
            raise ValidationError(f"ForecastBench requires a {key} array")
    if any(questions.get(key) != resolutions.get(key) for key in ("forecast_due_date", "question_set")):
        raise ValidationError("ForecastBench question/resolution sets do not match")
    due = nonempty(questions.get("forecast_due_date"), "forecast_due_date")
    date.fromisoformat(due)
    grouped = defaultdict(list)
    for item in resolutions["resolutions"]:
        if not isinstance(item, dict):
            raise ValidationError("invalid resolution object")
        grouped[(nonempty(item.get("source"), "source"), nonempty(item.get("id"), "id"))].append(item)
    candidates, seen, source_counts = [], set(), Counter()
    for index, question in enumerate(questions["questions"]):
        source = nonempty(question.get("source"), "source")
        native_id = nonempty(question.get("id"), "id")
        key = source, native_id
        if key in seen:
            raise ValidationError(f"duplicate ForecastBench question: {key}")
        seen.add(key)
        source_counts[source] += 1
        frozen = iso(timestamp(question.get("freeze_datetime"), "freeze_datetime"))
        horizons = question.get("resolution_dates")
        if horizons == "N/A":
            horizons = [None]
        elif not isinstance(horizons, list) or not horizons:
            raise ValidationError("invalid ForecastBench resolution_dates")
        if len(set(horizons)) != len(horizons):
            raise ValidationError("duplicate ForecastBench resolution horizon")
        for horizon in horizons:
            if horizon is not None:
                date.fromisoformat(horizon)
            matches = [row for row in grouped.get(key, []) if horizon is None or row.get("resolution_date") == horizon]
            if len(matches) > 1:
                raise ValidationError(f"ambiguous ForecastBench resolution join: {key}, {horizon}")
            resolution = matches[0] if matches else None
            outcome, blockers, notes = None, [], ["background_omitted_without_version_timestamps"]
            if resolution is None or resolution.get("resolved") is not True:
                blockers.append("unresolved_or_missing_resolution")
            elif resolution.get("direction") is not None:
                blockers.append("unsupported_resolution_direction")
            else:
                outcome = native_binary(resolution.get("resolved_to"))
                if outcome is None:
                    blockers.append("non_binary_resolution")
            market = None
            # Crowd forecasts and time-series levels are NOT traded market prices.
            if source in ("polymarket", "manifold"):
                try:
                    p = probability(numeric(question.get("freeze_datetime_value"), "freeze value"))
                    market = {"probability": p, "observed_at": frozen, "available_at": frozen}
                except ValidationError:
                    notes.append("invalid_market_probability_omitted")
            else:
                notes.append("non_market_reference_value_not_used_as_probability")
            text = nonempty(question.get("question"), "question")
            text = text.replace("{forecast_due_date}", due)
            if horizon is not None:
                text = text.replace("{resolution_date}", horizon)
            if "{resolution_date}" in text:
                blockers.append("unexpanded_question_template")
            event_id = identifier(source, native_id) if horizon is None else identifier(source, native_id, due, horizon)
            criteria = question.get("resolution_criteria")
            if isinstance(criteria, str) and criteria and criteria != "N/A":
                text += "\nResolution criteria: " + criteria
            candidates.append(Candidate(
                identifier("forecastbench", due, event_id), event_id, identifier(source, native_id),
                text, None, outcome, market, f"questions[{index}];horizon={horizon}",
                {"source": source, "native_id": native_id, "forecast_due_date": due,
                 "freeze_datetime": frozen, "resolution_horizon": horizon,
                 "resolution_date_hint": None if resolution is None else resolution.get("resolution_date"),
                 "raw_resolved_to": None if resolution is None else resolution.get("resolved_to"),
                 "raw_resolved": None if resolution is None else resolution.get("resolved")},
                blockers, notes,
            ))
    return candidates, {"input_rows": len(questions["questions"]),
                        "resolution_rows": len(resolutions["resolutions"]),
                        "question_set": questions.get("question_set"), "sources": dict(sorted(source_counts.items()))}


def kalshi_snapshot_candidates(path: Path) -> tuple[list[Candidate], dict]:
    """PMA Kalshi metadata shards only. No retrospective trade-price reconstruction."""
    try:
        import pyarrow
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ValidationError("Parquet import requires the optional 'parquet' extra") from exc
    grouped = defaultdict(list)
    parquet_file = parquet.ParquetFile(path)
    required = {"ticker", "event_ticker", "title", "market_type", "status", "result", "_fetched_at"}
    if not required <= set(parquet_file.schema_arrow.names):
        raise ValidationError("expected PMA Kalshi market metadata schema")
    rows_count = 0
    for batch in parquet_file.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            rows_count += 1
            fetched = row["_fetched_at"]
            if isinstance(fetched, datetime):
                fetched = fetched.isoformat()
            fetched = iso(timestamp(fetched, "_fetched_at"))
            grouped[nonempty(row["ticker"], "ticker")].append((fetched, rows_count, row))
    candidates = []
    for ticker, snapshots in sorted(grouped.items()):
        snapshots.sort(key=lambda item: timestamp(item[0], "snapshot time"))
        outcomes = {row["result"] for _, _, row in snapshots if row["status"] == "finalized" and row["result"] in ("yes", "no")}
        for fetched, row_number, row in snapshots:
            blockers = []
            if row["market_type"] != "binary":
                blockers.append("non_binary_market")
            if row["status"] != "open" or row["result"] not in (None, ""):
                blockers.append("not_a_pre_resolution_open_snapshot")
            later = [(when, data) for when, _, data in snapshots
                     if timestamp(when, "label snapshot") > timestamp(fetched, "observation snapshot")
                     and data["status"] == "finalized" and data["result"] in ("yes", "no")]
            outcome = None
            label_available = None
            if len(outcomes) > 1:
                blockers.append("conflicting_finalized_outcomes")
            elif not later:
                blockers.append("no_later_finalized_label_snapshot")
            else:
                label_available = later[0][0]
                outcome = int(later[0][1]["result"] == "yes")
            market = midpoint_cents(row, fetched)
            parent = nonempty(row["event_ticker"], "event_ticker")
            title = nonempty(row["title"], "title")
            if row.get("yes_sub_title"):
                title += "\nYes outcome: " + row["yes_sub_title"]
            candidates.append(Candidate(
                identifier("prediction_market_analysis", ticker, fetched), identifier("kalshi", ticker),
                identifier("kalshi", parent), title, fetched, outcome, market, f"parquet_row:{row_number}",
                {"ticker": ticker, "event_ticker": parent, "label_available_at_floor": label_available},
                blockers, ["market_probability_is_quote_midpoint"],
            ))
    return candidates, {"input_rows": rows_count, "native_market_count": len(grouped),
                        "input_sha256": sha256_file(path), "supported_subset": "kalshi_market_snapshots",
                        "pyarrow_version": pyarrow.__version__}
