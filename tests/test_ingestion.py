"""Native-format fixtures are synthetic; no benchmark text is redistributed."""

import copy
import csv
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.adapters import forecastbench_candidates, kalshi_snapshot_candidates, prophet_candidates
from foretellmesh.config import load_config
from foretellmesh.data import load_records, sha256_bytes
from foretellmesh.ingestion import annotation_template, audit_candidates, ingest_source, promote_candidates
from foretellmesh.schema import ValidationError
from foretellmesh.sources import download_source, load_source
from tests.test_pipeline import CONFIG, run_cli


REVISION = "a" * 40


def write_prophet(path: Path, rows=None):
    default = {
        "submission_id": "s1", "event_ticker": "EVENT-1", "title": "Synthetic threshold event?",
        "snapshot_time": "2024-01-10T00:00:00Z", "close_time": "2024-01-11T00:00:00Z",
        "market_data": json.dumps({"Above 10": {"yes_bid": 60, "yes_ask": 80}}),
        "market_outcome": json.dumps({"Above 10": 1, "Above 20": 0}),
        "category": "Economics", "markets": "['Above 10', 'Above 20']",
        "augmented_title": "UNTRUSTED post-event augmented text", "rules": "UNTRUSTED generated rules",
        "sources": json.dumps([{"summary": "UNTRUSTED untimestamped summary", "url": "synthetic://source"}]),
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(default))
        writer.writeheader()
        writer.writerows([default] if rows is None else rows)
    return default


def write_forecast(path: Path, resolution_path: Path, resolved=True, value=1.0):
    questions = {
        "forecast_due_date": "2024-01-10", "question_set": "synthetic-round.json",
        "questions": [
            {"id": "shared-id", "source": "polymarket", "question": "Will synthetic event A occur?",
             "freeze_datetime": "2024-01-01T00:00:00Z", "freeze_datetime_value": "0.7",
             "resolution_dates": "N/A", "resolution_criteria": "Synthetic criterion",
             "background": "UNTRUSTED later background"},
            {"id": "shared-id", "source": "fred", "question": "Will the synthetic series rise from {forecast_due_date} to {resolution_date}?",
             "freeze_datetime": "2024-01-01T00:00:00Z", "freeze_datetime_value": "123.45",
             "resolution_dates": ["2024-01-20", "2024-02-20"], "resolution_criteria": "Synthetic series rule"},
        ],
    }
    resolutions = {
        "forecast_due_date": "2024-01-10", "question_set": "synthetic-round.json",
        "resolutions": [
            {"source": "polymarket", "id": "shared-id", "direction": None,
             "resolution_date": "2024-01-20", "resolved": resolved, "resolved_to": value},
            {"source": "fred", "id": "shared-id", "direction": None,
             "resolution_date": "2024-01-20", "resolved": True, "resolved_to": 0.0},
            {"source": "fred", "id": "shared-id", "direction": None,
             "resolution_date": "2024-02-20", "resolved": True, "resolved_to": 1.0},
        ],
    }
    path.write_text(json.dumps(questions), encoding="utf-8")
    resolution_path.write_text(json.dumps(resolutions), encoding="utf-8")
    return questions, resolutions


def write_registry(path: Path, source: str, data: Path, resolutions: Path | None = None):
    files = {data.name: {"url": f"https://huggingface.co/datasets/synthetic/test/resolve/{REVISION}/{data.name}",
                        "sha256": sha256_bytes(data.read_bytes()), "max_bytes": 1000000,
                        "purpose": "questions" if source == "forecastbench" else "data"}}
    if resolutions is not None:
        files[resolutions.name] = {"url": f"https://huggingface.co/datasets/synthetic/test/resolve/{REVISION}/{resolutions.name}",
                                  "sha256": sha256_bytes(resolutions.read_bytes()), "max_bytes": 1000000,
                                  "purpose": "resolutions"}
    registry = {"schema_version": "1", "sources": {source: {
        "revision": REVISION, "role": "eval_only" if source == "forecastbench" else "train_eval",
        "homepage": "https://example.org/synthetic", "license": "synthetic fixture",
        "artifacts": files,
    }}}
    path.write_text(json.dumps(registry), encoding="utf-8")
    return registry


def included_annotation(candidate):
    return {
        "decision": "include", "event_group_id": candidate.suggested_event_group_id,
        "observation_time": candidate.observation_time or "2024-01-10T12:00:00Z",
        "question_available_at": "2024-01-01T00:00:00Z",
        "resolution_time": "2024-01-20T12:00:00Z", "label_available_at": "2024-01-21T00:00:00Z",
        "provenance": "Synthetic test assertion, not real provenance",
    }


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_prophet_expands_contracts_and_preserves_parent_group(self):
        path = self.root / "native.csv"
        write_prophet(path)
        candidates, stats = prophet_candidates(path)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len({c.event_id for c in candidates}), 2)
        self.assertEqual(len({c.suggested_event_group_id for c in candidates}), 1)
        self.assertEqual(candidates[0].market["probability"], 0.7)
        self.assertIsNone(candidates[1].market)
        self.assertEqual(stats["omitted_source_entries"], 1)
        self.assertNotIn("UNTRUSTED", json.dumps([c.payload() for c in candidates]))
        self.assertNotIn("resolution_time", candidates[0].payload())

    def test_prophet_invalid_quote_is_omitted_with_reason(self):
        path = self.root / "native.csv"
        row = write_prophet(path)
        row["market_data"] = json.dumps({"Above 10": {"yes_bid": 90, "yes_ask": 10}})
        write_prophet(path, [row])
        candidates, _ = prophet_candidates(path)
        self.assertIsNone(candidates[0].market)
        self.assertIn("invalid_market_quote_omitted", candidates[0].notes)

    def test_prophet_rejects_bad_keys_duplicate_submissions_and_executable_literals(self):
        path = self.root / "native.csv"
        original = write_prophet(path)
        cases = []
        bad = copy.deepcopy(original)
        bad["market_outcome"] = '{"wrong": 1}'
        cases.append([bad])
        cases.append([original, original])
        bad = copy.deepcopy(original)
        bad["markets"] = "__import__('os').system('false')"
        cases.append([bad])
        for rows in cases:
            write_prophet(path, rows)
            with self.subTest(rows=rows), self.assertRaises(ValidationError):
                prophet_candidates(path)

    def test_forecastbench_join_uses_source_and_horizon_and_does_not_leak_background(self):
        q, r = self.root / "q.json", self.root / "r.json"
        write_forecast(q, r)
        candidates, _ = forecastbench_candidates(q, r)
        self.assertEqual([item.outcome for item in candidates], [1, 0, 1])
        self.assertEqual(len({item.event_id for item in candidates}), 3)
        self.assertEqual(candidates[1].suggested_event_group_id, candidates[2].suggested_event_group_id)
        self.assertIsNone(candidates[1].market)
        self.assertEqual(candidates[0].market["probability"], 0.7)
        self.assertNotIn("{resolution_date}", candidates[1].question)
        self.assertIn("2024-02-20", candidates[2].question)
        self.assertNotIn("UNTRUSTED", json.dumps([item.payload() for item in candidates]))

    def test_forecastbench_unresolved_probability_is_never_an_outcome(self):
        q, r = self.root / "q.json", self.root / "r.json"
        for value in (0.77, 0, 1):
            write_forecast(q, r, resolved=False, value=value)
            candidates, _ = forecastbench_candidates(q, r)
            self.assertIsNone(candidates[0].outcome)
            self.assertIn("unresolved_or_missing_resolution", candidates[0].blockers)

    def test_forecastbench_fractional_resolution_is_blocked(self):
        q, r = self.root / "q.json", self.root / "r.json"
        write_forecast(q, r, value=0.5)
        candidates, _ = forecastbench_candidates(q, r)
        self.assertIsNone(candidates[0].outcome)
        self.assertIn("non_binary_resolution", candidates[0].blockers)

    def test_crowd_forecast_is_not_a_market_baseline(self):
        q, r = self.root / "q.json", self.root / "r.json"
        questions, resolutions = write_forecast(q, r)
        questions["questions"][0]["source"] = "metaculus"
        resolutions["resolutions"][0]["source"] = "metaculus"
        q.write_text(json.dumps(questions))
        r.write_text(json.dumps(resolutions))
        candidates, _ = forecastbench_candidates(q, r)
        self.assertIsNone(candidates[0].market)

    def test_forecastbench_mismatched_sets_and_ambiguous_joins_rejected(self):
        q, r = self.root / "q.json", self.root / "r.json"
        _, resolutions = write_forecast(q, r)
        resolutions["forecast_due_date"] = "2024-01-11"
        r.write_text(json.dumps(resolutions))
        with self.assertRaisesRegex(ValidationError, "do not match"):
            forecastbench_candidates(q, r)
        _, resolutions = write_forecast(q, r)
        resolutions["resolutions"].append(resolutions["resolutions"][0])
        r.write_text(json.dumps(resolutions))
        with self.assertRaisesRegex(ValidationError, "ambiguous"):
            forecastbench_candidates(q, r)


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "native.csv"
        write_prophet(self.path)
        self.candidates, _ = prophet_candidates(self.path)
        self.inputs = {"data": sha256_bytes(self.path.read_bytes())}
        self.annotations = annotation_template("prophet_arena", self.inputs, self.candidates)

    def promote(self, annotations):
        return promote_candidates(self.candidates, "prophet_arena", REVISION, self.inputs, annotations)

    def test_unannotated_or_pending_candidates_cannot_reach_evaluation(self):
        for annotations in (None, self.annotations):
            accepted, disposition = self.promote(annotations)
            self.assertEqual(accepted, [])
            self.assertTrue(all(item["status"] == "quarantined" for item in disposition))

    def test_reviewed_record_preserves_label_isolation_and_does_not_invent_evidence(self):
        candidate = self.candidates[0]
        self.annotations["entries"][candidate.sample_id] = included_annotation(candidate)
        accepted, _ = self.promote(self.annotations)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["evidence"], [])
        self.assertEqual(accepted[0]["label"]["resolution_time"], "2024-01-20T12:00:00Z")
        self.assertNotEqual(accepted[0]["label"]["resolution_time"], "2024-01-11T00:00:00Z")

    def test_wrong_hash_or_unknown_sample_cannot_be_annotated(self):
        for change in ("hash", "sample"):
            annotations = copy.deepcopy(self.annotations)
            if change == "hash":
                annotations["input_hashes"]["data"] = "wrong"
            else:
                annotations["entries"]["not-in-dataset"] = {}
            with self.subTest(change=change), self.assertRaises(ValidationError):
                self.promote(annotations)

    def test_temporal_and_provenance_gaps_cannot_be_overridden_by_include(self):
        candidate = self.candidates[0]
        for key, value in (("question_available_at", "2024-01-11T00:00:00Z"),
                           ("observation_time", "2024-01-12T00:00:00Z"),
                           ("resolution_time", "2024-01-01T00:00:00Z"),
                           ("label_available_at", None), ("provenance", "")):
            annotation = included_annotation(candidate)
            annotation[key] = value
            self.annotations["entries"][candidate.sample_id] = annotation
            with self.subTest(key=key), self.assertRaises(ValidationError):
                self.promote(self.annotations)

    def test_sibling_contracts_cannot_be_split_into_different_groups(self):
        for index, candidate in enumerate(self.candidates):
            annotation = included_annotation(candidate)
            annotation["event_group_id"] = f"different-{index}"
            self.annotations["entries"][candidate.sample_id] = annotation
        with self.assertRaisesRegex(ValidationError, "native event group"):
            self.promote(self.annotations)

    def test_conflicting_event_outcomes_block_all_affected_candidates(self):
        first = self.candidates[0]
        second = replace(first, sample_id="different-snapshot", outcome=0, blockers=[])
        audit_candidates([first, second])
        self.assertIn("conflicting_event_outcomes", first.blockers)
        self.assertIn("conflicting_event_outcomes", second.blockers)
        self.annotations["entries"][first.sample_id] = included_annotation(first)
        with self.assertRaisesRegex(ValidationError, "cannot include"):
            self.promote(self.annotations)


class ForecastPromotionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.q, self.r = root / "q.json", root / "r.json"
        write_forecast(self.q, self.r)
        self.candidates, _ = forecastbench_candidates(self.q, self.r)
        self.inputs = {"data": sha256_bytes(self.q.read_bytes()),
                       "resolutions": sha256_bytes(self.r.read_bytes())}

    def test_annotated_forecast_preserves_frozen_price_and_resolved_outcome(self):
        annotations = annotation_template("forecastbench", self.inputs, self.candidates)
        candidate = self.candidates[0]
        annotations["entries"][candidate.sample_id] = included_annotation(candidate)
        accepted, _ = promote_candidates(self.candidates, "forecastbench", REVISION, self.inputs, annotations)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["label"]["outcome"], 1)
        self.assertEqual(accepted[0]["market"]["observed_at"], "2024-01-01T00:00:00Z")
        self.assertEqual(accepted[0]["observation_time"], "2024-01-10T12:00:00Z")

    def test_due_day_and_resolution_date_cannot_be_moved(self):
        candidate = self.candidates[0]
        for key, value in (("observation_time", "2024-01-11T00:00:00Z"),
                           ("resolution_time", "2024-01-19T12:00:00Z")):
            annotations = annotation_template("forecastbench", self.inputs, self.candidates)
            entry = included_annotation(candidate)
            entry[key] = value
            annotations["entries"][candidate.sample_id] = entry
            with self.subTest(key=key), self.assertRaises(ValidationError):
                promote_candidates(self.candidates, "forecastbench", REVISION, self.inputs, annotations)

    def test_unresolved_source_cannot_be_promoted_even_with_complete_times(self):
        write_forecast(self.q, self.r, resolved=False, value=0.7)
        candidates, _ = forecastbench_candidates(self.q, self.r)
        inputs = {"data": sha256_bytes(self.q.read_bytes()), "resolutions": sha256_bytes(self.r.read_bytes())}
        annotations = annotation_template("forecastbench", inputs, candidates)
        annotations["entries"][candidates[0].sample_id] = included_annotation(candidates[0])
        with self.assertRaisesRegex(ValidationError, "cannot include"):
            promote_candidates(candidates, "forecastbench", REVISION, inputs, annotations)


class SourceAndCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "native.csv"
        write_prophet(self.data)
        self.registry = self.root / "registry.json"
        write_registry(self.registry, "prophet_arena", self.data)

    def test_download_checks_hash_and_never_publishes_partial_output(self):
        output = self.root / "download"
        with patch("foretellmesh.sources.urlopen", return_value=io.BytesIO(b"corrupted")):
            with self.assertRaisesRegex(ValidationError, "SHA-256 mismatch"):
                download_source(self.registry, "prophet_arena", output)
        self.assertFalse(output.exists())
        with patch("foretellmesh.sources.urlopen", return_value=io.BytesIO(self.data.read_bytes())):
            manifest = download_source(self.registry, "prophet_arena", output)
        self.assertEqual((output / self.data.name).read_bytes(), self.data.read_bytes())
        self.assertEqual(manifest["revision"], REVISION)

    def test_download_size_limit_is_enforced(self):
        registry = json.loads(self.registry.read_text())
        registry["sources"]["prophet_arena"]["artifacts"][self.data.name]["max_bytes"] = 2
        self.registry.write_text(json.dumps(registry))
        with patch("foretellmesh.sources.urlopen", return_value=io.BytesIO(b"too long")):
            with self.assertRaisesRegex(ValidationError, "size limit"):
                download_source(self.registry, "prophet_arena", self.root / "download")

    def test_unpinned_or_traversing_registry_is_rejected(self):
        original = json.loads(self.registry.read_text())
        for kind in ("revision", "path", "url"):
            registry = copy.deepcopy(original)
            source = registry["sources"]["prophet_arena"]
            if kind == "revision":
                source["revision"] = "main"
            elif kind == "path":
                source["artifacts"]["../escape"] = source["artifacts"].pop(self.data.name)
            else:
                source["artifacts"][self.data.name]["url"] = "https://example.org/main/data.csv"
            self.registry.write_text(json.dumps(registry))
            with self.subTest(kind=kind), self.assertRaises(ValidationError):
                load_source(self.registry, "prophet_arena")

    def test_source_hash_mismatch_prevents_import(self):
        self.data.write_text(self.data.read_text() + "\n")
        with self.assertRaisesRegex(ValidationError, "SHA-256 mismatch"):
            ingest_source(source_name="prophet_arena", registry=self.registry, data=self.data,
                          output=self.root / "audit")
        self.assertFalse((self.root / "audit").exists())

    def test_forecastbench_cannot_be_configured_as_training(self):
        config = json.loads(CONFIG.read_text())
        config["sources"]["forecastbench"] = {"version": REVISION, "role": "train_eval", "synthetic": False}
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValidationError, "reserved for evaluation"):
            load_config(path)

    def test_cli_audit_then_annotated_promotion_and_deterministic_replay(self):
        output = self.root / "audit"
        command = [sys.executable, "-m", "foretellmesh", "ingest", "--source", "prophet_arena",
                   "--registry", str(self.registry), "--data", str(self.data), "--output", str(output)]
        run = run_cli(command)
        self.assertEqual(run.returncode, 0, run.stderr)
        audit = json.loads((output / "audit.json").read_text())
        self.assertEqual(audit["accepted_count"], 0)
        with self.assertRaisesRegex(ValidationError, "no records"):
            load_records(output / "records.jsonl")
        candidates, _ = prophet_candidates(self.data)
        annotations = json.loads((output / "annotations.template.json").read_text())
        annotations["entries"][candidates[0].sample_id] = included_annotation(candidates[0])
        annotations_path = self.root / "annotations.json"
        annotations_path.write_text(json.dumps(annotations))
        args = dict(source_name="prophet_arena", registry=self.registry, data=self.data,
                    annotations_path=annotations_path)
        first = ingest_source(**args, output=self.root / "accepted-a")
        second = ingest_source(**args, output=self.root / "accepted-b")
        self.assertEqual(first, second)
        records, _ = load_records(self.root / "accepted-a/records.jsonl")
        self.assertEqual(len(records), 1)
        self.assertNotIn("outcome", json.dumps(records[0].forecast_input.to_payload()))
        self.assertEqual(first["records_sha256"], sha256_bytes((self.root / "accepted-a/records.jsonl").read_bytes()))


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow dependency not installed")
class ParquetAdapterTests(unittest.TestCase):
    def test_pma_requires_pre_resolution_snapshot_and_later_finalized_label(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "markets.parquet"
            base = {"ticker": "MARKET-A", "event_ticker": "EVENT-A", "title": "Synthetic event A?",
                    "market_type": "binary", "yes_bid": 60, "yes_ask": 80}
            rows = [{**base, "status": "open", "result": "", "_fetched_at": "2024-01-10T00:00:00Z"},
                    {**base, "status": "finalized", "result": "yes", "_fetched_at": "2024-01-21T00:00:00Z"}]
            pq.write_table(pa.Table.from_pylist(rows), path)
            candidates, _ = kalshi_snapshot_candidates(path)
            self.assertEqual(candidates[0].outcome, 1)
            self.assertEqual(candidates[0].market["probability"], 0.7)
            self.assertEqual(candidates[0].blockers, [])
            self.assertIn("not_a_pre_resolution_open_snapshot", candidates[1].blockers)
            inputs = {"data": sha256_bytes(path.read_bytes())}
            annotations = annotation_template("prediction_market_analysis", inputs, candidates)
            entry = included_annotation(candidates[0])
            entry["label_available_at"] = "2024-01-20T23:00:00Z"
            annotations["entries"][candidates[0].sample_id] = entry
            with self.assertRaisesRegex(ValidationError, "archived finalized snapshot"):
                promote_candidates(candidates, "prediction_market_analysis", "sha256:fixture", inputs, annotations)

    def test_naive_parquet_timestamp_is_not_silently_assumed_utc(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "naive.parquet"
            pq.write_table(pa.Table.from_pylist([{
                "ticker": "M", "event_ticker": "E", "title": "Synthetic?", "market_type": "binary",
                "status": "open", "result": "", "_fetched_at": "2024-01-10T00:00:00",
            }]), path)
            with self.assertRaisesRegex(ValidationError, "timezone is required"):
                kalshi_snapshot_candidates(path)


if __name__ == "__main__":
    unittest.main()
