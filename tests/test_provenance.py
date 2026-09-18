"""Synthetic proofs exercise the real historical-verification path offline."""

import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.config import load_config
from foretellmesh.data import load_records, sha256_bytes
from foretellmesh.evaluation import evaluate, json_text
from foretellmesh.provenance import (
    HF_API, HF_DATA, MANIFOLD_API, download_provenance, load_plan,
    resolution_projection, verify_forecastbench_subset,
)
from foretellmesh.schema import ValidationError
from tests.test_ingestion import REVISION, write_registry
from tests.test_pipeline import CONFIG, DATA


class ProofTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.proofs = self.root / "proofs"
        self.proofs.mkdir()
        self.questions = self.root / "questions.json"
        self.resolutions = self.root / "resolutions.json"
        self.registry = self.root / "registry.json"
        self.plan_path = self.root / "plan.json"
        self.release_revision = "b" * 40
        questions, resolutions = [], []
        for market_id, resolved, outcome in (("one", True, 1.0), ("two", True, 0.0), ("pending", False, 0.9)):
            questions.append({"id": market_id, "source": "manifold", "question": f"Synthetic event {market_id}?",
                              "freeze_datetime": "2024-01-01T00:00:00Z", "freeze_datetime_value": "0.7",
                              "resolution_dates": "N/A", "resolution_criteria": "Synthetic criterion"})
            resolutions.append({"id": market_id, "source": "manifold", "direction": None,
                                "resolution_date": "2024-01-20", "resolved": resolved, "resolved_to": outcome})
        common = {"forecast_due_date": "2024-01-10", "question_set": "synthetic-round.json"}
        self.questions.write_text(json_text({**common, "questions": questions}))
        self.resolutions.write_text(json_text({**common, "resolutions": resolutions}))
        write_registry(self.registry, "forecastbench", self.questions, self.resolutions)
        (self.proofs / "question.json").write_bytes(self.questions.read_bytes())
        (self.proofs / "question_commit.json").write_text(json_text([
            {"id": self.release_revision, "date": "2024-01-01T00:05:00Z"}]))
        (self.proofs / "label_commit.json").write_text(json_text([
            {"id": REVISION, "date": "2024-03-01T00:00:00Z"}]))
        for market_id, day, outcome in (("one", 20, "YES"), ("two", 21, "NO")):
            millis = int(datetime(2024, 1, day, 12, tzinfo=timezone.utc).timestamp() * 1000)
            (self.proofs / f"{market_id}.json").write_text(json_text({
                "id": market_id, "outcomeType": "BINARY", "isResolved": True,
                "resolution": outcome, "resolutionTime": millis, "probability": 0.999,
                "question": "UNTRUSTED current question", "description": "UNTRUSTED later evidence",
            }))
        artifacts = {}
        for name, purpose, url, market_id in (
            ("question.json", "question_snapshot", HF_DATA + self.release_revision + "/datasets/question_sets/synthetic-round.json", None),
            ("question_commit.json", "question_commit", HF_API + self.release_revision + "?limit=1", None),
            ("label_commit.json", "label_commit", HF_API + REVISION + "?limit=1", None),
            ("one.json", "market_resolution", MANIFOLD_API + "one", "one"),
            ("two.json", "market_resolution", MANIFOLD_API + "two", "two"),
        ):
            raw = (self.proofs / name).read_bytes()
            hashed = json_text(resolution_projection(json.loads(raw))).encode() if market_id else raw
            artifacts[name] = {"url": url, "sha256": sha256_bytes(hashed), "max_bytes": 100000,
                               "purpose": purpose, "market_id": market_id}
        self.plan = {
            "schema_version": "1", "source_revision": REVISION, "question_release_revision": self.release_revision,
            "question_publication_time": "2024-01-01T00:05:00Z", "label_publication_time": "2024-03-01T00:00:00Z",
            "observation_time": "2024-01-10T12:00:00Z", "cohort": "all_manifold_questions_in_round",
            "event_groups": {"one": "synthetic-group-one", "two": "synthetic-group-two"},
            "grouping_notes": "Synthetic groups for offline regression tests", "artifacts": artifacts,
        }
        self.save_plan()

    def save_plan(self):
        self.plan_path.write_text(json_text(self.plan))

    def verify(self, suffix="result"):
        return verify_forecastbench_subset(plan_path=self.plan_path, registry=self.registry,
                                           data=self.questions, resolutions=self.resolutions,
                                           proofs=self.proofs, output=self.root / suffix)

    def update_proof(self, name, transform, *, update_hash=False):
        path = self.proofs / name
        raw = json.loads(path.read_text())
        transform(raw)
        path.write_text(json_text(raw))
        if update_hash:
            content = path.read_bytes()
            if self.plan["artifacts"][name]["purpose"] == "market_resolution":
                content = json_text(resolution_projection(raw)).encode()
            self.plan["artifacts"][name]["sha256"] = sha256_bytes(content)
            self.save_plan()

    def test_verified_record_uses_historical_features_and_conservative_availability(self):
        report = self.verify()
        self.assertEqual((report["cohort_count"], report["resolved_in_benchmark_count"], report["verified_count"]), (3, 2, 1))
        records, _ = load_records(self.root / "result/import/records.jsonl")
        self.assertEqual(len(records), 1)
        record = records[0]
        payload = record.forecast_input.to_payload()
        self.assertNotIn("UNTRUSTED", json.dumps(payload))
        self.assertNotIn("resolutionTime", json.dumps(payload))
        self.assertEqual(payload["market"]["probability"], 0.7)
        self.assertEqual(payload["market"]["available_at"], "2024-01-01T00:05:00Z")
        self.assertEqual(record.label.available_at.isoformat(), "2024-03-01T00:00:00+00:00")
        self.assertEqual(record.label.resolution_time.isoformat(), "2024-01-20T12:00:00+00:00")
        reasons = {item["native_id"]: item["reasons"] for item in report["checks"]}
        self.assertIn("platform_resolution_date_disagrees_with_benchmark", reasons["two"])
        self.assertIn("unresolved_or_missing_resolution", reasons["pending"])

    def test_current_prices_and_descriptions_cannot_change_outputs(self):
        first = self.verify("before")
        self.update_proof("one.json", lambda raw: raw.update(probability=0.01, description="DIFFERENT LIVE CONTENT"))
        second = self.verify("after")
        self.assertEqual(first, second)
        self.assertEqual((self.root / "before/import/records.jsonl").read_bytes(),
                         (self.root / "after/import/records.jsonl").read_bytes())

    def test_changed_resolution_fails_pinned_fact_hash(self):
        self.update_proof("one.json", lambda raw: raw.update(resolution="NO"))
        with self.assertRaisesRegex(ValidationError, "SHA-256 mismatch"):
            self.verify()
        self.assertFalse((self.root / "result").exists())

    def test_platform_outcome_conflict_is_quarantined_not_rewritten(self):
        self.update_proof("one.json", lambda raw: raw.update(resolution="NO"), update_hash=True)
        report = self.verify()
        self.assertEqual(report["verified_count"], 0)
        self.assertTrue(any("platform_outcome_disagrees_with_benchmark" in row["reasons"] for row in report["checks"]))

    def test_revised_question_file_cannot_pass_as_historical_content(self):
        changed = json.loads(self.questions.read_text())
        changed["questions"][0]["question"] = "A revised question after the cutoff"
        self.questions.write_text(json_text(changed))
        write_registry(self.registry, "forecastbench", self.questions, self.resolutions)
        with self.assertRaisesRegex(ValidationError, "differs from.*historical"):
            self.verify()

    def test_commit_metadata_must_support_declared_timestamp(self):
        self.update_proof("question_commit.json", lambda rows: rows[0].update(date="2024-01-11T00:00:00Z"), update_hash=True)
        with self.assertRaisesRegex(ValidationError, "commit metadata disagrees"):
            self.verify()

    def test_question_publication_after_cutoff_is_rejected(self):
        self.plan["question_publication_time"] = "2024-01-11T00:00:00Z"
        self.save_plan()
        with self.assertRaisesRegex(ValidationError, "question publication"):
            load_plan(self.plan_path)

    def test_removing_a_resolved_candidate_from_the_plan_is_rejected(self):
        del self.plan["event_groups"]["two"]
        del self.plan["artifacts"]["two.json"]
        self.save_plan()
        with self.assertRaisesRegex(ValidationError, "every resolved"):
            self.verify()

    def test_download_is_bounded_and_verified_before_publication(self):
        by_url = {artifact["url"]: (self.proofs / name).read_bytes() for name, artifact in self.plan["artifacts"].items()}
        with patch("foretellmesh.provenance.urlopen", side_effect=lambda url, timeout: io.BytesIO(by_url[url])):
            manifest = download_provenance(self.plan_path, self.root / "download")
        self.assertEqual(len(manifest["artifacts"]), 5)
        self.assertEqual(manifest["artifacts"]["one.json"]["hash_scope"], "resolution_fields")
        with patch("foretellmesh.provenance.urlopen", return_value=io.BytesIO(b"{}")):
            with self.assertRaises(ValidationError):
                download_provenance(self.plan_path, self.root / "bad-download")
        self.assertFalse((self.root / "bad-download").exists())

    def test_untrusted_endpoint_or_output_path_is_rejected(self):
        original = copy.deepcopy(self.plan)
        for mutation in ("url", "path"):
            self.plan = copy.deepcopy(original)
            if mutation == "url":
                self.plan["artifacts"]["one.json"]["url"] = "https://example.org/other"
            else:
                self.plan["artifacts"]["../outside"] = self.plan["artifacts"].pop("one.json")
            self.save_plan()
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                load_plan(self.plan_path)


class HeldoutBaselineTests(unittest.TestCase):
    def test_explicit_no_fit_mode_evaluates_without_any_training_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [json.loads(line) for line in DATA.read_text().splitlines()]
            rows = [row for row in rows if row["sample_id"] in ("test-f", "test-g", "test-h")]
            data = root / "heldout.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = json.loads(CONFIG.read_text())
            config["baselines"] = ["constant_0_5", "market"]
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            report, predictions, _ = evaluate(data, config_path)
            self.assertIsNone(report["empirical_base_rate"])
            self.assertEqual(report["audit"]["retained_counts"]["train"], 0)
            self.assertEqual(set(report["evaluation"]["test"]["baselines"]), {"constant_0_5", "market"})
            self.assertTrue(all("empirical_base_rate" not in row["probabilities"] for row in predictions))
            config.pop("baselines")
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValidationError, "no eligible training"):
                evaluate(data, config_path)

    def test_invalid_baseline_config_is_rejected(self):
        original = json.loads(CONFIG.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for baselines in ([], ["unknown"], ["market", "market"], "market", [True]):
                path.write_text(json.dumps({**original, "baselines": baselines}))
                with self.subTest(baselines=baselines), self.assertRaises(ValidationError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
