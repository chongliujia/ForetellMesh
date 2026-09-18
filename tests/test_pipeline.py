import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from foretellmesh.config import load_config
from foretellmesh.data import load_records, sha256_bytes, strict_json, validate_identities
from foretellmesh.evaluation import evaluate
from foretellmesh.metrics import score_predictions
from foretellmesh.schema import ValidationError, parse_record, probability, timestamp
from foretellmesh.splits import split_records


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/synthetic_forecasts.jsonl"
CONFIG = ROOT / "configs/synthetic_baselines_v1.json"


def run_cli(command):
    return subprocess.run(command, capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


def sample(**changes):
    raw = {
        "sample_id": "sample-a", "dataset_source": "synthetic_training", "dataset_version": "1",
        "event_id": "a", "event_group_id": "a", "question": "Will synthetic event A occur?",
        "observation_time": "2024-01-10T00:00:00Z",
        "evidence": [{"evidence_id": "e1", "text": "Artificial historical observation.",
                      "source": "synthetic://a", "published_at": "2024-01-09T00:00:00Z",
                      "available_at": "2024-01-10T00:00:00Z"}],
        "market": {"probability": 0.7, "observed_at": "2024-01-09T00:00:00Z",
                   "available_at": "2024-01-10T00:00:00Z"},
        "label": {"outcome": 1, "resolution_time": "2024-01-20T00:00:00Z",
                  "available_at": "2024-01-21T00:00:00Z"},
    }
    raw.update(changes)
    return raw


class SchemaTests(unittest.TestCase):
    def test_label_and_metadata_never_enter_model_payload(self):
        record = parse_record(sample())
        payload = record.forecast_input.to_payload()
        self.assertEqual(set(payload), {"question", "observation_time", "evidence", "market"})
        text = json.dumps(payload)
        for forbidden in ("outcome", "resolution_time", "event_group_id", "dataset_source", "split"):
            self.assertNotIn(forbidden, text)
        altered = sample(label={"outcome": 0, "resolution_time": "2024-01-25T00:00:00Z",
                                "available_at": "2024-01-25T00:00:00Z"})
        self.assertEqual(payload, parse_record(altered).forecast_input.to_payload())

    def test_timezone_normalization_and_naive_rejection(self):
        self.assertEqual(timestamp("2024-01-10T08:00:00+08:00", "time"),
                         datetime(2024, 1, 10, tzinfo=timezone.utc))
        for value in ("2024-01-10", "2024-01-10T00:00:00", "invalid", None,
                      "0001-01-01T00:00:00+01:00"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                timestamp(value, "time")

    def test_probabilities_reject_nonfinite_out_of_range_and_bool(self):
        for value in (-0.1, 1.1, math.nan, math.inf, -math.inf, True, "0.5", None, 10**400):
            with self.subTest(value=str(value)), self.assertRaises(ValidationError):
                probability(value)
        for value in (0, 0.5, 1):
            self.assertEqual(probability(value), value)

    def test_evidence_and_market_future_or_backdated_availability_rejected(self):
        for kind, time_key in (("evidence", "published_at"), ("market", "observed_at")):
            for field, value in ((time_key, "2024-01-11T00:00:00Z"),
                                 ("available_at", "2024-01-11T00:00:00Z"),
                                 ("available_at", "2024-01-08T00:00:00Z")):
                raw = sample()
                target = raw[kind][0] if kind == "evidence" else raw[kind]
                target[field] = value
                with self.subTest(kind=kind, field=field, value=value), self.assertRaises(ValidationError):
                    parse_record(raw)

    def test_evidence_at_cutoff_is_allowed(self):
        raw = sample()
        raw["evidence"][0]["published_at"] = raw["observation_time"]
        raw["market"]["observed_at"] = raw["observation_time"]
        parse_record(raw)

    def test_unknown_or_missing_fields_and_duplicate_evidence_rejected(self):
        cases = []
        raw = sample(outcome=1)
        cases.append(raw)
        raw = sample()
        del raw["evidence"][0]["available_at"]
        cases.append(raw)
        raw = sample()
        raw["evidence"].append(copy.deepcopy(raw["evidence"][0]))
        cases.append(raw)
        raw = sample()
        raw["market"]["future_price"] = 1
        cases.append(raw)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                parse_record(raw)

    def test_labels_require_binary_outcomes_and_future_resolution(self):
        for value in (True, 1.0, "1", 2, None):
            raw = sample()
            raw["label"]["outcome"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                parse_record(raw)
        for key, value in (("resolution_time", "2024-01-10T00:00:00Z"),
                           ("available_at", "2024-01-19T00:00:00Z")):
            raw = sample()
            raw["label"][key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                parse_record(raw)
        self.assertIsNone(parse_record(sample(label=None, market=None)).label)


class DataAndSplitTests(unittest.TestCase):
    def setUp(self):
        self.config, _ = load_config(CONFIG)

    def test_strict_json_rejects_duplicate_keys_and_nonstandard_numbers(self):
        for text in ('{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}', '{broken'):
            with self.subTest(text=text), self.assertRaises(ValidationError):
                strict_json(text)

    def test_identity_collisions_and_conflicting_labels_rejected(self):
        first = parse_record(sample())
        cases = [sample(), sample(sample_id="b"),
                 sample(sample_id="b", event_id="b", event_group_id="b"),
                 sample(sample_id="b", observation_time="2024-01-11T00:00:00Z",
                        event_group_id="b", question="Different wording for A")]
        raw = sample(sample_id="b", observation_time="2024-01-11T00:00:00Z")
        raw["label"]["outcome"] = 0
        cases.append(raw)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                validate_identities([first, parse_record(raw)])

    def test_normalized_cross_source_question_requires_shared_group(self):
        first = parse_record(sample())
        second = parse_record(sample(sample_id="b", dataset_source="synthetic_benchmark",
                                     event_group_id="b", question="  WILL synthetic event A occur?  "))
        with self.assertRaisesRegex(ValidationError, "equivalent question"):
            validate_identities([first, second])

    def test_cross_boundary_group_is_dropped_in_full(self):
        label = {"outcome": 1, "resolution_time": "2024-03-20T00:00:00Z",
                 "available_at": "2024-03-20T00:00:00Z"}
        records = [parse_record(sample(label=label)),
                   parse_record(sample(sample_id="late", observation_time="2024-03-01T00:00:00Z",
                                       label=label))]
        result = split_records(records, self.config)
        self.assertEqual(sum(map(len, result.partitions.values())), 0)
        self.assertEqual([item["reason"] for item in result.exclusions],
                         ["group_crosses_time_boundary"] * 2)

    def test_heldout_source_reserves_entire_group_from_training(self):
        records = [parse_record(sample()),
                   parse_record(sample(sample_id="heldout", dataset_source="synthetic_benchmark"))]
        result = split_records(records, self.config)
        self.assertEqual(result.partitions["train"], [])
        self.assertEqual(len(result.exclusions), 2)
        self.assertTrue(all(item["reason"] == "heldout_source_group_in_train_window"
                            for item in result.exclusions))

    def test_label_availability_not_resolution_controls_training_eligibility(self):
        raw = sample()
        raw["label"]["available_at"] = "2024-02-01T00:00:01Z"
        result = split_records([parse_record(raw)], self.config)
        self.assertEqual(result.exclusions[0]["reason"], "label_unavailable_at_split_cutoff")
        raw["label"]["available_at"] = "2024-02-01T00:00:00Z"
        self.assertEqual(len(split_records([parse_record(raw)], self.config).partitions["train"]), 1)

    def test_split_boundaries_are_left_inclusive(self):
        for date, split in (("2024-02-01T00:00:00Z", "validation"),
                            ("2024-03-01T00:00:00Z", "test")):
            resolution = date.replace("-01T", "-20T")
            record = parse_record(sample(observation_time=date, label={
                "outcome": 1, "resolution_time": resolution, "available_at": resolution,
            }))
            with self.subTest(split=split):
                self.assertEqual(len(split_records([record], self.config).partitions[split]), 1)

    def test_unknown_source_or_version_rejected(self):
        for change in ({"dataset_source": "unknown"}, {"dataset_version": "wrong"}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                split_records([parse_record(sample(**change))], self.config)

    def test_fixture_exclusions_deduplication_and_group_isolation(self):
        records, _ = load_records(DATA)
        result = split_records(records, self.config)
        self.assertEqual({key: len(value) for key, value in result.partitions.items()},
                         {"train": 3, "validation": 2, "test": 3})
        self.assertEqual(len(result.exclusions), 9)
        self.assertIn("test-f", [r.sample_id for r in result.partitions["test"]])
        self.assertIn({"sample_id": "duplicate-test-f", "event_group_id": "f",
                       "reason": "duplicate_question_observation"}, result.exclusions)
        groups = [{r.event_group_id for r in rows} for rows in result.partitions.values()]
        for index, group in enumerate(groups):
            for other in groups[index + 1:]:
                self.assertFalse(group & other)
        self.assertEqual(len(result.manifest()), len(records))
        self.assertEqual(result.manifest(), split_records(list(reversed(records)), self.config).manifest())

    def test_config_rejects_invalid_dates_metrics_and_source_roles(self):
        original = json.loads(CONFIG.read_text())
        cases = [{**original, "test_start": original["validation_start"]},
                 {**original, "ece_bins": True}, {**original, "log_loss_epsilon": 0},
                 {**original, "sources": {"x": {"version": "1", "role": "anything", "synthetic": False}}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for raw in cases:
                path.write_text(json.dumps(raw))
                with self.subTest(raw=raw), self.assertRaises(ValidationError):
                    load_config(path)

    def test_loader_reports_line_number_for_bad_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(sample()) + '\n{"unexpected": true}\n')
            with self.assertRaisesRegex(ValidationError, "bad.jsonl:2"):
                load_records(path)
            path.write_text("\n")
            with self.assertRaisesRegex(ValidationError, "no records"):
                load_records(path)


class MetricsTests(unittest.TestCase):
    def test_hand_computed_scores_and_curve(self):
        result = score_predictions([0.1, 0.4, 0.8, 1.0], [0, 1, 1, 0], ece_bins=2,
                                   log_loss_epsilon=0.01)
        self.assertAlmostEqual(result["brier"], (0.01 + 0.36 + 0.04 + 1) / 4)
        self.assertAlmostEqual(result["log_loss"], -math.log(0.9 * 0.4 * 0.8 * 0.01) / 4)
        self.assertAlmostEqual(result["ece"], (0.25 * 2 + 0.4 * 2) / 4)
        self.assertEqual([item["count"] for item in result["calibration_curve"]], [2, 2])
        self.assertAlmostEqual(result["calibration_curve"][1]["mean_probability"], 0.9)

    def test_perfect_predictions_and_endpoint_log_safety(self):
        result = score_predictions([0, 1], [0, 1])
        self.assertEqual((result["brier"], result["log_loss"], result["ece"]), (0, 0, 0))
        for epsilon in (1e-15, 1e-30):
            wrong = score_predictions([1, 0], [0, 1], log_loss_epsilon=epsilon)
            self.assertAlmostEqual(wrong["log_loss"], -math.log(epsilon))
            self.assertEqual(wrong["brier"], 1)

    def test_missing_predictions_and_empty_sets_have_explicit_coverage(self):
        result = score_predictions([None, 0.5], [1, 0])
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["brier"], 0.25)
        for predictions, outcomes, coverage in (([], [], None), ([None], [1], 0)):
            result = score_predictions(predictions, outcomes)
            self.assertEqual(result["coverage"], coverage)
            for metric in ("brier", "log_loss", "ece"):
                self.assertIsNone(result[metric])

    def test_invalid_predictions_do_not_silently_disappear(self):
        for predictions, outcomes in (([math.nan], [1]), ([True], [1]), ([0.5], [2]),
                                      ([None], [True]), ([0.5], [])):
            with self.subTest(predictions=predictions), self.assertRaises(ValidationError):
                score_predictions(predictions, outcomes)

    def test_ece_bin_edges_and_one_are_included(self):
        result = score_predictions([0, 0.5, 1], [0, 1, 1], ece_bins=2)
        self.assertEqual([item["count"] for item in result["calibration_curve"]], [1, 2])


class EndToEndTests(unittest.TestCase):
    def test_scores_use_train_only_and_market_matched_support(self):
        report, predictions, _ = evaluate(DATA, CONFIG)
        self.assertEqual(report["empirical_base_rate"]["probability"], 2 / 3)
        test = report["evaluation"]["test"]
        self.assertAlmostEqual(test["baselines"]["empirical_base_rate"]["brier"], 1 / 3)
        self.assertEqual(test["baselines"]["market"]["coverage"], 2 / 3)
        self.assertAlmostEqual(test["baselines"]["market"]["brier"], 0.05125)
        paired = test["market_matched"]
        self.assertEqual(paired["sample_count"], 2)
        self.assertAlmostEqual(paired["baselines"]["empirical_base_rate"]["brier"], 5 / 18)
        self.assertEqual(len(predictions), 5)

    def test_replay_is_deterministic_and_hashes_match_artifacts(self):
        first, rows_a, split_a = evaluate(DATA, CONFIG)
        second, rows_b, split_b = evaluate(DATA, CONFIG)
        self.assertEqual(first, second)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(split_a.manifest(), split_b.manifest())
        self.assertEqual(first["experiment"]["dataset_sha256"], sha256_bytes(DATA.read_bytes()))
        self.assertEqual(first["experiment"]["config_sha256"], sha256_bytes(CONFIG.read_bytes()))

    def test_test_outcomes_do_not_change_fitted_base_rate(self):
        rows = [json.loads(line) for line in DATA.read_text().splitlines()]
        # Both copies of the duplicate event must keep consistent labels.
        for row in rows:
            if row["event_group_id"] in ("f", "g", "h"):
                row["label"]["outcome"] = 1 - row["label"]["outcome"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report, _, _ = evaluate(path, CONFIG)
        self.assertEqual(report["empirical_base_rate"]["probability"], 2 / 3)

    def test_missing_training_or_test_data_is_an_error(self):
        all_rows = DATA.read_text().splitlines()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset.jsonl"
            for ids, message in (({"test-f"}, "no eligible training"),
                                 ({"train-a", "train-b", "train-c"}, "no eligible test")):
                path.write_text("\n".join(line for line in all_rows
                                          if json.loads(line)["sample_id"] in ids))
                with self.subTest(ids=ids), self.assertRaisesRegex(ValidationError, message):
                    evaluate(path, CONFIG)

    def test_empty_validation_and_absent_market_report_null_metrics(self):
        rows = [json.loads(line) for line in DATA.read_text().splitlines()]
        rows = [row for row in rows if row["sample_id"] in ("train-a", "train-b", "test-h")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no-market.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report, _, _ = evaluate(path, CONFIG)
        self.assertEqual(report["evaluation"]["validation"]["sample_count"], 0)
        self.assertIsNone(report["evaluation"]["validation"]["baselines"]["constant_0_5"]["brier"])
        test = report["evaluation"]["test"]
        self.assertEqual(test["baselines"]["market"]["coverage"], 0)
        self.assertEqual(test["market_matched"]["sample_count"], 0)
        self.assertIsNone(test["market_matched"]["baseline_minus_market"]["empirical_base_rate"]["brier"])

    def test_cli_writes_complete_artifacts_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            command = [sys.executable, "-m", "foretellmesh", "evaluate", "--data", str(DATA),
                       "--config", str(CONFIG), "--output", str(output)]
            first = run_cli(command)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual({path.name for path in output.iterdir()},
                             {"report.json", "report.md", "split_manifest.json", "predictions.jsonl", "config.json"})
            before = (output / "report.json").read_bytes()
            report = json.loads(before)
            self.assertTrue(report["synthetic_only"])
            self.assertIn("Synthetic demonstration only", (output / "report.md").read_text())
            self.assertEqual(report["experiment"]["split_manifest_sha256"],
                             sha256_bytes((output / "split_manifest.json").read_bytes()))
            self.assertEqual(report["experiment"]["config_sha256"],
                             sha256_bytes((output / "config.json").read_bytes()))
            second = run_cli(command)
            self.assertEqual(second.returncode, 2)
            self.assertIn("output already exists", second.stderr)
            self.assertEqual(before, (output / "report.json").read_bytes())

    def test_cli_rejects_temporal_leakage_without_publishing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "unsafe.jsonl"
            output = Path(directory) / "run"
            raw = sample()
            raw["evidence"][0]["available_at"] = "2025-01-01T00:00:00Z"
            data.write_text(json.dumps(raw))
            result = run_cli(
                [sys.executable, "-m", "foretellmesh", "evaluate", "--data", str(data),
                 "--config", str(CONFIG), "--output", str(output)],
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("available_at <= observation_time", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
