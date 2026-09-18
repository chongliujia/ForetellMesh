import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh.config import SourcePolicy
from foretellmesh.data import sha256_bytes
from foretellmesh.evaluation import json_text
from foretellmesh.schema import ValidationError, parse_record
from foretellmesh.sft_data import (build_sft, encode_sft_pair, jsonl, model_text,
                                  validate_response, validate_target)
from foretellmesh.synthetic_sft import (SOURCE, VERSION, canonical_hash, generate_synthetic_sft,
                                       oracle, render_case)
from tests.test_lora_probe import CharacterTokenizer


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "configs/synthetic_sft_generator_v1.json"
SPLITS = ROOT / "configs/synthetic_sft_splits_v1.json"


class SFTOracleTests(unittest.TestCase):
    def test_six_independently_known_answers(self):
        cases = [
            ("complement", {"a": 1, "b": 4}, Fraction(3, 4)),
            ("at_least_one", {"a": 1, "b": 4, "n": 2}, Fraction(7, 16)),
            ("without_replacement", {"total": 10, "successes": 4, "draws": 2}, Fraction(2, 15)),
            ("mixture", {"weight": 25, "low": 20, "high": 80}, Fraction(7, 20)),
            ("beta_binomial", {"successes": 3, "failures": 1}, Fraction(2, 3)),
            ("bayes_signal", {"prior": 20, "sensitivity": 80, "false_positive": 10}, Fraction(2, 3)),
        ]
        for family, parameters, expected in cases:
            with self.subTest(family=family):
                self.assertEqual(oracle(family, parameters)[0], expected)

    def test_epistemic_confidence_is_not_event_likelihood(self):
        self.assertEqual(oracle("complement", {"a": 9, "b": 10})[2], "high")
        self.assertEqual(oracle("beta_binomial", {"successes": 3, "failures": 1})[2], "low")
        self.assertEqual(oracle("beta_binomial", {"successes": 89, "failures": 9})[2], "medium")

    def test_invalid_oracle_parameters_fail(self):
        for family, parameters in [
            ("complement", {"a": True, "b": 4}), ("complement", {"a": 5, "b": 4}),
            ("mixture", {"weight": 101, "low": 20, "high": 80}),
            ("bayes_signal", {"prior": 10, "sensitivity": 80, "false_positive": 100}),
            ("without_replacement", {"total": 10, "successes": 2, "draws": 3}),
        ]:
            with self.subTest(family=family), self.assertRaises(ValidationError):
                oracle(family, parameters)

    def test_bilingual_examples_share_event_and_simulated_outcome(self):
        case = {"version": VERSION, "family": "complement", "parameters": {"a": 1, "b": 4},
                "language": "en", "observation_time": "2024-01-15T00:00:00Z"}
        en, en_target = render_case(case)
        zh, zh_target = render_case({**case, "language": "zh"})
        self.assertEqual(en["event_group_id"], zh["event_group_id"])
        self.assertEqual(en["label"], zh["label"])
        self.assertEqual(en_target["probability"], zh_target["probability"])
        self.assertNotEqual(en["question"], zh["question"])


class SFTDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        generator = json.loads(GENERATOR.read_text())
        generator["groups_per_family"] = {s: 1 for s in ("train", "validation", "test")}
        self.generator = self.root / "generator.json"
        self.generator.write_text(json.dumps(generator))
        self.raw = self.root / "raw"
        generate_synthetic_sft(self.generator, self.raw)
        self.records = [json.loads(line) for line in (self.raw / "records.jsonl").read_text().splitlines()]
        self.targets = [json.loads(line) for line in (self.raw / "targets.jsonl").read_text().splitlines()]
        self.config = self.root / "splits.json"
        self.config.write_bytes(SPLITS.read_bytes())
        entries = [{"event_id": "heldout:a", "event_group_id": "heldout:a", "question": "Unrelated real benchmark question?"}]
        self.index = self.root / "heldout.json"
        self.index_data = {"schema_version": "1", "source": "forecastbench", "dataset_version": "fixture",
                           "input_hashes": {}, "entries": entries, "entries_sha256": canonical_hash(entries),
                           "coverage": "test fixture only"}
        self.index.write_text(json_text(self.index_data))
        self.source = SourcePolicy(VERSION, "train_eval", True)

    def write_inputs(self):
        (self.raw / "records.jsonl").write_text(jsonl(self.records))
        (self.raw / "targets.jsonl").write_text(jsonl(self.targets))

    def build(self, name="bundle", review=None):
        self.write_inputs()
        return build_sft(self.raw / "records.jsonl", self.raw / "targets.jsonl", self.config,
                         self.index, self.root / name, review)

    def test_deterministic_generation_and_build(self):
        other = self.root / "reproduced"
        generate_synthetic_sft(self.generator, other)
        for path in self.raw.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (other / path.relative_to(self.raw)).read_bytes())
        first = self.build()
        second = self.build("reproduced_bundle")
        self.assertEqual(first, second)
        self.assertEqual(first["counts"], {"train": 12, "validation": 12, "test": 12})
        self.assertTrue(first["ready_for_sft"])
        groups = {}
        for line in (self.root / "bundle/metadata.jsonl").read_text().splitlines():
            row = json.loads(line)
            groups.setdefault(row["event_group_id"], set()).add(row["split"])
            proof = self.root / "bundle" / row["provenance"]["artifact"]
            self.assertEqual(sha256_bytes(proof.read_bytes()), row["provenance"]["artifact_sha256"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))

    def test_model_text_is_invariant_to_outcome_and_resolution_changes(self):
        raw, target = self.records[0], self.targets[0]["target"]
        changed = copy.deepcopy(raw)
        changed["label"]["outcome"] = 1 - changed["label"]["outcome"]
        changed["label"]["resolution_time"] = "2024-04-10T00:00:00Z"
        changed["label"]["available_at"] = "2024-04-11T00:00:00Z"
        pair = model_text(parse_record(raw), target)
        self.assertEqual(pair, model_text(parse_record(changed), target))
        payload = json.loads(pair["prompt"].split("Input JSON:\n", 1)[1].split("\nForecast JSON:\n", 1)[0])
        self.assertEqual(set(payload), {"question", "observation_time", "evidence", "market"})
        self.assertNotIn("outcome", json.loads(pair["completion"]))

    def test_oracle_rejects_target_probability_tampering(self):
        self.targets[0]["target"]["probability"] = 0.123456
        with self.assertRaisesRegex(ValidationError, "oracle replay"):
            self.build()

    def test_non_object_supervision_is_rejected(self):
        self.targets[0] = []
        with self.assertRaisesRegex(ValidationError, "target must be an object"):
            self.build()

    def test_tokenizer_revision_must_be_immutable(self):
        from foretellmesh.sft_tokenize import tokenize_sft
        config = json.loads((ROOT / "configs/sft_tokenization_qwen3_v1.json").read_text())
        config["model_revision"] = "main"
        path = self.root / "tokenization.json"
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValidationError, "unsupported tokenization policy"):
            tokenize_sft(self.root / "bundle", self.root / "missing-model.json", path, self.root / "tokens")
        self.assertFalse((self.root / "bundle").exists())

    def test_response_rejects_bad_probability_references_time_and_extras(self):
        record = parse_record(self.records[0])
        target = self.targets[0]["target"]
        for field, value in [("probability", True), ("probability", -0.01), ("probability", float("nan")),
                             ("key_evidence", ["missing"]), ("counter_evidence", ["setup"]),
                             ("observation_time", "2024-06-01T00:00:00Z"), ("outcome", 1)]:
            with self.subTest(field=field), self.assertRaises(ValidationError):
                validate_response({**target, field: value}, record)

    def test_input_and_proof_hash_binding(self):
        record = parse_record(self.records[0])
        for field in ("input_sha256", "artifact_sha256"):
            row = copy.deepcopy(self.targets[0])
            if field == "input_sha256":
                row[field] = "0" * 64
            else:
                row["provenance"][field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, "hash mismatch"):
                validate_target(row, record, self.source, self.raw)

    def test_proof_cannot_escape_target_directory(self):
        row = copy.deepcopy(self.targets[0])
        path = self.root / "outside.json"
        path.write_text("{}")
        row["provenance"].update(artifact="../outside.json", artifact_sha256=sha256_bytes(path.read_bytes()))
        with self.assertRaisesRegex(ValidationError, "within the target directory"):
            validate_target(row, parse_record(self.records[0]), self.source, self.raw)

    def test_future_unknown_and_inconsistent_target_timestamps_rejected(self):
        record = parse_record(self.records[0])
        for field, value in [("issued_at", "2025-01-01T00:00:00Z"),
                             ("question_available_at", None), ("question_available_at", "2025-01-01T00:00:00Z"),
                             ("information_cutoff", "2023-01-01T00:00:00Z"),
                             ("available_at", "2023-01-01T00:00:00Z")]:
            row = copy.deepcopy(self.targets[0])
            row["provenance"][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                validate_target(row, record, self.source, self.raw)

    def test_target_availability_respects_split_deadline(self):
        index = next(i for i, row in enumerate(self.records) if row["observation_time"].startswith("2024-01"))
        self.targets[index]["provenance"]["available_at"] = "2024-02-02T00:00:00Z"
        report = self.build()
        self.assertEqual(report["counts"]["train"], 11)
        self.assertEqual(report["reason_counts"]["target_unavailable_at_split_cutoff"], 1)

    def test_missing_supervision_is_quarantined_not_filled_from_outcome(self):
        self.targets.pop()
        report = self.build()
        self.assertEqual(sum(report["counts"].values()), 35)
        self.assertEqual(report["reason_counts"]["supervision_missing"], 1)

    def test_market_and_outcome_supervision_modes_are_forbidden(self):
        for kind in ("market_probability", "realized_outcome"):
            row = copy.deepcopy(self.targets[0])
            row["provenance"]["kind"] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(ValidationError, "unsupported supervision"):
                validate_target(row, parse_record(self.records[0]), self.source, self.raw)

    def test_heldout_collision_blocks_the_entire_bilingual_group(self):
        self.index_data["entries"][0]["question"] = self.records[0]["question"]
        self.index_data["entries_sha256"] = canonical_hash(self.index_data["entries"])
        self.index.write_text(json_text(self.index_data))
        self.build()
        metadata = (self.root / "bundle/metadata.jsonl").read_text()
        self.assertNotIn(self.records[0]["event_group_id"], metadata)

    def test_eval_only_source_never_enters_any_sft_partition(self):
        config = json.loads(self.config.read_text())
        config["sources"][SOURCE]["role"] = "eval_only"
        self.config.write_text(json.dumps(config))
        report = self.build()
        self.assertEqual(sum(report["counts"].values()), 0)
        self.assertFalse(report["ready_for_sft"])

    def test_cross_time_group_is_dropped_before_supervision(self):
        first = next(r for r in self.records if r["observation_time"].startswith("2024-01"))
        group = first["event_group_id"]
        pair = [r for r in self.records if r["event_group_id"] == group]
        for row in pair:
            row["label"].update(resolution_time="2024-03-20T00:00:00Z", available_at="2024-03-21T00:00:00Z")
        pair[0]["observation_time"] = "2024-02-15T00:00:00Z"
        report = self.build()
        self.assertEqual(report["reason_counts"]["group_crosses_time_boundary"], 2)

    def test_completion_mask_and_no_silent_truncation(self):
        pair = model_text(parse_record(self.records[0]), self.targets[0]["target"])
        tokenizer = CharacterTokenizer()
        encoded = encode_sft_pair(tokenizer, pair, 10000)
        n = len(tokenizer.encode(pair["prompt"]))
        self.assertEqual(encoded["labels"][:n], [-100] * n)
        self.assertEqual(encoded["labels"][n:], encoded["input_ids"][n:])
        self.assertEqual(encoded["input_ids"][-1], tokenizer.eos_token_id)
        with self.assertRaisesRegex(ValidationError, "truncation is forbidden"):
            encode_sft_pair(tokenizer, pair, 10)

    def test_real_records_require_semantic_review_and_archive_match(self):
        real = copy.deepcopy(self.records[0])
        real.update(dataset_source="real_fixture", dataset_version="1")
        self.records = [real]
        target = copy.deepcopy(self.targets[0])
        target["provenance"].update(kind="historical_forecast", author="test_fixture_forecaster",
                                    artifact="historical.json", reviewer="test_fixture_reviewer")
        p = target["provenance"]
        archive = {"schema_version": "1", "input_sha256": target["input_sha256"], "target": target["target"],
                   **{k: p[k] for k in ("issued_at", "available_at", "information_cutoff", "question_available_at", "author")},
                   "source_uri": "fixture://not-real-data"}
        content = json_text(archive).encode()
        (self.raw / "historical.json").write_bytes(content)
        target["provenance"]["artifact_sha256"] = sha256_bytes(content)
        self.targets = [target]
        config = json.loads(self.config.read_text())
        config["sources"] = {"real_fixture": {"version": "1", "role": "train_eval", "synthetic": False}}
        self.config.write_text(json.dumps(config))
        self.assertEqual(self.build()["reason_counts"]["semantic_benchmark_review_required"], 1)
        review = self.root / "review.json"
        review.write_text(json_text({"heldout_index_sha256": sha256_bytes(self.index.read_bytes()), "groups": {
            real["event_group_id"]: {"decision": "clear", "reviewer": "fixture", "notes": "fixture only"}}}))
        self.assertEqual(sum(self.build("reviewed", review)["counts"].values()), 1)
        self.targets[0]["target"]["probability"] = 0.123456
        with self.assertRaisesRegex(ValidationError, "archived forecast"):
            self.build("tampered", review)

    def test_existing_output_is_not_overwritten(self):
        self.build()
        with self.assertRaisesRegex(ValidationError, "already exists"):
            self.build()


if __name__ == "__main__":
    unittest.main()
