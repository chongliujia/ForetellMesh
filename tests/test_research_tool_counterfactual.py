from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from foretellmesh.data import sha256_file
from foretellmesh.peft_runtime import render_agent_prompt
from foretellmesh.research_tool_counterfactual import (
    CONDITIONS, build_counterfactual_data, exclusion_signatures, information_state,
    load_counterfactual_config, render_counterfactual, visible_signatures,
)
from foretellmesh.research_tool_data import build_research_tool_data, read_research_tool_data
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import generate_synthetic_sft
from foretellmesh.uncertainty_diagnostic import build_uncertainty_cohort

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {
    "bayes": {"prior": .31, "sensitivity": .78, "false_positive_rate": .16},
    "mixture": {"high_rate": .91, "low_rate": .23, "weight": .68},
    "complement": {"hit_probability": .42, "lower": .28, "upper": .57},
    "sampling": {"successes": 19, "trials": 51, "population_probability": .39},
}


def example(family="bayes", condition="complete", scope="forecast", fmt="field_names", role="research"):
    return {"kind": "information_state", "family": family, "parameters": PARAMETERS[family], "condition": condition,
            "scope": scope, "format": fmt, "language": "en", "role": role, "observation_time": "2024-01-15T00:00:00Z"}


class InformationStateTests(unittest.TestCase):
    def test_exact_identifiability_for_all_disclosures(self):
        # Independent expected unknowns, with no dependence on generator output.
        expected = {
            "bayes": [set(), {"prior", "posterior_probability"}, {"sensitivity", "posterior_probability"},
                      {"prior", "sensitivity", "posterior_probability"}],
            "mixture": [set(), {"weight", "success_probability"}, {"high_rate", "success_probability"},
                        {"weight", "high_rate", "success_probability"}],
            "complement": [set(), set(), set(), {"hit_probability", "miss_probability"}],
            "sampling": [{"population_probability"}, set(), {"successes", "empirical_frequency", "population_probability"},
                         {"trials", "empirical_frequency", "population_probability"}],
        }
        for family, conditions in CONDITIONS.items():
            for condition, unknown in zip(conditions, expected[family]):
                with self.subTest(family=family, condition=condition):
                    row = render_counterfactual(example(family, condition), "train")
                    self.assertEqual(set(row["target"]["unknowns"]), unknown | {"future_outcome"})

    def test_missing_prior_really_has_multiple_answers_with_identical_visible_inputs(self):
        left = example(condition="omit_prior")
        right = deepcopy(left); right["parameters"]["prior"] = .61
        visible_left = information_state("bayes", left["parameters"], "omit_prior")[0]
        visible_right = information_state("bayes", right["parameters"], "omit_prior")[0]
        self.assertEqual(visible_left, visible_right)
        def posterior(prior):
            return prior * Fraction(78, 100) / (prior * Fraction(78, 100) + (1-prior) * Fraction(16, 100))
        self.assertNotEqual(posterior(Fraction(31, 100)), posterior(Fraction(61, 100)))
        # A hash of full latent parameters alone would miss this collision.
        self.assertTrue(visible_signatures("bayes", left["parameters"]) & visible_signatures("bayes", right["parameters"]))

    def test_scope_and_output_mapping_are_task_dependent(self):
        for family, complete in (("bayes", "complete"), ("mixture", "complete"), ("complement", "miss_only"), ("sampling", "counts_only")):
            calc = render_counterfactual(example(family, complete, "calculation"), "train")
            forecast = render_counterfactual(example(family, complete), "train")
            self.assertEqual(calc["target"]["unknowns"], [])
            self.assertIn("future_outcome", forecast["target"]["unknowns"])
        for role in ("research", "risk"):
            fields = render_counterfactual(example(condition="omit_prior", role=role), "train")
            aliases = render_counterfactual(example(condition="omit_prior", fmt="aliases", role=role), "train")
            mapping = aliases["oracle"]["output_names"]
            self.assertEqual(set(aliases["target"]["unknowns"]), {mapping[k] for k in fields["target"]["unknowns"]})
            self.assertNotEqual(fields["target"]["unknowns"], aliases["target"]["unknowns"])
            self.assertEqual(fields["request"]["input"]["evidence"], aliases["request"]["input"]["evidence"])

    def test_paired_disclosures_share_question_and_group_without_exposing_missing_values(self):
        before = render_counterfactual(example(fmt="aliases"), "train")
        after = render_counterfactual(example(condition="omit_prior", fmt="aliases"), "train")
        self.assertEqual(before["event_group_id"], after["event_group_id"])
        self.assertEqual(before["request"]["input"]["question"], after["request"]["input"]["question"])
        prompt = render_agent_prompt(after["request"])
        self.assertNotIn('"prior":0.31', prompt.replace('\\"', '"'))
        for forbidden in ("source_case", "unknown_fields", "known_fields", "case_sha256", "partition", "oracle"):
            self.assertNotIn(forbidden, prompt)
        self.assertNotIn(after["sample_id"], prompt)
        self.assertNotIn("label", after["request"]["input"])

    def test_interval_and_finite_sampling_do_not_become_exact_probabilities(self):
        interval = render_counterfactual(example("complement", "interval_only", "calculation"), "train")
        self.assertEqual(set(interval["target"]["unknowns"]), {"hit_probability", "miss_probability"})
        sample = render_counterfactual(example("sampling", "counts_only"), "train")
        self.assertIn("population_probability", sample["target"]["unknowns"])
        self.assertNotIn("empirical_frequency", sample["target"]["unknowns"])

    def test_degenerate_or_unsupported_parameters_fail_closed(self):
        invalid = [example(), example("mixture", "complete"), example("complement", "interval_only"), example("sampling", "counts_only")]
        invalid[0] = deepcopy(invalid[0]); invalid[0]["parameters"]["prior"] = 0
        invalid[1] = deepcopy(invalid[1]); invalid[1]["parameters"]["high_rate"] = invalid[1]["parameters"]["low_rate"]
        invalid[2] = deepcopy(invalid[2]); invalid[2]["parameters"]["upper"] = .01
        invalid[3] = deepcopy(invalid[3]); invalid[3]["parameters"]["trials"] = True
        for case in invalid:
            with self.assertRaises(ValidationError):render_counterfactual(case, "train")


class CounterfactualBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        generator = json.loads((ROOT / "configs/synthetic_sft_generator_v1.json").read_text())
        generator["groups_per_family"] = {p: 2 for p in ("train", "validation", "test")}
        path = cls.root / "generator.json"; path.write_text(json.dumps(generator))
        raw = cls.root / "raw"; generate_synthetic_sft(path, raw)
        config = json.loads((ROOT / "configs/research_tool_dataset_v1.json").read_text())
        config["evidence_groups"] = {p: 3 for p in ("train", "validation", "test")}
        path = cls.root / "legacy_config.json"; path.write_text(json.dumps(config)); cls.legacy = cls.root / "legacy"
        build_research_tool_data(raw, ROOT / "configs/synthetic_sft_splits_v1.json", path, cls.legacy)
        cls.diagnostic = cls.root / "diagnostic"
        build_uncertainty_cohort(ROOT / "configs/uncertainty_diagnostic_v1.json", cls.diagnostic)
        config = json.loads((ROOT / "configs/research_tool_dataset_v3.json").read_text())
        config["groups_per_family"] = {p: 1 for p in ("train", "validation", "test")}
        cls.config_path = cls.root / "config.json"; cls.config_path.write_text(json.dumps(config))
        cls.bundle = cls.root / "bundle"
        build_counterfactual_data(cls.legacy, cls.diagnostic, cls.config_path, cls.bundle)
        cls.manifest, cls.parts, cls.config = read_research_tool_data(cls.bundle)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_original_tasks_and_frozen_cohorts_are_preserved(self):
        _, legacy, _ = read_research_tool_data(self.legacy)
        self.assertEqual((self.bundle / "evaluation_ids.json").read_bytes(), (self.legacy / "evaluation_ids.json").read_bytes())
        for p, rows in legacy.items():
            lookup = {r["sample_id"]: r for r in self.parts[p]}
            for old in rows:
                self.assertEqual(old["request"]["input"], lookup[old["sample_id"]]["request"]["input"])
                self.assertEqual(old["target"], lookup[old["sample_id"]]["target"])
        self.assertEqual(json.loads((self.bundle / "excluded_diagnostic_config.json").read_text()),
                         json.loads((self.diagnostic / "config.json").read_text()))

    def test_all_64_variants_stay_together_and_visible_scenarios_are_disjoint(self):
        _, legacy, _ = read_research_tool_data(self.legacy)
        probe = json.loads((self.diagnostic / "config.json").read_text())
        signatures = exclusion_signatures(probe, legacy)
        owners = {}
        times = []
        for p, rows in self.parts.items():
            groups = {}
            for row in rows:
                self.assertEqual(owners.setdefault(row["event_group_id"], p), p)
                self.assertEqual(row["request"]["input"]["observation_time"], self.config["observation_times"][p])
                if row["case"]["kind"] == "information_state":groups.setdefault(row["event_group_id"], []).append(row)
            self.assertEqual(len(groups), 4)
            for members in groups.values():
                self.assertEqual(len(members), 64)
                c = members[0]["case"]
                visible = visible_signatures(c["family"], c["parameters"])
                self.assertFalse(visible & signatures); signatures.update(visible)
            times.append(self.config["observation_times"][p])
        self.assertEqual(times, sorted(times))

    def test_rehashed_corruption_of_targets_time_scope_and_group_members_is_rejected(self):
        for name in ("target", "future", "scope", "variant", "label"):
            with self.subTest(name=name):
                copy = self.root / ("bad_" + name); shutil.copytree(self.bundle, copy)
                rows = [json.loads(s) for s in (copy / "train.jsonl").read_text().splitlines()]
                row = next(r for r in rows if r["case"]["kind"] == "information_state")
                if name == "target":row["target"]["unknowns"] = ["invented"]
                elif name == "future":row["request"]["input"]["evidence"][0]["available_at"] = "2025-01-01T00:00:00Z"
                elif name == "scope":row["case"]["scope"] = "forecast" if row["case"]["scope"] == "calculation" else "calculation"
                elif name == "variant":rows.remove(row)
                else:row["request"]["input"]["label"] = {"outcome": 1}
                (copy / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                manifest = json.loads((copy / "manifest.json").read_text())
                manifest["artifact_hashes"]["train.jsonl"] = sha256_file(copy / "train.jsonl")
                (copy / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValidationError, "replay mismatch"):read_research_tool_data(copy)

    def test_config_rejects_random_splits_and_empty_groups(self):
        for key, value in (("observation_times", {p: "2024-01-01T00:00:00Z" for p in self.parts}),
                           ("groups_per_family", {p: 0 for p in self.parts}), ("synthetic_only", False)):
            config = deepcopy(self.config); config[key] = value
            path = self.root / "invalid_config.json"; path.write_text(json.dumps(config))
            with self.assertRaises(ValidationError):load_counterfactual_config(path)

    def test_rebuild_is_identical_and_cannot_overwrite(self):
        rebuilt = self.root / "rebuilt"
        build_counterfactual_data(self.legacy, self.diagnostic, self.config_path, rebuilt)
        for name in self.manifest["artifact_hashes"]:
            self.assertEqual((self.bundle / name).read_bytes(), (rebuilt / name).read_bytes())
        with self.assertRaises(ValidationError):
            build_counterfactual_data(self.legacy, self.diagnostic, self.config_path, self.bundle)


if __name__ == "__main__":unittest.main()
