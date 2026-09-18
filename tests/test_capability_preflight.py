from copy import deepcopy
import unittest

from foretellmesh.capability_preflight import check_tokens
from foretellmesh.schema import ValidationError


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if "FROZEN_TEST_SENTINEL" in text:
            raise AssertionError("test partition must never be tokenized")
        return [ord(char) for char in text]


class CapabilityPreflightTests(unittest.TestCase):
    def setUp(self):
        row = {"sample_id": "example", "request": {"instruction": "return JSON", "input": {}, "upstream": {}}, "target": {"unknowns": []}}
        self.parts = {"train": [deepcopy(row)], "validation": [deepcopy(row)], "test": [deepcopy(row)]}
        self.parts["test"][0]["request"]["instruction"] = "FROZEN_TEST_SENTINEL"

    def test_training_encoder_masks_prompt_and_never_encodes_test(self):
        report = check_tokens(CharacterTokenizer(), self.parts, 1024)
        self.assertEqual(set(report), {"train", "validation"})
        self.assertEqual(report["train"], report["validation"])
        self.assertEqual(report["train"]["max_completion_tokens"], len('{"unknowns":[]}') + 1)
        self.assertGreater(report["train"]["max_prompt_tokens"], 0)

    def test_overlong_examples_fail_without_truncation_and_identify_row(self):
        with self.assertRaisesRegex(ValidationError, "train example:.*truncation is forbidden"):
            check_tokens(CharacterTokenizer(), self.parts, 10)

    def test_empty_validation_is_not_reported_as_passed(self):
        self.parts["validation"] = []
        with self.assertRaisesRegex(ValidationError, "empty preflight partition"):
            check_tokens(CharacterTokenizer(), self.parts, 1024)


if __name__ == "__main__":unittest.main()
