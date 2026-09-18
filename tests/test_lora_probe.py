import copy
import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh.data import sha256_bytes
from foretellmesh.lora_probe import build_probe_tokens, load_probe_config, verify_model_manifest
from foretellmesh.schema import ValidationError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen3_8b_bf16_lora_probe_v1.json"


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class LoRAProbeTests(unittest.TestCase):
    def test_full_context_and_completion_only_loss(self):
        batch = build_probe_tokens(CharacterTokenizer(), 1024)
        self.assertEqual(len(batch["input_ids"]), 1024)
        self.assertEqual(batch["attention_mask"], [1] * 1024)
        self.assertEqual(batch["input_ids"][-1], 0)
        first = next(i for i, value in enumerate(batch["labels"]) if value != -100)
        self.assertGreater(first, 0)
        self.assertEqual(batch["labels"][:first], [-100] * first)
        self.assertEqual(batch["labels"][first:], batch["input_ids"][first:])
        decoded = "".join(chr(i) for i in batch["labels"][first:-1])
        self.assertEqual(json.loads(decoded)["probability"], 0.5)

    def test_length_change_never_truncates_answer(self):
        shorter = build_probe_tokens(CharacterTokenizer(), 1024)
        longer = build_probe_tokens(CharacterTokenizer(), 2048)
        self.assertEqual([t for t in shorter["labels"] if t != -100],
                         [t for t in longer["labels"] if t != -100])
        self.assertEqual(shorter, build_probe_tokens(CharacterTokenizer(), 1024))
        with self.assertRaises(ValidationError):
            build_probe_tokens(CharacterTokenizer(), 20)

    def test_probe_configuration_cannot_silently_change_method(self):
        config = load_probe_config(CONFIG)
        for key, value in (("model_revision", "main"), ("quantization", "nf4"),
                           ("micro_batch_size", True), ("use_cache", True),
                           ("base_dtype", "float32"), ("measured_optimizer_steps", 0),
                           ("sequence_lengths", [1024, 1024]), ("lora_r", False),
                           ("lora_dropout", 1), ("target_modules", ["q_proj"]),
                           ("learning_rate", 0), ("unknown_option", 1)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                changed = copy.deepcopy(config)
                changed[key] = value
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValidationError):
                    load_probe_config(path)

    def test_manifest_checks_revision_shards_and_file_content(self):
        config = load_probe_config(CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contents = {"config.json": "{}", "tokenizer_config.json": "{}",
                        "tokenizer.json": "{}", "weights.safetensors": "fake fixture",
                        "model.safetensors.index.json": '{"weight_map": {"weight": "weights.safetensors"}}'}
            manifest = {"model": config["model"], "revision": config["model_revision"],
                        "snapshot": str(root), "files": []}
            for name, content in contents.items():
                data = content.encode()
                (root / name).write_bytes(data)
                manifest["files"].append({"name": name, "bytes": len(data), "sha256": sha256_bytes(data)})
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(verify_model_manifest(path, config)[0], root)
            changed = {**config, "model_revision": "0" * 40}
            with self.assertRaisesRegex(ValidationError, "pinned config"):
                verify_model_manifest(path, changed)
            manifest["files"] = [f for f in manifest["files"] if f["name"] != "weights.safetensors"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValidationError, "unverified.*shards"):
                verify_model_manifest(path, config)
            (root / "config.json").write_text("[]")
            with self.assertRaisesRegex(ValidationError, "checksum"):
                verify_model_manifest(path, config)


if __name__ == "__main__":
    unittest.main()
