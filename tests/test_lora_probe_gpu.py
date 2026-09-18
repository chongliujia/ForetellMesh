"""Opt-in lifecycle test with a tiny random Qwen3, never a model benchmark."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_bytes
from foretellmesh.lora_probe import load_probe_config, run_probe
from tests.test_lora_probe import CONFIG, CharacterTokenizer


@unittest.skipUnless(os.environ.get("FORETELLMESH_GPU_TEST") == "1", "opt-in CUDA lifecycle test")
class ProbeGPULifecycleTests(unittest.TestCase):
    def test_real_forward_backward_optimizer_and_report(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_probe_config(CONFIG)
            config.update(run_name="tiny_random_qwen3_lifecycle_test_only",
                          gradient_accumulation_steps=2, measured_optimizer_steps=1)
            config_path = root / "probe.json"
            config_path.write_text(json.dumps(config))
            manifest = {"model": config["model"], "revision": config["model_revision"],
                        "snapshot": str(root), "files": []}
            for name, contents in {
                "config.json": "{}", "tokenizer_config.json": "{}", "tokenizer.json": "{}",
                "model.safetensors.index.json": '{"weight_map": {"weight": "fixture.safetensors"}}',
                "fixture.safetensors": "test fixture; loader replaced with tiny random Qwen3",
            }.items():
                data = contents.encode()
                (root / name).write_bytes(data)
                manifest["files"].append({"name": name, "bytes": len(data), "sha256": sha256_bytes(data)})
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            def tiny_model(*args, **kwargs):
                architecture = Qwen3Config(
                    vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                    max_position_embeddings=2048, attention_dropout=0.0,
                )
                architecture._attn_implementation = "sdpa"
                return Qwen3ForCausalLM(architecture).to(device="cuda", dtype=torch.bfloat16)

            with patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=tiny_model), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()):
                report = run_probe(config_path, manifest_path, 1024, root / "run")
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["adapter_updated"])
            self.assertFalse(report["checkpoint_saved"])
            self.assertEqual(len(report["optimizer_steps"]), 2)
            self.assertGreater(report["final_memory"]["peak_allocated_bytes"], 0)
            self.assertEqual(json.loads((root / "run/report.json").read_text())["status"], "passed")


if __name__ == "__main__":
    unittest.main()
