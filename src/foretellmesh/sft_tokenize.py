"""Offline tokenization for the pinned Qwen Base tokenizer; no GPU or training."""

from pathlib import Path
import re
import tempfile

from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .schema import ValidationError, fields
from .sft_data import PROMPT_VERSION, encode_sft_pair, jsonl


def tokenize_sft(bundle: Path, model_manifest: Path, config_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValidationError("tokenized output already exists")
    config_bytes = config_path.read_bytes()
    config = fields(strict_json(config_bytes.decode()), {"model", "model_revision", "max_sequence_length", "serialization",
                                                       "add_special_tokens", "append_eos", "loss", "truncation"}, "tokenization config")
    if (config["model"] != "Qwen/Qwen3-8B-Base" or config["serialization"] != PROMPT_VERSION
            or not re.fullmatch(r"[0-9a-f]{40}", str(config["model_revision"]))
            or config["add_special_tokens"] is not False or config["append_eos"] is not True
            or config["truncation"] is not False or config["loss"] != "completion_only"
            or type(config["max_sequence_length"]) is not int or not 1 <= config["max_sequence_length"] <= 32768):
        raise ValidationError("unsupported tokenization policy")
    model_bytes = model_manifest.read_bytes()
    model = strict_json(model_bytes.decode())
    if model["model"] != config["model"] or model["revision"] != config["model_revision"]:
        raise ValidationError("tokenizer model/revision mismatch")
    needed = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"}
    files = {row["name"]: row for row in model["files"]}
    if not needed <= files.keys():
        raise ValidationError("missing tokenizer artifacts")
    model_path = Path(model["snapshot"])
    for name in needed:
        path = model_path / name
        if path.stat().st_size != files[name]["bytes"] or sha256_file(path) != files[name]["sha256"]:
            raise ValidationError("tokenizer artifact hash mismatch")
    bundle_bytes = (bundle / "manifest.json").read_bytes()
    manifest = strict_json(bundle_bytes.decode())
    if manifest["kind"] != "forecast_sft_dataset" or manifest["prompt_version"] != PROMPT_VERSION:
        raise ValidationError("unsupported SFT bundle")
    if not manifest["ready_for_sft"]:
        raise ValidationError("SFT bundle needs nonempty train and validation partitions")
    contents = {}
    for name, expected in manifest["artifact_hashes"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValidationError("invalid bundle artifact path")
        content = path.read_bytes()
        if sha256_bytes(content) != expected:
            raise ValidationError("SFT bundle artifact hash mismatch")
        contents[name] = content
    from transformers import AutoTokenizer
    from importlib.metadata import version
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    outputs, statistics = {}, {}
    for split in ("train", "validation", "test"):
        rows = []
        for line_number, line in enumerate(contents[f"{split}.jsonl"].decode().splitlines(), 1):
            try:
                rows.append(encode_sft_pair(tokenizer, strict_json(line), config["max_sequence_length"]))
            except ValidationError as exc:
                raise ValidationError(f"{split} row {line_number}: {exc}") from exc
        if len(rows) != manifest["counts"][split]:
            raise ValidationError("SFT split counts disagree with manifest")
        lengths = [len(row["input_ids"]) for row in rows]
        statistics[split] = {"count": len(rows), "min_tokens": min(lengths) if lengths else None,
                             "max_tokens": max(lengths) if lengths else None,
                             "mean_tokens": sum(lengths) / len(lengths) if lengths else None,
                             "supervised_tokens": sum(sum(t != -100 for t in r["labels"]) for r in rows)}
        outputs[f"{split}.jsonl"] = jsonl(rows).encode()
    # Metadata stays outside tensor rows. Preserve row-index mappings and audit source.
    outputs["metadata.jsonl"] = contents["metadata.jsonl"]
    report = {"schema_version": "1", "kind": "tokenized_forecast_sft", "statistics": statistics,
              "synthetic_only": manifest["synthetic_only"], "config": config,
              "config_sha256": sha256_bytes(config_bytes), "source_manifest_sha256": sha256_bytes(bundle_bytes),
              "model_manifest_sha256": sha256_bytes(model_bytes), "eos_token_id": tokenizer.eos_token_id,
              "transformers_version": version("transformers"), "tokenizers_version": version("tokenizers"),
              "artifact_hashes": {name: sha256_bytes(value) for name, value in outputs.items()},
              "code": code_provenance(), "training_performed": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sft-tokens-", dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        for name, value in outputs.items():
            (stage / name).write_bytes(value)
        (stage / "config.json").write_bytes(config_bytes)
        (stage / "source_manifest.json").write_bytes(bundle_bytes)
        (stage / "manifest.json").write_text(json_text(report), encoding="utf-8")
        stage.rename(output)
    return report
