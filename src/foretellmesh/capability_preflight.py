"""Offline token and completion-mask checks, without loading model weights."""
import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path
import sys
import tempfile

from .capability_training import encode_capability_row, load_training_config
from .capability_curriculum import select_training_rows
from .data import sha256_file
from .evaluation import code_provenance, json_text
from .lora_probe import verify_model_manifest
from .research_tool_data import read_research_tool_data
from .schema import ValidationError


def check_tokens(tokenizer, partitions: dict, max_length: int) -> dict:
    """Tokenize train/development only, using the actual training encoder."""
    result = {}
    for partition in ("train", "validation"):
        rows = partitions[partition]
        if not rows:raise ValidationError("empty preflight partition")
        lengths, prompt_lengths, supervised = [], [], []
        digest = hashlib.sha256()
        for row in rows:
            try:
                encoded = encode_capability_row(tokenizer, row, max_length)
            except ValidationError as exc:
                raise ValidationError(f'{partition} {row["sample_id"]}: {exc}') from exc
            labels = encoded["labels"]
            n = sum(label == -100 for label in labels)
            if (n == 0 or n == len(labels) or labels[:n] != [-100] * n
                    or labels[n:] != encoded["input_ids"][n:] or labels[-1] != tokenizer.eos_token_id):
                raise ValidationError("completion-only masking/EOS failed")
            digest.update(json_text(encoded).encode())
            lengths.append(len(labels)); prompt_lengths.append(n); supervised.append(len(labels) - n)
        result[partition] = {"examples": len(rows), "max_length": max(lengths), "max_prompt_tokens": max(prompt_lengths),
            "max_completion_tokens": max(supervised), "total_supervised_tokens": sum(supervised),
            "mean_length": sum(lengths) / len(lengths), "encoded_stream_sha256": digest.hexdigest()}
    return result


def preflight(bundle: Path, model_manifest: Path, training_config: Path, output: Path) -> dict:
    if output.exists():raise ValidationError("preflight output already exists")
    config = load_training_config(training_config)
    manifest, partitions, _ = read_research_tool_data(bundle)
    selected_train, selection = select_training_rows(partitions['train'], config.get('curriculum'))
    model_path, model_hash = verify_model_manifest(model_manifest, config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenized = check_tokens(tokenizer, {'train': selected_train, 'validation': partitions['validation']}, config["max_sequence_length"])
    report = {"schema_version": "1", "kind": "capability_data_preflight", "status": "passed",
        "dataset_version": manifest["dataset_version"], "dataset_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "training_config_sha256": sha256_file(training_config), "model_manifest_sha256": model_hash,
        "model": config["model"], "model_revision": config["model_revision"],
        "max_sequence_length": config["max_sequence_length"], "tokenized": tokenized,
        "test_tokenized": False, "truncation": False, "training_performed": False,
        "python_executable": sys.executable, "transformers_version": version("transformers"), "code": code_provenance(),
        "limitations": ["Data/encoding check only; no GPU memory measurement, optimization or model-quality evaluation."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preflight-", dir=output.parent) as tmp:
        stage = Path(tmp) / "report"; stage.mkdir()
        if selection is not None:
            (stage / 'training_selection.json').write_text(json_text(selection))
            report['training_selection_sha256'] = sha256_file(stage / 'training_selection.json')
        (stage / "report.json").write_text(json_text(report))
        (stage / "training_config.json").write_bytes(training_config.read_bytes())
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "model-manifest", "training-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    print(json_text(preflight(a.bundle, a.model_manifest, a.training_config, a.output)))


if __name__ == "__main__":main()
