"""Pinned public source files; downloading never executes upstream code."""

from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

from .data import sha256_bytes, strict_json
from .evaluation import json_text
from .schema import ValidationError, fields, nonempty


@dataclass(frozen=True)
class Source:
    name: str
    revision: str
    role: str
    metadata: dict


def load_source(registry: Path, name: str) -> Source:
    raw = strict_json(registry.read_text(encoding="utf-8"))
    fields(raw, {"schema_version", "sources"}, "source registry")
    if raw["schema_version"] != "1" or not isinstance(raw["sources"], dict):
        raise ValidationError("unsupported source registry")
    if name not in raw["sources"]:
        raise ValidationError(f"unregistered source: {name}")
    source = raw["sources"][name]
    if not isinstance(source, dict):
        raise ValidationError("source metadata must be an object")
    for key in ("revision", "role", "homepage", "license", "artifacts"):
        if key not in source:
            raise ValidationError(f"source missing {key}")
    revision = nonempty(source["revision"], "source revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValidationError("source revision must be a full immutable Git/Hugging Face commit")
    if source["role"] not in ("train_eval", "eval_only"):
        raise ValidationError("invalid source role")
    if name == "forecastbench" and source["role"] != "eval_only":
        raise ValidationError("ForecastBench is reserved for evaluation")
    if not isinstance(source["artifacts"], dict) or not source["artifacts"]:
        raise ValidationError("source artifacts must be a nonempty object")
    for filename, artifact in source["artifacts"].items():
        if Path(filename).name != filename or filename in ("", ".", "..", "download_manifest.json"):
            raise ValidationError("artifact names must be plain filenames")
        fields(artifact, {"url", "sha256", "max_bytes", "purpose"}, "artifact")
        url = urlparse(nonempty(artifact["url"], "artifact URL"))
        if url.scheme != "https" or url.hostname not in ("huggingface.co", "raw.githubusercontent.com"):
            raise ValidationError("artifact URL must use an approved public HTTPS host")
        if revision not in url.path.split("/"):
            raise ValidationError("artifact URL must contain the pinned revision")
        if not re.fullmatch(r"[0-9a-f]{64}", str(artifact["sha256"])):
            raise ValidationError("artifact requires an exact SHA-256")
        if type(artifact["max_bytes"]) is not int or not 1 <= artifact["max_bytes"] <= 50_000_000:
            raise ValidationError("artifact max_bytes must be in [1, 50000000]")
    return Source(name, revision, source["role"], source)


def verify_artifact(path: Path, source: Source, purpose: str) -> dict:
    matches = [value for value in source.metadata["artifacts"].values() if value["purpose"] == purpose]
    if len(matches) != 1:
        raise ValidationError(f"source needs exactly one {purpose} artifact")
    artifact = matches[0]
    if path.stat().st_size > artifact["max_bytes"]:
        raise ValidationError(f"{purpose}: file exceeds registered size limit")
    digest = sha256_bytes(path.read_bytes())
    if digest != artifact["sha256"]:
        raise ValidationError(f"{purpose}: SHA-256 mismatch; register the exact source version first")
    return {"sha256": digest, "url": artifact["url"]}


def download_source(registry: Path, name: str, output: Path) -> dict:
    source = load_source(registry, name)
    if output.exists():
        raise ValidationError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    with tempfile.TemporaryDirectory(prefix=".source-", dir=output.parent) as staging:
        stage = Path(staging) / "data"
        stage.mkdir()
        for filename, artifact in source.metadata["artifacts"].items():
            with urlopen(artifact["url"], timeout=30) as response:
                content = response.read(artifact["max_bytes"] + 1)
            if len(content) > artifact["max_bytes"]:
                raise ValidationError(f"download exceeds size limit: {filename}")
            digest = sha256_bytes(content)
            if digest != artifact["sha256"]:
                raise ValidationError(f"download SHA-256 mismatch: {filename}")
            (stage / filename).write_bytes(content)
            artifacts[filename] = {**artifact, "bytes": len(content)}
        manifest = {"source": name, "revision": source.revision, "role": source.role,
                    "artifacts": artifacts, "license": source.metadata["license"]}
        (stage / "download_manifest.json").write_text(json_text(manifest), encoding="utf-8")
        stage.rename(output)
    return manifest
