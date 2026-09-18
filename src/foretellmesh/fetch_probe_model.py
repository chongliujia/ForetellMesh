"""Fetch the public pinned probe model and verify upstream LFS SHA-256 hashes."""

import argparse
from pathlib import Path

from .data import sha256_file
from .evaluation import json_text
from .lora_probe import load_probe_config
from .schema import ValidationError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("checkpoints/hf_cache"))
    parser.add_argument("--manifest", type=Path, default=Path("checkpoints/qwen3_8b_base_manifest.json"))
    args = parser.parse_args()
    config = load_probe_config(args.config)
    if args.manifest.exists():
        raise ValidationError("manifest already exists; use a new path or reuse the verified model")
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi(token=False).model_info(config["model"], revision=config["model_revision"], files_metadata=True)
    if info.sha != config["model_revision"]:
        raise ValidationError("upstream revision does not match config")
    root = Path(snapshot_download(
        config["model"], revision=config["model_revision"], cache_dir=args.cache_dir.resolve(),
        allow_patterns=["*.json", "*.safetensors", "*.txt"], token=False, max_workers=5,
    ))
    files = []
    for item in info.siblings:
        path = root / item.rfilename
        if not path.is_file():
            continue
        digest = sha256_file(path)
        if item.size != path.stat().st_size or (item.lfs and digest != item.lfs.sha256):
            raise ValidationError(f"upstream size/SHA-256 mismatch: {item.rfilename}")
        files.append({"name": item.rfilename, "bytes": path.stat().st_size,
                      "sha256": digest, "lfs_hash_verified": bool(item.lfs)})
        print(f"Verified: {item.rfilename}", flush=True)
    manifest = {"model": config["model"], "revision": config["model_revision"],
                "snapshot": str(root), "files": files}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as stream:
        stream.write(json_text(manifest))
    print(f"Ready: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
