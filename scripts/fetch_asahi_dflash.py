"""Fetch the pinned small trained draft into this repository, checking SHA256."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = "orestis-z/dflash-qwen3-0.6b-microcycle-dflash"
REVISION = "4dd1e04078f993593338ef1f9403179e41e4580e"
HASHES = {
    "config.json": "7795568c67ea825f967e252f0a7c8e7b2ef06b843af0d92d936185c4f2c43c7a",
    "model.safetensors": "74a0b0f98b2a5b6a82c3c0bf8a4c7fbbe2a85f1920a7b6836ba6a9b716541a64",
    "val_metrics.json": "fd03e24e55d0533a08a5ebdde2fa22fd8c56e3c11b19920aef801504185d661b",
    "train_command.txt": "fbb201344d37aa1adaf92074181bf414d65fcd074562c97bddeef6a05f21cd43",
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    dest = REPO / ".asahi/models/microcycle-dflash"
    dest.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        path = dest / name
        if path.exists():
            if digest(path) != expected:
                raise RuntimeError(f"Existing {path} has an unexpected SHA256")
            print(f"verified {name}", flush=True)
            continue
        temp = path.with_suffix(path.suffix + ".part")
        print(f"downloading {name}", flush=True)
        urllib.request.urlretrieve(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", temp)
        if digest(temp) != expected:
            raise RuntimeError(f"Downloaded {name} has an unexpected SHA256")
        temp.rename(path)
    (dest / "SOURCE.json").write_text(json.dumps({"model": MODEL, "revision": REVISION}, indent=2) + "\n")


if __name__ == "__main__":
    main()
