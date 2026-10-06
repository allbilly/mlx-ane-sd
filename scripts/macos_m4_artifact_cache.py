"""Content checks for reusable macOS conversion artifacts; no device imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class CacheMismatchError(RuntimeError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def artifact_sha256(path):
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not path.is_dir() or not files:
        raise CacheMismatchError(f"Missing or empty artifact: {path}")
    return fingerprint({str(p.relative_to(path)): file_sha256(p) for p in files})


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_cached(receipt, build_inputs, artifacts):
    """Return only a verified match. Preserve unknown/stale files and fail closed."""
    receipt = Path(receipt)
    if not receipt.exists() and not any(Path(p).exists() for p in artifacts.values()):
        return None

    def reject(reason):
        raise CacheMismatchError(f"{receipt.parent}: {reason}. Use a fresh --out directory; "
                                 "the existing artifacts have been preserved.")

    if not receipt.is_file():
        reject("cached artifacts have no fingerprint receipt")
    try:
        info = json.loads(receipt.read_text())
    except (ValueError, OSError):
        reject("unreadable cache receipt")
    if (not isinstance(info, dict) or info.get("cache_schema") != 1
            or info.get("build_inputs") != build_inputs
            or info.get("build_key") != fingerprint(build_inputs)):
        reject("cache inputs differ or the legacy cache is unverified")
    for name, path in artifacts.items():
        if info.get(name) != str(Path(path).resolve()):
            reject(f"{name} path differs")
        try:
            actual = artifact_sha256(path)
        except CacheMismatchError:
            reject(f"{name} is missing or empty")
        if actual != info.get("artifacts_sha256", {}).get(name):
            reject(f"{name} contents differ")
    return info


def save_receipt(receipt, info, build_inputs, artifacts):
    info = {**info, "cache_schema": 1, "build_inputs": build_inputs,
            "build_key": fingerprint(build_inputs),
            "artifacts_sha256": {name: artifact_sha256(path)
                                 for name, path in artifacts.items()}}
    info.update({name: str(Path(path).resolve()) for name, path in artifacts.items()})
    write_json(receipt, info)
    return info
