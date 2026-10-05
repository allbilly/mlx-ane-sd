"""Verify transferred capture inputs; record all macOS artifact hashes."""
import argparse
import hashlib
import json
import platform
import plistlib
from pathlib import Path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kit", type=Path)
    ap.add_argument("--record-capture", action="store_true")
    args = ap.parse_args()
    manifest = json.loads((args.kit / "manifest.json").read_text())
    expected = dict(manifest["tools"])
    for case in manifest["cases"]:
        expected.update({case["directory"] + "/" + p: h for p, h in case["files"].items()})
    for name, wanted in expected.items():
        if digest(args.kit / name) != wanted:
            raise ValueError(f"Transferred file hash differs: {name}")
    print(f"Verified {len(expected)} input/tool hashes")
    if args.record_capture:
        files = {str(p.relative_to(args.kit)): digest(p) for p in sorted(args.kit.rglob("*")) if p.is_file()
                 and str(p.relative_to(args.kit)) not in expected and p.name != "capture-manifest.json"}
        frameworks = {}
        for name in ("ANECompiler", "AppleNeuralEngine"):
            base = Path("/System/Library/PrivateFrameworks") / (name + ".framework")
            for candidate in (base / "Resources/Info.plist", base / "Versions/A/Resources/Info.plist"):
                if candidate.is_file():
                    info = plistlib.loads(candidate.read_bytes())
                    frameworks[name] = {k: info.get(k) for k in ("CFBundleVersion", "CFBundleShortVersionString")}
                    break
        (args.kit / "capture-manifest.json").write_text(json.dumps({"schema": 1,
            "platform": platform.platform(), "private_framework_versions": frameworks,
            "input_manifest_sha256": digest(args.kit / "manifest.json"), "captured_files": files}, indent=2) + "\n")


if __name__ == "__main__":
    main()
