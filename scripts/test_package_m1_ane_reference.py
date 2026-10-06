"""Verify portable kernel packaging preserves bytes and rejects damaged data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import package_m1_ane_reference as package


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.capture = self.root / "capture"; self.capture.mkdir()
        self.bundle = self.root / "bundle"
        payload = b"reference coefficients" * 20
        hwx = self.capture / "case/capture/offline-hwx/model.hwx"
        hwx.parent.mkdir(parents=True)
        hwx.write_bytes(package.struct.pack("<8I", 0xBEEFFACE, 128, 4, 0, 0, 0, 0, 0) + payload)
        decoded = hwx.with_name("model.hwx-decoded"); decoded.mkdir()
        container = decoded / "container.json"
        container.write_text(json.dumps({"segments": [{"fileoff": 32, "filesize": len(payload),
                             "payload_sha256": hashlib.sha256(payload).hexdigest()}]}))
        a = self.capture / "input.bin"; b = self.capture / "golden.bin"
        a.write_bytes(payload); b.write_bytes(payload)
        inputs = self.capture / "manifest.json"; inputs.write_text('{"tools": {}, "cases": []}')
        (self.capture / "capture-manifest.json").write_text(json.dumps({
            "captured_files": {}, "input_manifest_sha256": package.sha256(inputs)}))
        self.files = [hwx, container, a, b]
        summary = {"hardware": "test", "macos": "test", "golden_provenance": "test",
                   "selected_cases": [], "missing": ["Linux replay"]}
        with patch.object(package, "selected_files", return_value=(summary, self.files)):
            self.manifest = package.pack(self.capture, self.bundle)

    def test_roundtrip_recreates_exact_files_and_embedded_coefficient_segments(self):
        output = self.root / "restored"
        package.verify_or_extract(self.bundle, output)
        for original in self.files:
            self.assertEqual((output / original.relative_to(self.capture)).read_bytes(), original.read_bytes())
        segment = output / "case/capture/offline-hwx/model.hwx-decoded/payloads/segment-0.bin"
        self.assertEqual(segment.read_bytes(), (self.capture / "input.bin").read_bytes())
        self.assertLess(len(self.manifest["objects"]), sum(len(r["chunks"]) for r in self.manifest["files"].values()))

    def test_corrupted_pack_is_rejected_before_extraction(self):
        pack = next(self.bundle.glob("*.pack"))
        raw = bytearray(pack.read_bytes()); raw[-1] ^= 1; pack.write_bytes(raw)
        output = self.root / "restored"
        with self.assertRaisesRegex(ValueError, "Pack hash mismatch"):
            package.verify_or_extract(self.bundle, output)
        self.assertFalse(output.exists())

    def test_output_is_never_overwritten(self):
        output = self.root / "existing"; output.mkdir()
        (output / "keep").write_text("keep")
        with self.assertRaisesRegex(ValueError, "fresh extraction"):
            package.verify_or_extract(self.bundle, output)
        self.assertEqual((output / "keep").read_text(), "keep")

    def test_path_traversal_is_rejected(self):
        manifest = self.manifest
        manifest["files"]["../escape"] = manifest["files"].pop("input.bin")
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "relative artifact path"):
            package.verify_or_extract(self.bundle, self.root / "restored")
        self.assertFalse((self.root / "escape").exists())


if __name__ == "__main__": unittest.main()
