"""Regression checks for cache provenance and benchmark validity; no GPU/ANE work."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from macos_m4_artifact_cache import CacheMismatchError, load_cached, save_receipt
from macos_m4_benchmark_checks import finish_benchmark, require_same_target_identity, summarize
from report_macos_m4_recipe import verified_source


class ArtifactCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.receipt = self.folder / "receipt.json"
        package = self.folder / "model.mlpackage"
        compiled = self.folder / "model.mlmodelc"
        for p in (package, compiled):
            p.mkdir(); (p / "weights.bin").write_bytes(b"model weights")
        self.artifacts = {"package": package, "compiled": compiled}
        self.inputs = {"weights_sha256": "target-a", "source_sha256": "source-a",
                       "shape": [1, 8, 1024], "capacity": 256, "bits": 6}
        self.info = save_receipt(self.receipt, {"placement": {"math_all_ane": True}},
                                 self.inputs, self.artifacts)

    def test_matching_cache_is_reused_without_mutation(self):
        before = self.receipt.read_bytes()
        self.assertEqual(load_cached(self.receipt, self.inputs, self.artifacts), self.info)
        self.assertEqual(self.receipt.read_bytes(), before)

    def test_changed_inputs_are_rejected_without_overwriting(self):
        before = self.receipt.read_bytes()
        for name, value in (("weights_sha256", "target-b"), ("source_sha256", "source-b"),
                            ("shape", [1, 1, 1024]), ("capacity", 512), ("bits", 8)):
            with self.subTest(name=name):
                with self.assertRaisesRegex(CacheMismatchError, "fresh --out"):
                    load_cached(self.receipt, {**self.inputs, name: value}, self.artifacts)
                self.assertEqual(self.receipt.read_bytes(), before)

    def test_modified_artifact_is_rejected(self):
        (self.artifacts["compiled"] / "weights.bin").write_bytes(b"different weights")
        with self.assertRaisesRegex(CacheMismatchError, "contents differ"):
            load_cached(self.receipt, self.inputs, self.artifacts)

    def test_legacy_and_missing_receipts_are_rejected(self):
        self.receipt.write_text(json.dumps({"placement": {"math_all_ane": True}}))
        with self.assertRaisesRegex(CacheMismatchError, "legacy"):
            load_cached(self.receipt, self.inputs, self.artifacts)
        self.receipt.unlink()
        with self.assertRaisesRegex(CacheMismatchError, "no fingerprint"):
            load_cached(self.receipt, self.inputs, self.artifacts)

    def test_empty_compiled_directory_is_rejected(self):
        (self.artifacts["compiled"] / "weights.bin").unlink()
        with self.assertRaisesRegex(CacheMismatchError, "missing or empty"):
            load_cached(self.receipt, self.inputs, self.artifacts)

    def test_embedding_contents_are_also_checked(self):
        embedding = self.folder / "embedding.npy"
        embedding.write_bytes(b"embedding-a")
        receipt = self.folder / "embedding.receipt.json"
        save_receipt(receipt, {}, self.inputs, {"embedding": embedding})
        embedding.write_bytes(b"embedding-b")
        with self.assertRaisesRegex(CacheMismatchError, "contents differ"):
            load_cached(receipt, self.inputs, {"embedding": embedding})

    def test_fresh_output_can_build(self):
        self.assertIsNone(load_cached(self.folder / "fresh.json", self.inputs,
                                     {"compiled": self.folder / "fresh.mlmodelc"}))


class BenchmarkGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent.parent
        cls.receipt = json.loads((root / "notes/m1_m4_recipe.json").read_text())

    def test_existing_receipt_and_summary_pass(self):
        r = copy.deepcopy(self.receipt)
        self.assertEqual(summarize(r["rows"]), r["summary"])
        finish_benchmark(r)
        self.assertEqual(r["same_target_check"]["matched_tokens"], 800)
        self.assertEqual(r["same_target_check"]["pairs"], 8)

    def test_token_and_length_mismatches_never_complete(self):
        for change in ("token", "length"):
            with self.subTest(change=change):
                r = copy.deepcopy(self.receipt); r["status"] = "benchmarking"
                sd = next(x for x in r["rows"] if x["impl"] == "ane_lut6_sd")
                if change == "token": sd["tokens"][5] += 1
                else: sd["tokens"].pop()
                with self.assertRaisesRegex(ValueError, "SD differs.*capital.*repeat 1"):
                    finish_benchmark(r)
                self.assertEqual(r["status"], "benchmarking")
                self.assertNotIn("same_target_check", r)

    def test_missing_and_duplicate_pairs_fail(self):
        for rows in (self.receipt["rows"][:-1], self.receipt["rows"] + [self.receipt["rows"][0]]):
            with self.assertRaises(ValueError):
                require_same_target_identity(rows, 2)

    def test_bf16_drift_is_not_a_same_target_failure(self):
        rows = copy.deepcopy(self.receipt["rows"])
        for row in rows:
            if row["impl"] == "mlx_bf16": row["tokens"] = [0]
        self.assertTrue(require_same_target_identity(rows, 2)["passed"])

    def test_source_snapshot_requires_exact_recorded_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sha = "receipt-sha"
            current = root / "scripts/bench.py"; current.parent.mkdir()
            current.write_bytes(b"new code")
            archived = root / "notes/m1_m4_recipe_sources" / sha / "scripts/bench.py"
            archived.parent.mkdir(parents=True); archived.write_bytes(b"measured code")
            expected = hashlib.sha256(b"measured code").hexdigest()
            self.assertEqual(verified_source(root, sha, "bench.py", expected), str(archived.relative_to(root)))
            archived.write_bytes(b"tampered code")
            with self.assertRaises(ValueError): verified_source(root, sha, "bench.py", expected)


class DriverIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import bench_macos_m4_recipe
        import convert_macos_m4_recipe
        cls.driver = bench_macos_m4_recipe
        cls.converter = convert_macos_m4_recipe

    def test_rejected_legacy_conversion_preserves_the_working_manifest(self):
        c = self.converter
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            target = folder / "target"; target.mkdir()
            draft = folder / "draft"; draft.mkdir()
            (target / "config.json").write_text("{}")
            (draft / "config.json").write_text('{"transformer_layer_config": {}}')
            (folder / "embedding.npy").write_bytes(b"legacy embedding")
            manifest = folder / "manifest.json"
            manifest.write_text('{"status": "complete", "original": true}')
            original = manifest.read_bytes()
            argv = ["convert", "--out", str(folder), "--target", str(target), "--draft", str(draft)]
            with patch("sys.argv", argv), \
                 patch.object(c.bench, "device_locks", side_effect=contextlib.nullcontext), \
                 patch.object(c.bench, "sha256", return_value="mock-sha"), \
                 patch.object(c.subprocess, "check_output", return_value="/mock/compiler\n"):
                with self.assertRaisesRegex(CacheMismatchError, "no fingerprint"):
                    c.main()
            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual((folder / "embedding.npy").read_bytes(), b"legacy embedding")

    def test_converter_reuses_identical_model_but_rejects_weight_and_shape_changes(self):
        import torch
        c = self.converter
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model = torch.nn.Linear(4, 3, bias=False, dtype=torch.float16)
            example = torch.zeros(1, 8, 4, dtype=torch.float16)

            class FakeModel:
                def save(self, destination):
                    path = Path(destination); path.mkdir()
                    (path / "weights.bin").write_bytes(b"compiled weights")

            def compile_model(source, destination_path):
                path = Path(destination_path); path.mkdir()
                (path / "weights.bin").write_bytes(b"compiled program")

            with patch.object(c.ct, "convert", return_value=FakeModel()) as convert, \
                 patch.object(c.ct.models.utils, "compile_model", side_effect=compile_model), \
                 patch.object(c, "placement", return_value={"math_all_ane": True, "math_counts": {"ANE": 1}}), \
                 contextlib.redirect_stdout(io.StringIO()):
                first = c.build(model, (example,), ["input"], ["output"], folder, 0, "per_tensor", {"source": "a"})
                cached = c.build(model, (example,), ["input"], ["output"], folder, 0, "per_tensor", {"source": "a"})
                self.assertEqual(first, cached); self.assertEqual(convert.call_count, 1)
                with self.assertRaises(CacheMismatchError):
                    c.build(model, (example[:, :1],), ["input"], ["output"], folder, 0, "per_tensor", {"source": "a"})
                with torch.no_grad(): model.weight[0, 0] += 1
                with self.assertRaises(CacheMismatchError):
                    c.build(model, (example,), ["input"], ["output"], folder, 0, "per_tensor", {"source": "a"})
                self.assertEqual(convert.call_count, 1)

    def test_driver_preserves_failed_trials_and_raises_on_mismatch(self):
        d = self.driver
        mx = SimpleNamespace(bfloat16="bf16", device_info=lambda: {"device_name": "mock"})
        weight = SimpleNamespace(dtype="bf16")
        model = SimpleNamespace(model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight)), eval=lambda: None)
        tok = SimpleNamespace(decode=lambda ids: str(ids))
        loader = SimpleNamespace(load=lambda _: (model, tok))
        stack = SimpleNamespace(manifest={"target": "target", "draft": "draft"})

        def generate(_, __, prompt, limit, speculative):
            return {"tokens": [1, 2, 99 if speculative else 3], "tok_per_s_decode": 20,
                    "cycles": 2, "accepted": 0, "profile_s": {}}

        with tempfile.TemporaryDirectory() as directory, \
             patch.dict("sys.modules", {"mlx_lm": loader}), \
             patch.object(d.bench, "require_metal", return_value=mx), \
             patch.object(d.bench, "device_locks", side_effect=contextlib.nullcontext), \
             patch.object(d.bench, "host_state", return_value={}), \
             patch.object(d.bench, "sha256", return_value="mock-sha"), \
             patch.object(d.bench, "stock_baseline", side_effect=lambda *args: {"tokens": [1, 2, 3], "generation_tps": 10}), \
             patch.object(d.importlib.metadata, "version", return_value="test"), \
             patch.object(d, "FullANEStack", return_value=stack), \
             patch.object(d, "generate", side_effect=generate), \
             contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory) / "result.json"
            with self.assertRaisesRegex(ValueError, "SD differs"):
                d.main(["--artifacts", directory, "--out", str(out), "--repeats", "1", "--skip-validation"])
            result = json.loads(out.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["rows"]), 12)
            self.assertIn("first mismatch 2", result["error"])


if __name__ == "__main__":
    unittest.main()
