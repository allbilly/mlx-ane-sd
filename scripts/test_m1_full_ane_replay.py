"""Check real HWX planning failures and padded tensor transport without ANE."""

import ctypes
from contextlib import ExitStack, nullcontext, redirect_stdout
import gzip
import hashlib
import io
import json
import platform
from pathlib import Path
import struct
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from m1_common import pack_tensor, unpack_tensor
from m1_ane_replay import (
    plan,
    BOInit,
    BOFree,
    Submit,
    InputTransport,
    Program,
    CacheOutput,
    finite_fp16,
)
from m1_full_ane_host import trace_matches
from m1_hwx_decode import container
from run_m1_full_ane_replay import verify_tensors
from package_m1_full_stack import verify_or_extract
from m1_lut6_weights import Checkpoint, reconstruct
import run_m1_full_ane_replay as runner


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = Path(__file__).resolve().parent.parent / "artifacts/m1-full-ane"
        cls.manifest = json.loads((cls.bundle / "manifest.json").read_text())
        cls.hwx = gzip.decompress(cls.read_file("programs/projector/kernel.hwx.gz"))
        cls.status = cls.read_file("programs/projector/status.json")
        cls.expected = json.loads(cls.read_file("programs/projector/linux-plan.json"))
        cls.expected["hwx_sha256"] = hashlib.sha256(cls.hwx).hexdigest()

    @classmethod
    def read_file(cls, name):
        recipe = cls.manifest["files"][name]
        raw = (cls.bundle / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != recipe["sha256"]:
            raise ValueError("Bad reference file")
        return bytes(raw)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "hwx").mkdir()
        (self.root / "hwx/model.hwx").write_bytes(self.hwx)
        (self.root / "status.json").write_bytes(self.status)

    def test_real_full_projector_plan_matches_committed_binding_contract(self):
        self.assertEqual(plan(self.root), self.expected)
        self.assertEqual(
            {i["name"] for i in self.expected["io"]},
            {"cos", "sin", "features", "new_k", "new_v"},
        )

    def test_aliased_request_surfaces_are_rejected(self):
        s = json.loads(self.status)
        s["NetworkStatusList"][0]["LiveInputList"].append(
            s["NetworkStatusList"][0]["LiveInputList"][0]
        )
        (self.root / "status.json").write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError, "surfaces alias"):
            plan(self.root)

    def test_interior_bar_cannot_be_silently_bound_to_segment_base(self):
        c = container(self.hwx)
        command = next(
            cmd
            for cmd in c["load_commands"]
            if cmd["command"] == 4
            and struct.unpack_from("<I", bytes.fromhex(cmd["raw_hex"]), 8)[0] == 1
        )
        raw = bytearray(self.hwx)
        pos = command["offset"] + 16 + 4 * 8
        struct.pack_into("<Q", raw, pos, struct.unpack_from("<Q", raw, pos)[0] + 64)
        (self.root / "hwx/model.hwx").write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "interior or alias"):
            plan(self.root)

    def test_driver_constant_delimiter_must_follow_all_tasks(self):
        c = container(self.hwx)
        command = next(
            cmd
            for cmd in c["load_commands"]
            if cmd["command"] == 4
            and struct.unpack_from("<I", bytes.fromhex(cmd["raw_hex"]), 8)[0] == 1
        )
        raw = bytearray(self.hwx)
        base = struct.unpack_from("<Q", raw, command["offset"] + 16)[0]
        struct.pack_into("<Q", raw, command["offset"] + 24, base + 16)
        (self.root / "hwx/model.hwx").write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "BAR1 delimiter"):
            plan(self.root)

    def test_padded_causal_mask_transport_preserves_values_and_zeroes_padding(self):
        d = {
            "Type": "Float16",
            "Interleave": 1,
            "Batches": 1,
            "Channels": 1,
            "Depth": 1,
            "Height": 8,
            "Width": 264,
            "BatchStride": 4608,
            "PlaneStride": 4608,
            "DepthStride": 4608,
            "RowStride": 576,
        }
        mask = np.full((1, 1, 8, 264), -10000, np.float16)
        mask[0, 0, 0, 0] = 0
        raw = pack_tensor(mask, d)
        self.assertEqual(unpack_tensor(raw, d, mask.shape).tobytes(), mask.tobytes())
        self.assertTrue(
            all(raw[i * 576 + 528 : (i + 1) * 576] == bytes(48) for i in range(8))
        )
        mask[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            pack_tensor(mask, d)

    def test_driver_structure_sizes_match_ane_accel_abi(self):
        self.assertEqual(
            (ctypes.sizeof(BOInit), ctypes.sizeof(BOFree), ctypes.sizeof(Submit)),
            (24, 8, 152),
        )

    def test_regenerated_input_mismatch_stops_golden_verification(self):
        d = {
            "Type": "Float16",
            "Interleave": 1,
            "Batches": 1,
            "Channels": 1,
            "Depth": 1,
            "Height": 1,
            "Width": 8,
            "BatchStride": 64,
            "PlaneStride": 64,
            "DepthStride": 64,
            "RowStride": 64,
        }
        value = np.zeros((1, 1, 1, 8), np.float16)
        expected = [
            {
                "name": "mask",
                "shape": list(value.shape),
                "compiler_geometry": d,
                "sha256": hashlib.sha256(pack_tensor(value, d)).hexdigest(),
            }
        ]
        verify_tensors({"mask": value}, expected)
        value[0, 0, 0, 0] = 1
        with self.assertRaisesRegex(RuntimeError, "Regenerated tensor differs"):
            verify_tensors({"mask": value}, expected)

    def test_compact_bundle_contains_no_learned_banks_or_tensor_binaries(self):
        manifest = verify_or_extract(self.bundle)
        self.assertLess(manifest["stored_bytes"], 16 * 1024 * 1024)
        self.assertFalse(manifest["learned_weight_banks_included"])
        self.assertFalse(manifest["captured_tensor_binaries_included"])
        recipe = json.loads(self.read_file("programs/projector/packing.json"))
        for op in recipe["operations"]:
            self.assertFalse(any(self.hwx[op["offset"] : op["offset"] + op["size"]]))

    def test_external_projector_reconstructs_complete_captured_hwx(self):
        draft = self.bundle.parent.parent / ".asahi/models/m1-full-ane-draft"
        if not (draft / "model.safetensors").exists():
            self.skipTest("Pinned draft not downloaded locally")
        stack = json.loads(self.read_file("stack.json"))
        reader = Checkpoint(draft, stack["draft"]["safetensors_sha256"])
        self.addCleanup(reader.close)
        root = self.bundle / "programs/projector"
        recipe = json.loads((root / "packing.json").read_text())
        raw = reconstruct(root, recipe, {"draft": reader})
        self.assertEqual(hashlib.sha256(raw).hexdigest(), recipe["sha256"])
        self.assertNotEqual(raw, self.hwx)

    def test_changed_external_checkpoint_is_rejected(self):
        (self.root / "model.safetensors").write_bytes(b"not the pinned model")
        with self.assertRaisesRegex(ValueError, "Checkpoint differs"):
            Checkpoint(self.root, "0" * 64)


class TransportTests(unittest.TestCase):
    descriptor = {
        "Type": "Float16",
        "Interleave": 1,
        "Batches": 2,
        "Channels": 3,
        "Depth": 2,
        "Height": 9,
        "Width": 5,
        "BatchStride": 1920,
        "PlaneStride": 640,
        "DepthStride": 320,
        "RowStride": 32,
    }

    class Surface:
        def __init__(self, size):
            self.data = bytearray(size)
            self.written = 0

        def write(self, data):
            self.data[: len(data)] = data
            self.written += len(data)

        def write_at(self, data, offset):
            self.data[offset : offset + len(data)] = data
            self.written += len(data)

    def test_finite_check_agrees_for_every_fp16_bit_pattern(self):
        values = np.arange(65536, dtype="<u2").view("<f2")
        finite = np.isfinite(values)
        self.assertTrue(finite_fp16(values[finite]))
        for value in values[~finite]:
            self.assertFalse(finite_fp16(value))

    def test_reused_storage_matches_reference_with_padding_and_strided_inputs(self):
        transport = InputTransport(self.descriptor)
        surface = self.Surface(transport.size)
        values = np.arange(np.prod(transport.shape), dtype=np.float16).reshape(
            transport.shape
        )
        for value in (values, values[..., ::-1], np.zeros_like(values)):
            transport.write(value, surface)
            self.assertEqual(bytes(surface.data), pack_tensor(value, self.descriptor))

    def test_output_prefix_matches_compiler_layout_and_rejects_stale_or_nonfinite_rows(
        self,
    ):
        d = {
            "Type": "Float16",
            "Interleave": 1,
            "Batches": 2,
            "Channels": 3,
            "Depth": 1,
            "Height": 8,
            "Width": 5,
            "BatchStride": 768,
            "PlaneStride": 256,
            "DepthStride": 768,
            "RowStride": 32,
        }
        logical = (2, 1, 3, 8, 5)
        values = np.arange(np.prod(logical), dtype=np.float16).reshape(logical)
        raw = bytearray(pack_tensor(values, d))
        program = SimpleNamespace(
            epoch=3, buffers={4: SimpleNamespace(map=raw)}, shapes={"new_k": logical}
        )
        tensor = {"name": "new_k", "bank": 4, "compiler_geometry": d}
        pending = CacheOutput(program, tensor)
        for count in (1, 3, 8):
            self.assertEqual(
                pending.read_prefix(count).tobytes(), values[:, :, :, :count].tobytes()
            )
        with self.assertRaisesRegex(ValueError, "prefix"):
            pending.read_prefix(9)
        values[:, :, :, 4:] = np.nan
        raw[:] = pack_tensor(np.zeros(logical, np.float16), d)
        view = np.ndarray(
            (2, 3, 1, 8, 5), dtype="<f2", buffer=raw, strides=(768, 256, 768, 32, 2)
        )
        view[...] = values.reshape(view.shape)
        self.assertEqual(
            pending.read_prefix(4).tobytes(), values[:, :, :, :4].tobytes()
        )
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            pending.read_prefix(8)
        program.epoch += 1
        with self.assertRaisesRegex(ValueError, "Stale"):
            pending.read_prefix(1)

    def test_early_stop_reads_through_first_rejection_or_bonus_and_preserves_trace(
        self,
    ):
        program = Program.__new__(Program)
        program.verify_readback = "accepted-prefix"
        predictions = [3, 5, 8, 13]
        for candidates, count in (
            ([99, 5, 8], 1),
            ([3, 99, 8], 2),
            ([3, 5, 99], 3),
            ([3, 5, 8], 4),
            ([], 1),
        ):
            read = Mock(side_effect=lambda start, end: predictions[start:end])
            program.read_token_ids = read
            actual = program.read_verified_ids(candidates, 0, len(candidates) + 1)
            self.assertEqual(actual, predictions[:count])
            self.assertEqual(read.call_count, count)
        golden = [
            {
                "candidates": [3, 9, 8],
                "predictions": predictions,
                "accepted": 1,
                "committed": [3, 5],
                "offset": 10,
            }
        ]
        actual = [dict(golden[0], predictions=[3, 5])]
        self.assertTrue(trace_matches(actual, golden))
        for field, value in (
            ("predictions", [3]),
            ("predictions", [3, 6]),
            ("accepted", 2),
            ("offset", 11),
        ):
            self.assertFalse(trace_matches([dict(actual[0], **{field: value})], golden))

    def test_resident_cache_writes_only_committed_prefix_and_resets_between_prompts(
        self,
    ):
        program = Program.__new__(Program)
        program.transport = "resident"
        program.cache_ready = False
        program.cache_end = 0
        program.inputs = {
            name: InputTransport(self.descriptor) for name in ("cache_k", "cache_v")
        }
        program.input_banks = {"cache_k": 0, "cache_v": 1}
        program.buffers = {
            bank: self.Surface(program.inputs["cache_k"].size) for bank in (0, 1)
        }
        program.reset_transport()
        expected = {
            name: np.zeros(transport.shape, np.float16)
            for name, transport in program.inputs.items()
        }
        # Two speculative blocks contain eight rows; only accepted prefixes enter the cache.
        block = np.arange(2 * 3 * 2 * 8 * 5, dtype=np.float16).reshape(2, 3, 2, 8, 5)
        for offset, count in ((0, 1), (1, 3)):
            written = sum(buffer.written for buffer in program.buffers.values())
            program.commit_cache(
                block[:, :, :, :count], -block[:, :, :, :count], offset, count
            )
            self.assertLess(
                sum(buffer.written for buffer in program.buffers.values()) - written,
                2 * program.inputs["cache_k"].size,
            )
            for name, sign in (("cache_k", 1), ("cache_v", -1)):
                expected[name][:, :, :, offset : offset + count] = (
                    sign * block[:, :, :, :count]
                )
                self.assertEqual(
                    bytes(program.buffers[program.input_banks[name]].data),
                    pack_tensor(expected[name], self.descriptor),
                )
        before = {bank: bytes(buffer.data) for bank, buffer in program.buffers.items()}
        invalid = block[:, :, :, :1].copy()
        invalid[0, 0, 0, 0, 0] = np.inf
        for offset, count, v in (
            (5, 1, block[:, :, :, :1]),
            (4, 6, block[:, :, :, :6]),
            (4, 1, invalid),
        ):
            with self.assertRaises(ValueError):
                program.commit_cache(block[:, :, :, :count], v, offset, count)
            self.assertEqual(program.cache_end, 4)
            self.assertEqual(
                before,
                {bank: bytes(buffer.data) for bank, buffer in program.buffers.items()},
            )
        program.reset_transport()
        self.assertEqual(program.cache_end, 0)
        self.assertTrue(
            all(
                bytes(buffer.data) == bytes(len(buffer.data))
                for buffer in program.buffers.values()
            )
        )

    def test_mapped_and_row_reads_match_full_argmax_with_chunk_order_ties_and_padding(
        self,
    ):
        program = Program.__new__(Program)
        program.meta = {"io": []}
        program.buffers = {}
        program.shapes = {}
        program.kv_readback = "full"
        program.native_vocabulary = None
        chunks = []
        # More than ten numbered outputs catches lexicographic chunk ordering.
        for i in range(12):
            width = 3 if i == 11 else 8
            descriptor = {
                "Type": "Float16",
                "Interleave": 1,
                "Batches": 1,
                "Channels": 1,
                "Depth": 1,
                "Height": 4,
                "Width": width,
                "BatchStride": 128,
                "PlaneStride": 128,
                "DepthStride": 128,
                "RowStride": 32,
            }
            values = np.full((1, 4, width), -10 - i, np.float16)
            values[0, 1, (i + 1) % width] = 20
            values[0, 2, (i + 2) % width] = i
            values[0, 3, :] = np.arange(width, dtype=np.float16)
            tensor = {
                "name": f"logits{i}",
                "direction": "outputs",
                "bank": i,
                "compiler_geometry": descriptor,
                "allocation_bytes": 128,
            }
            surface = self.Surface(128)
            surface.write(pack_tensor(values, descriptor))
            surface.map = surface.data
            program.meta["io"].append(tensor)
            program.buffers[i] = surface
            program.shapes[f"logits{i}"] = values.shape
            chunks.append(values)
        program.meta["io"].reverse()
        expected = (
            np.concatenate(chunks, axis=-1)[0].astype(np.float32).argmax(-1).tolist()
        )
        modes = ["full", "rows", "mapped"]
        if platform.system() == "Linux" and platform.machine() in ("arm64", "aarch64"):
            modes.append("native")
        for mode in modes:
            program.head_readback = mode
            self.assertEqual(program.read_token_ids(0, 4), expected)
            self.assertEqual(program.read_token_ids(1, 3), expected[1:3])
        if "native" in modes:
            from m1_native_transport import configure

            configure(4)
            try:
                self.assertEqual(program.read_token_ids(0, 4), expected)
            finally:
                configure(1)
        # NaN in an unused physical row must not affect the selected rows.
        program.buffers[0].map[:2] = np.array([np.nan], dtype="<f2").tobytes()
        for mode in modes[1:]:
            program.head_readback = mode
            self.assertEqual(program.read_token_ids(1, 3), expected[1:3])
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                program.read_token_ids(0, 1)
            with self.assertRaisesRegex(ValueError, "requested rows|vocabulary rows"):
                program.read_token_ids(0, 5)

    @unittest.skipUnless(
        platform.system() == "Linux" and platform.machine() in ("arm64", "aarch64"),
        "M1 native CPU transport",
    )
    def test_native_reduction_preserves_all_finite_fp16_values_and_rejects_nonfinite(
        self,
    ):
        from m1_native_transport import argmax, configure

        self.addCleanup(configure, 1)
        bits = np.arange(65536, dtype="<u2")
        half = bits.view("<f2")
        finite = half[np.isfinite(half)]
        values = np.repeat(finite[:, None], 8, axis=1)
        ids, maxima = argmax(values, 0, len(values), 8, 16)
        self.assertTrue(np.all(ids == 0))
        self.assertTrue(
            np.array_equal(maxima.view("<u4"), finite.astype("<f4").view("<u4"))
        )
        rng = np.random.default_rng(31)
        for width in (1, 3, 7, 8, 9, 19, 8192, 65536):
            values = rng.choice(finite, size=(4, width)).astype("<f2")
            ids, maxima = argmax(values, 0, 4, width, width * 2)
            expected = values.astype(np.float32).argmax(-1)
            self.assertEqual(ids.tolist(), expected.tolist())
            self.assertTrue(
                np.array_equal(
                    maxima, values[np.arange(4), expected].astype(np.float32)
                )
            )
        for width in (3, 8, 19):
            for invalid in (np.nan, np.inf, -np.inf):
                values = np.zeros((2, width), np.float16)
                values[1, -1] = invalid
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    argmax(values, 0, 2, width, width * 2)
        configure(4)
        ids, maxima = argmax(
            np.repeat(finite[:, None], 8, axis=1), 0, len(finite), 8, 16
        )
        self.assertTrue(np.all(ids == 0))
        self.assertTrue(
            np.array_equal(maxima.view("<u4"), finite.astype("<f4").view("<u4"))
        )
        values = np.zeros((8, 19), np.float16)
        values[-1, 8] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            argmax(values, 0, 8, 19, 38)


class BenchmarkTests(unittest.TestCase):
    def exercise_benchmark(self, warmup_error=None):
        """Exercise the CLI/receipt flow with a cold GPU and no ANE device."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        receipt = root / "bench.json"
        bundle = Path(__file__).resolve().parent.parent / "artifacts/m1-full-ane"
        stack = json.loads((bundle / "stack.json").read_text())
        plans = {
            info["program"]: {"hwx_sha256": info["hwx_sha256"]}
            for info in stack["artifacts"]
        }
        events = []
        cold = True

        def record(label, prompt, limit):
            status = json.loads(receipt.read_text())["status"]
            events.append((label, prompt, limit, status))
            return status

        def stock(model, tokenizer, prompt, limit):
            nonlocal cold
            record("mlx_bf16", prompt, limit)
            if warmup_error:
                raise warmup_error
            rate = 1.0 if cold else 100.0
            cold = False
            return {
                "tokens": [7] * limit,
                "generation_tokens": limit,
                "generation_tps": rate,
            }

        def ane(host, tokenizer, prompt, limit, speculative, num_draft=None):
            record("ane_lut6_sd" if speculative else "ane_lut6_ar", prompt, limit)
            return {"tokens": [7] * limit, "tok_per_s_decode": 100.0}

        def verified(root, programs, capture, values, result, save):
            result["goldens_verified"] = True
            save()

        model = Mock()
        tokenizer = SimpleNamespace(decode=lambda ids: str(ids))
        mlx_lm = ModuleType("mlx_lm")
        mlx_lm.load = Mock(return_value=(model, tokenizer))
        argv = [
            "run_m1_full_ane_replay.py",
            str(bundle),
            "--mode",
            "bench",
            "--target",
            str(root / "target"),
            "--draft",
            str(root / "draft"),
            "--cache",
            str(root / "cache"),
            "--max-new",
            "12",
            "--repeats",
            "2",
            "--out",
            str(receipt),
        ]
        with ExitStack() as context:
            context.enter_context(patch("sys.argv", argv))
            context.enter_context(patch.dict("sys.modules", {"mlx_lm": mlx_lm}))
            for name, replacement in {
                "check_sources": Mock(return_value=(stack, plans)),
                "device_locks": lambda: nullcontext(),
                "device_path": lambda requested: root / "device",
                "prepare": Mock(return_value=root / "cache"),
                "embedding": Mock(return_value=None),
                "Program": Mock(),
                "verify_cases": verified,
                "FullANEStack": Mock(),
                "stock_baseline": stock,
                "generate": ane,
                "synchronize_mlx": Mock(),
            }.items():
                context.enter_context(patch.object(runner, name, replacement))
            context.enter_context(
                patch.object(
                    runner,
                    "os",
                    SimpleNamespace(
                        open=Mock(return_value=99),
                        close=Mock(),
                        O_RDWR=runner.os.O_RDWR,
                        O_CLOEXEC=runner.os.O_CLOEXEC,
                        sched_getaffinity=lambda pid: {0, 1, 2, 3, 4, 5, 6, 7},
                    ),
                )
            )
            context.enter_context(redirect_stdout(io.StringIO()))
            if warmup_error:
                with self.assertRaisesRegex(RuntimeError, str(warmup_error)):
                    runner.main()
            else:
                runner.main()
        return json.loads(receipt.read_text()), events, model

    def test_cold_stock_baseline_is_warmed_and_excluded_from_speedup(self):
        result, events, model = self.exercise_benchmark()
        self.assertEqual(result["status"], "complete")
        model.eval.assert_called_once()
        self.assertEqual(len(result["warmup"]["calls"]), 12)
        self.assertTrue(result["warmup"]["excluded_from_rows"])
        self.assertEqual(
            [event[3] for event in events], ["warming"] * 12 + ["benchmarking"] * 24
        )
        self.assertEqual([event[2] for event in events], [12] * 12 + [12] * 24)
        self.assertEqual(
            {(row["name"], row["impl"]) for row in result["warmup"]["calls"]},
            {(row["name"], row["impl"]) for row in result["rows"]},
        )
        self.assertEqual(len(result["rows"]), 24)
        baseline = [row for row in result["rows"] if row["impl"] == "mlx_bf16"]
        self.assertTrue(
            all(
                row["generation_tps"] == row["tok_per_s_decode"] == 100.0
                for row in baseline
            )
        )
        self.assertEqual(result["sd_vs_linux_mlx"], 1.0)
        self.assertIn("stream_generate", result["mlx_baseline"])
        self.assertIn("first token excluded", result["timing_by_impl"]["mlx_bf16"])

    def test_failed_warmup_writes_failure_without_measured_rows(self):
        result, events, _ = self.exercise_benchmark(RuntimeError("GPU warm-up failed"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "GPU warm-up failed")
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["warmup"]["calls"], [])
        self.assertEqual([event[3] for event in events], ["warming"])


if __name__ == "__main__":
    unittest.main()
