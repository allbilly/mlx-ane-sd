"""Check shared checkpoint decoding and host acceptance without a device."""

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from m1_common import BF16Tensors, sha256
from m1_lut6_weights import Checkpoint
from m1_full_ane_host import FullANEStack, generate


class CheckpointTests(unittest.TestCase):
    def test_converter_and_reconstructor_decode_identical_bf16_rows(self):
        bits = np.array(
            [0, 0x8000, 0x3F80, 0xBF80, 0x3FC0, 0x4010], dtype="<u2"
        ).reshape(3, 2)
        expected = (bits.astype(np.uint32) << 16).view(np.float32)
        header = json.dumps(
            {"weight": {"dtype": "BF16", "shape": [3, 2], "data_offsets": [0, 12]}}
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + bits.tobytes())
            converter = BF16Tensors(path)
            replay = Checkpoint(directory, sha256(path))
            try:
                self.assertEqual(converter.read("weight").tobytes(), expected.tobytes())
                self.assertEqual(
                    replay.read("weight").tobytes(),
                    expected.astype(np.float16).tobytes(),
                )
                self.assertEqual(
                    replay.read("weight", [1, 3]).tobytes(),
                    converter.read("weight", np.float16, slice(1, 3)).tobytes(),
                )
            finally:
                converter.close()
                replay.close()

    def test_truncated_header_and_payload_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            path.write_bytes(struct.pack("<Q", 100) + b"{}")
            with self.assertRaisesRegex(ValueError, "header"):
                BF16Tensors(path)
            header = json.dumps(
                {"weight": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}
            ).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(2))
            reader = BF16Tensors(path)
            try:
                with self.assertRaisesRegex(ValueError, "extent"):
                    reader.read("weight")
            finally:
                reader.close()


class FixtureProgram:
    """Simple token transition with one accepted and one rejected draft row."""

    def __init__(self, role):
        self.role = role

    def predict(self, inputs, role=None):
        if self.role == "head":
            hidden = inputs["hidden"]
            ids = (hidden[0, :, 0].astype(int) + 1) % 10
            logits = np.full((1, 4, 10), -10, np.float16)
            logits[0, np.arange(4), ids] = 10
            return {"logits0": logits}
        if self.role == "projector":
            hidden = inputs["features"]
        else:
            hidden = inputs["hidden"].copy()
        if self.role == "draft":
            anchor = int(hidden[0, 0, 0])
            hidden[0, :, 0] = [anchor, anchor, 6, anchor + 2]
            return {"hidden_out": hidden}
        kv = hidden.reshape(1, 1, 1, 4, 2)
        result = {"new_k": kv, "new_v": kv * 2}
        if self.role == "target":
            result.update(hidden_out=hidden, captures=hidden[None])
        return result


def fake_coreml():
    return SimpleNamespace(
        ComputeUnit=SimpleNamespace(CPU_AND_NE="ANE"),
        models=SimpleNamespace(
            CompiledMLModel=lambda path, **kwargs: FixtureProgram(path)
        ),
    )


def make_manifest(root):
    manifest = {
        "status": "complete",
        "math_all_ane": True,
        "block": 4,
        "capacity": 32,
        "config": {
            "head_dim": 2,
            "num_key_value_heads": 1,
            "hidden_size": 2,
            "num_hidden_layers": 1,
            "vocab_size": 10,
            "rope_theta": 10000,
        },
        "draft_config": {
            "aux_hidden_state_layer_ids": [1],
            "mask_token_id": 9,
            "transformer_layer_config": {"num_hidden_layers": 1},
        },
        "artifacts": [
            {
                "role": role,
                "compiled": role,
                "placement": {"math_all_ane": True},
                **(
                    {"start": 0, "count": 1, "captures": [1]}
                    if role == "target"
                    else {}
                ),
            }
            for role in ("target", "projector", "draft", "head")
        ],
    }
    for filename in ("manifest.json", "stack.json"):
        (root / filename).write_text(json.dumps(manifest))
    embedding = np.column_stack((np.arange(10), np.ones(10))).astype(np.float16)
    np.save(root / "embedding.npy", embedding)
    return embedding


class SharedHostTests(unittest.TestCase):
    def test_coreml_and_linux_share_rejection_eos_and_committed_cache_behavior(self):
        path = Path(__file__).with_name("macos_m4_coreml_runtime.py")
        spec = importlib.util.spec_from_file_location("test_coreml_adapter", path)
        adapter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"coremltools": fake_coreml()}):
            spec.loader.exec_module(adapter)
        self.assertIs(adapter.generate, generate)
        self.assertTrue(issubclass(adapter.FullANEStack, FullANEStack))
        tokenizer = SimpleNamespace(
            encode=lambda _: [0], decode=lambda ids: str(ids), eos_token_ids={5}
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedding = make_manifest(root)
            for speculative in (False, True):
                linux = FullANEStack(
                    root, embedding, lambda info: FixtureProgram(info["role"])
                )
                macos = adapter.FullANEStack(root)
                observed = [
                    generate(stack, tokenizer, "prompt", 8, speculative)
                    for stack in (linux, macos)
                ]
                self.assertEqual(observed[0]["tokens"], [1, 2, 3, 4, 5])
                self.assertEqual(observed[0]["trace"], observed[1]["trace"])
                self.assertEqual(linux.offset, 5)
                self.assertEqual(macos.offset, 5)
                for name in ("cache_k", "cache_v", "draft_k", "draft_v"):
                    self.assertEqual(
                        getattr(linux, name).tobytes(), getattr(macos, name).tobytes()
                    )
                self.assertEqual(
                    linux.cache_k[0, 0, 0, :5, 0].tolist(), [0, 1, 2, 3, 4]
                )
                if speculative:
                    self.assertEqual(
                        [t["accepted"] for t in observed[0]["trace"]], [1, 1]
                    )
                    self.assertEqual(linux.draft_offset, 5)
                    self.assertEqual(
                        linux.draft_k[0, 0, 0, :5, 0].tolist(), [0, 1, 2, 3, 4]
                    )


if __name__ == "__main__":
    unittest.main()
