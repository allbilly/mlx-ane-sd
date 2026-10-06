"""Shared M1 checkpoint, tensor transport and benchmark helpers.

Model and device imports stay lazy so reference inspection needs only NumPy.
"""

from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import mmap
from pathlib import Path
import struct
import subprocess
import numpy as np
from bench_final_stack import PROMPTS as PROMPTS

REPO = Path(__file__).resolve().parent.parent
_LOCK_DEPTH = 0


def host_state():
    state = {}
    for name, command in (
        ("swap", ["sysctl", "vm.swapusage"]),
        ("thermal", ["pmset", "-g", "therm"]),
    ):
        probe = subprocess.run(command, text=True, capture_output=True)
        state[name] = {
            "returncode": probe.returncode,
            "stdout": probe.stdout.strip(),
            "stderr": probe.stderr.strip(),
        }
    return state


def require_metal():
    # Check before waiting for device locks: a restricted terminal can import
    # stdlib successfully while MLX cannot access the Metal device at all.
    try:
        import mlx.core as mx
    except ImportError as error:
        raise RuntimeError(f"MLX cannot access Metal: {error}") from error
    if not mx.metal.is_available():
        raise RuntimeError("Native Metal MLX required")
    return mx


@contextlib.contextmanager
def device_locks():
    # A synchronous sweep can hold the reservation across several main() calls.
    global _LOCK_DEPTH
    if _LOCK_DEPTH:
        _LOCK_DEPTH += 1
        try:
            yield
        finally:
            _LOCK_DEPTH -= 1
        return
    with contextlib.ExitStack() as stack:
        for name in ("ane.lock", "gpu.lock"):
            path = Path.home() / name
            try:
                stream = path.open("r")
            except FileNotFoundError:
                stream = path.open("a")
            lock = stack.enter_context(stream)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"[wait] another workload holds {name}", flush=True)
                fcntl.flock(lock, fcntl.LOCK_EX)
        _LOCK_DEPTH = 1
        try:
            yield
        finally:
            _LOCK_DEPTH = 0


def target_forward(model, ids, cache, feature_ids=(), logits=True):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    h = model.model.embed_tokens(mx.array(ids, mx.uint32)[None])
    mask = create_attention_mask(h, cache[0])
    captured = {}
    for i, (layer, c) in enumerate(zip(model.model.layers, cache), 1):
        h = layer(h, mask, c)
        if i in feature_ids:
            captured[i] = h
    features = (
        mx.concatenate([captured[i] for i in feature_ids], axis=-1)[0]
        if feature_ids
        else None
    )
    if logits:
        h = model.model.norm(h)
        scores = (
            model.model.embed_tokens.as_linear(h)
            if model.args.tie_word_embeddings
            else model.lm_head(h)
        )
        tokens = mx.argmax(scores, axis=-1)[0]
    else:
        tokens = None
    mx.eval(
        [c.state for c in cache],
        *([tokens] if tokens is not None else []),
        *([features] if features is not None else []),
    )
    return tokens.tolist() if tokens is not None else None, features


def stock_baseline(model, tokenizer, prompt, limit):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    last = None
    tokens = []
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=limit,
        sampler=make_sampler(temp=0),
        prefill_step_size=32,
    ):
        tokens.append(response.token)
        last = response
    return {
        "tokens": tokens,
        "generation_tps": last.generation_tps,
        "generation_tokens": last.generation_tokens,
    }


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def geometry(desc):
    if desc["Type"] != "Float16" or desc["Interleave"] != 1:
        raise ValueError(
            "Only compiler-declared non-interleaved fp16 tensors are supported"
        )
    shape = tuple(desc[k] for k in ("Batches", "Channels", "Depth", "Height", "Width"))
    strides = tuple(
        desc[k] for k in ("BatchStride", "PlaneStride", "DepthStride", "RowStride")
    ) + (2,)
    size = desc["BatchStride"] * desc["Batches"]
    extent = sum((n - 1) * s for n, s in zip(shape, strides)) + 2
    if min(shape) <= 0 or extent > size:
        raise ValueError("Invalid compiler tensor extent")
    return shape, strides, size


def pack_tensor(values, desc):
    shape, strides, size = geometry(desc)
    values = np.asarray(values, dtype="<f2")
    if values.size != np.prod(shape) or not np.isfinite(values).all():
        raise ValueError("Wrong-size or nonfinite real tensor")
    raw = bytearray(size)
    np.ndarray(shape, dtype="<f2", buffer=raw, strides=strides)[...] = values.reshape(
        shape
    )
    return bytes(raw)


def unpack_tensor(raw, desc, logical_shape):
    shape, strides, size = geometry(desc)
    if len(raw) != size:
        raise ValueError("Wrong-size surface output")
    return (
        np.ndarray(shape, dtype="<f2", buffer=raw, strides=strides)
        .copy()
        .reshape(logical_shape)
    )


class BF16Tensors:
    """Read BF16 safetensors without importing or executing model code."""

    def __init__(self, path):
        self.file = Path(path).open("rb")
        try:
            self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            if len(self.map) < 8:
                raise ValueError("Invalid safetensors header")
            size = struct.unpack_from("<Q", self.map)[0]
            if 8 + size > len(self.map):
                raise ValueError("Invalid safetensors header")
            self.header = json.loads(self.map[8 : 8 + size])
            self.offset = 8 + size
        except BaseException:
            if hasattr(self, "map"):
                self.map.close()
            self.file.close()
            raise

    def read(self, name, dtype=np.float32, rows=None):
        meta = self.header[name]
        if meta["dtype"] != "BF16":
            raise ValueError(f"Expected BF16 weights: {name}")
        begin, end = meta["data_offsets"]
        shape = meta["shape"]
        if (
            begin < 0
            or end < begin
            or any(n < 1 for n in shape)
            or end - begin != int(np.prod(shape)) * 2
            or self.offset + end > len(self.map)
        ):
            raise ValueError(f"Invalid tensor extent: {name}")
        bits = np.ndarray(
            shape, dtype="<u2", buffer=self.map, offset=self.offset + begin
        )
        if rows is not None:
            bits = bits[rows]
        return (bits.astype(np.uint32) << 16).view(np.float32).astype(dtype)

    def close(self):
        self.map.close()
        self.file.close()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    """Atomically checkpoint a receipt or manifest, including failure updates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class ReferenceTokenizer:
    """Use recorded prompt IDs when verifying kernels, without model downloads."""

    def __init__(self, generations, eos_token_id):
        self.prompts = {g["prompt"]: g["prompt_tokens"] for g in generations}
        self.eos_token_ids = {eos_token_id}

    def encode(self, prompt):
        return self.prompts[prompt]

    def decode(self, tokens):
        return ""


class ProgramAdapter:
    """Delegate transport operations shared by verification and profiling."""

    def __init__(self, program):
        self.program = program

    def reset_transport(self):
        self.program.reset_transport()

    def commit_cache(self, *args):
        self.program.commit_cache(*args)
