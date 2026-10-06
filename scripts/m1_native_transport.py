"""Build/load this repo's exact NEON readback helper on M1 Linux."""

from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import tempfile
import numpy as np

_loaded = None
_threads = 1


def configure(threads):
    global _threads
    if not 1 <= threads <= 32:
        raise ValueError("Native readback threads must be in 1..32")
    _threads = threads


def library():
    global _loaded
    if _loaded is not None:
        return _loaded
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("Native transport requires arm64 Linux with FP16 NEON")
    root = Path(__file__).resolve().parent.parent
    source = root / "asahi/m1_transport.c"
    compiler = shlex.split(os.environ.get("CC", "cc"))
    version = subprocess.run(
        [*compiler, "--version"], check=True, capture_output=True, text=True
    ).stdout
    flags = [
        "-O3",
        "-std=c11",
        "-fPIC",
        "-shared",
        "-march=armv8.2-a+fp16",
        "-fno-fast-math",
        "-fopenmp",
    ]
    recipe = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": compiler,
        "compiler_version": version,
        "flags": flags,
    }
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    build = root / ".asahi/build"
    build.mkdir(parents=True, exist_ok=True)
    path = build / f"libm1_transport_{key}.so"
    if not path.exists():
        with tempfile.TemporaryDirectory(dir=build) as temporary:
            output = Path(temporary) / "transport.so"
            try:
                subprocess.run(
                    [*compiler, *flags, str(source), "-o", str(output)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    "Native transport compilation failed:\n" + error.stderr
                ) from error
            os.replace(output, path)
    # Park readback workers between calls so they do not compete with ANE
    # coordination or the paired MLX baseline. Respect explicit user settings.
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("GOMP_SPINCOUNT", "0")
    loaded = ctypes.CDLL(str(path))
    loaded.m1_fp16_argmax.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    loaded.m1_fp16_argmax.restype = ctypes.c_int
    loaded.m1_fp16_argmax_chunks.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    loaded.m1_fp16_argmax_chunks.restype = ctypes.c_int
    recipe["library_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    recipe["wait_policy"] = {
        n: os.environ.get(n) for n in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT")
    }
    _loaded = (loaded, recipe)
    return _loaded


def argmax(mapping, offset, rows, width, row_stride):
    """The synchronous caller owns mapping and holds it open for this call."""
    mapping = memoryview(mapping).cast("B")
    if not (
        0 <= offset
        and rows > 0
        and width > 0
        and row_stride >= width * 2
        and row_stride % 2 == 0
        and offset + (rows - 1) * row_stride + width * 2 <= len(mapping)
    ):
        raise ValueError("Invalid native vocabulary extent")
    loaded, _ = library()
    address = ctypes.addressof(ctypes.c_char.from_buffer(mapping)) + offset
    ids = np.empty(rows, np.int32)
    values = np.empty(rows, np.float32)
    status = loaded.m1_fp16_argmax(
        address, rows, width, row_stride, ids.ctypes.data, values.ctypes.data, _threads
    )
    if status:
        raise ValueError(
            "Unwritten/nonfinite native vocabulary logits"
            if status == -2
            else "Invalid native vocabulary layout"
        )
    return ids, values


class Vocabulary:
    """Persistent pointer/geometry plan; release pinned mappings before BO close."""

    def __init__(self, surfaces):
        pointers = []
        strides = []
        widths = []
        heights = []
        self.views = []
        for mapping, height, width, row_stride in surfaces:
            if not (
                height > 0
                and width > 0
                and row_stride >= width * 2
                and row_stride % 2 == 0
                and (height - 1) * row_stride + width * 2 <= len(mapping)
            ):
                raise ValueError("Invalid native vocabulary surface")
            pointers.append(ctypes.addressof(ctypes.c_char.from_buffer(mapping)))
            self.views.append(memoryview(mapping))
            strides.append(row_stride)
            widths.append(width)
            heights.append(height)
        if not pointers:
            raise ValueError("Empty vocabulary plan")
        self.pointers = np.array(pointers, np.uintp)
        self.strides = np.array(strides, np.uint64)
        self.widths = np.array(widths, np.uint32)
        self.height = min(heights)

    def ids(self, start, end):
        if not self.views:
            raise ValueError("Closed native vocabulary")
        if not 0 <= start < end <= self.height:
            raise ValueError("Invalid native vocabulary rows")
        rows = end - start
        ids = np.empty(rows, np.int32)
        values = np.empty(rows, np.float32)
        loaded, _ = library()
        status = loaded.m1_fp16_argmax_chunks(
            self.pointers.ctypes.data,
            self.strides.ctypes.data,
            self.widths.ctypes.data,
            len(self.widths),
            start,
            rows,
            ids.ctypes.data,
            values.ctypes.data,
            _threads,
        )
        if status:
            raise ValueError("Unwritten/nonfinite native vocabulary logits")
        return ids.tolist()

    def close(self):
        for view in self.views:
            view.release()
        self.views = []
