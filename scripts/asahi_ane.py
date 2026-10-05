"""Resident FP16 ANE linears; no CoreML and no silent CPU fallback.

Uses the vendored qwen3.c base-M1 register program. Calls share device scratch
buffers, so a device is deliberately single-threaded. Acquire the ANE lock in
the caller for its lifetime.
"""
from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


class Profile(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in
                ("calls", "pack_ns", "upload_ns", "submit_ns", "download_ns", "unpack_ns")]


class ANE:
    def __init__(self, library=REPO / ".asahi/build/libane_sd.so"):
        self.lib = ct.CDLL(str(library))
        signatures = {
            "ane_device_open": ([], ct.c_void_p),
            "ane_device_close": ([ct.c_void_p], None),
            "ane_plan_create_fp16": ([ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int], ct.c_void_p),
            "ane_plan_free": ([ct.c_void_p], None),
            "ane_plan_run_batch": ([ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int], ct.c_int),
            "ane_device_submissions": ([ct.c_void_p], ct.c_ulonglong),
            "ane_device_profile": ([ct.c_void_p, ct.POINTER(Profile)], None),
            "ane_device_profile_reset": ([ct.c_void_p], None),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        self.device = self.lib.ane_device_open()
        self.plans = []
        if not self.device:
            raise RuntimeError("Cannot open base-M1 ANE device")

    @property
    def submissions(self):
        return self.lib.ane_device_submissions(self.device)

    @property
    def profile(self):
        result = Profile()
        self.lib.ane_device_profile(self.device, ct.byref(result))
        return {name: getattr(result, name) for name, _ in Profile._fields_}

    def reset_profile(self):
        self.lib.ane_device_profile_reset(self.device)

    def linear(self, weights):
        w = np.ascontiguousarray(weights, dtype=np.float16)
        if w.ndim != 2:
            raise ValueError("ANE weights must have shape [outputs, inputs]")
        plan = self.lib.ane_plan_create_fp16(self.device, w.ctypes.data, w.shape[1], w.shape[0])
        if not plan:
            raise RuntimeError(f"ANE could not prepare FP16 weights {w.shape}")
        self.plans.append(plan)

        # Do not retain another host copy of the resident matrix.
        shape = w.shape
        del w
        def apply(x):
            if not self.device:
                raise RuntimeError("ANE device is closed")
            x = np.ascontiguousarray(x, dtype=np.float32)
            if x.ndim != 2 or x.shape[1] != shape[1] or not 1 <= x.shape[0] <= 32:
                raise ValueError(f"Expected 1..32 rows with {shape[1]} inputs")
            out = np.empty((x.shape[0], shape[0]), dtype=np.float32)
            if not self.lib.ane_plan_run_batch(plan, x.ctypes.data, out.ctypes.data, x.shape[0]):
                raise RuntimeError("ANE submission failed or returned nonfinite/unwritten output")
            return out
        return apply

    def close(self):
        for plan in reversed(self.plans):
            self.lib.ane_plan_free(plan)
        self.plans.clear()
        if self.device:
            self.lib.ane_device_close(self.device)
            self.device = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
