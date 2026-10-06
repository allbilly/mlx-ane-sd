"""CoreML adapter for the shared M1 full-ANE host loop."""

from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import coremltools as ct
from m1_full_ane_host import FullANEStack as HostStack, generate as generate


class CoreMLProgram:
    def __init__(self, info, owner, role):
        self.owner, self.role = owner, role
        if not info["placement"]["math_all_ane"]:
            raise RuntimeError(
                f"{role}: refusing unverified ANE math placement: {info['placement']['math_counts']}"
            )
        self.model = ct.models.CompiledMLModel(
            info["compiled"], compute_units=ct.ComputeUnit.CPU_AND_NE
        )

    def predict(self, inputs, role=None):
        start = time.perf_counter()
        result = self.model.predict(inputs)
        label = role or self.role
        self.owner.profile[label] = (
            self.owner.profile.get(label, 0) + time.perf_counter() - start
        )
        self.owner.calls[label] = self.owner.calls.get(label, 0) + 1
        return result


class FullANEStack(HostStack):
    def __init__(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["status"] != "complete" or not manifest["math_all_ane"]:
            raise RuntimeError(
                "Need a completed manifest with verified ANE math placement"
            )
        embedding = np.load(directory / "embedding.npy", mmap_mode="r")

        def factory(info):
            role = (
                f"target{info['start']}" if info["role"] == "target" else info["role"]
            )
            return CoreMLProgram(info, self, role)

        super().__init__(directory, embedding, factory, manifest_name="manifest.json")
