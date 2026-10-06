"""Rebuild the measured H13G LUT6 coefficients from pinned BF16 safetensors.

The portable kit contains zeroed kernel templates and quantizer boundaries,
not learned matrix banks. Boundaries preserve the original compiler's cluster
assignment; palette means and packed six-bit indices come from the checkpoint.
This is a recipe for the captured model, not a general ANE compiler.
"""

from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from m1_common import require, BF16Tensors, sha256


class Checkpoint(BF16Tensors):
    """Hash-verified checkpoint decoded at the measured FP16 precision."""

    def __init__(self, directory, expected):
        path = Path(directory) / "model.safetensors"
        require(
            sha256(path) == expected,
            "Checkpoint differs from pinned HF download: " + str(path),
        )
        super().__init__(path)

    def read(self, name, rows=None):
        return super().read(
            name, np.float16, slice(*rows) if rows is not None else None
        )


def pack6(indices):
    a = np.asarray(indices, np.uint8).reshape(-1, 4).astype(np.uint32)
    require(bool((a < 64).all()), "Invalid LUT6 index")
    words = a[:, 0] | (a[:, 1] << 6) | (a[:, 2] << 12) | (a[:, 3] << 18)
    return (
        np.column_stack((words & 255, (words >> 8) & 255, words >> 16))
        .astype(np.uint8)
        .tobytes()
    )


def unpack6(raw, shape):
    a = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.uint32)
    words = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
    return (
        np.column_stack(tuple((words >> shift) & 63 for shift in (0, 6, 12, 18)))
        .astype(np.uint8)
        .reshape(shape)
    )


def palette_means(values, indices):
    ids = indices.reshape(-1)
    count = np.bincount(ids, minlength=64)
    require(bool((count > 0).all()), "Empty LUT cluster needs an explicit recipe")
    return (
        np.bincount(ids, weights=values.reshape(-1).astype(np.float64), minlength=64)
        / count
    ).astype("<f2")


def quantize(values, operation, boundaries):
    group_rows = operation["group_rows"]
    groups = 1 if group_rows == 0 else len(values) // group_rows
    cuts = np.frombuffer(
        boundaries, "<f2", count=groups * 63, offset=operation["boundary_offset"]
    ).reshape(groups, 63)
    indices = np.empty(values.shape, np.uint8)
    palettes = np.empty((groups, 64), "<f2")
    for group in range(groups):
        rows = (
            slice(None)
            if group_rows == 0
            else slice(group * group_rows, (group + 1) * group_rows)
        )
        require(
            bool(np.all(cuts[group, 1:] > cuts[group, :-1])),
            "Nonmonotone quantizer boundaries",
        )
        ids = np.searchsorted(cuts[group], values[rows], side="left").astype(np.uint8)
        indices[rows] = ids
        palettes[group] = palette_means(values[rows], ids)
    for group, index, bits in operation.get("palette_corrections", []):
        palettes.view("<u2")[group, index] = bits
    return indices, palettes


def matrix_bytes(indices, palettes, group_rows, tile=32):
    rows, columns = indices.shape
    require(rows % 16 == 0 and columns % 4 == 0, "Unsupported H13 matrix shape")
    output = bytearray()
    if group_rows:
        require(group_rows == 16, "Expected per-grouped-channel group size 16")
        for begin in range(0, rows, 256):
            width = min(16, (rows - begin) // 16)
            require(width in (8, 16), "Unsupported final H13 output tile")
            for engine in range(16):
                start = begin + engine * width
                output.extend(palettes[start // 16].tobytes())
                output.extend(pack6(indices[start : start + width].T))
    else:
        require(
            tile in (16, 32) and rows % (16 * tile) == 0,
            "Unsupported per-tensor schedule",
        )
        for engine in range(16):
            output.extend(palettes[0].tobytes())
            values = np.concatenate(
                [
                    indices[
                        base + engine * tile : base + (engine + 1) * tile
                    ].T.reshape(-1)
                    for base in range(0, rows, 16 * tile)
                ]
            )
            output.extend(pack6(values))
    return bytes(output)


def group_payloads(indices, palettes):
    return [
        palette.tobytes() + pack6(indices[group * 16 : (group + 1) * 16].T)
        for group, palette in enumerate(palettes)
    ]


def reconstruct(root, recipe, checkpoints):
    root = Path(root)
    template = root / recipe["template"]
    require(sha256(template) == recipe["template_sha256"], "Changed kernel template")
    data = bytearray(gzip.decompress(template.read_bytes()))
    require(len(data) == recipe["size"], "Wrong template size")
    boundary_path = root / recipe["boundaries"]
    require(
        sha256(boundary_path) == recipe["boundaries_sha256"],
        "Changed quantization recipe",
    )
    boundaries = gzip.decompress(boundary_path.read_bytes())
    for operation in recipe["operations"]:
        values = checkpoints[operation["checkpoint"]].read(
            operation["tensor"], operation.get("rows")
        )
        require(list(values.shape) == operation["shape"], "Wrong external tensor shape")
        if operation["kind"] == "lut6":
            indices, palettes = quantize(values, operation, boundaries)
            parts = (
                group_payloads(indices, palettes)
                if "group_offsets" in operation
                else None
            )
            payload = (
                b"".join(parts)
                if parts is not None
                else matrix_bytes(
                    indices,
                    palettes,
                    operation["group_rows"],
                    operation.get("tile", 32),
                )
            )
        else:
            require(operation["kind"] == "vector", "Unknown coefficient recipe")
            payload = values.tobytes()
        require(
            len(payload) == operation["size"]
            and hashlib.sha256(payload).hexdigest() == operation["sha256"],
            "Reconstructed coefficients differ: " + operation["tensor"],
        )
        if "group_offsets" in operation:
            extents = zip(operation["group_offsets"], parts)
        elif "engine_offsets" in operation:
            width = len(payload) // 16
            extents = [
                (begin, payload[index * width : (index + 1) * width])
                for index, begin in enumerate(operation["engine_offsets"])
            ]
        else:
            extents = [(operation["offset"], payload)]
        for begin, part in extents:
            end = begin + len(part)
            require(
                0 <= begin < end <= len(data) and not any(data[begin:end]),
                "Overlapping/nonzero learned template region",
            )
            data[begin:end] = part
    require(
        hashlib.sha256(data).hexdigest() == recipe["sha256"],
        "Reconstructed HWX differs from captured reference",
    )
    return data


def prepare(root, target, draft, cache):
    """Build large runtime assets outside Git, checking every full HWX hash."""
    import shutil

    root, cache = Path(root), Path(cache)
    stack = json.loads((root / "stack.json").read_text())
    checkpoints = {}
    try:
        for name, directory in [("target", target), ("draft", draft)]:
            checkpoints[name] = Checkpoint(directory, stack[name]["safetensors_sha256"])
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "stack.json", cache / "stack.json")
        if "positional_tables" in stack:
            name = stack["positional_tables"]["file"]
            shutil.copyfile(root / name, cache / name)
        for info in stack["artifacts"]:
            source = root / "programs" / info["program"]
            recipe = json.loads((source / "packing.json").read_text())
            destination = cache / "programs" / info["program"]
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("status.json", "linux-plan.json"):
                shutil.copyfile(source / name, destination / name)
            path = destination / "hwx/model.hwx"
            if path.is_file() and sha256(path) == recipe["sha256"]:
                continue
            print("Reconstructing", info["program"], flush=True)
            data = reconstruct(source, recipe, checkpoints)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".hwx-", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return cache
    finally:
        for checkpoint in checkpoints.values():
            checkpoint.close()
