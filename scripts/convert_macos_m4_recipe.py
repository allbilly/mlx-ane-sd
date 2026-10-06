"""Build Core ML artifacts for the small-model M4 full-ANE recipe on M1."""

from __future__ import annotations
import argparse
import collections
import gc
import hashlib
import platform
import subprocess
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.models.compute_plan import MLComputePlan
import m1_common as bench
from macos_m4_coreml_models import (
    TargetChunk,
    DraftProjector,
    DraftBody,
    VocabularyHead,
    cache_examples,
)
from m1_common import BF16Tensors as Tensors
from macos_m4_artifact_cache import load_cached, save_receipt
from m1_common import write_json


def placement(compiled):
    plan = MLComputePlan.load_from_path(
        str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    rows = []

    def visit(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            preferred = (
                type(usage.preferred_compute_device).__name__
                if usage is not None
                else "unassigned"
            )
            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            rows.append(
                {
                    "op": op.operator_name.split(".")[-1],
                    "outputs": [v.name for v in op.outputs],
                    "preferred": preferred,
                    "cost": cost.weight if cost is not None else None,
                }
            )
            for child in op.blocks:
                visit(child)

    for function in plan.model_structure.program.functions.values():
        visit(function.block)
    math = {"matmul", "linear", "conv", "softmax", "reduce_mean", "reduce_max", "rsqrt"}
    expensive = [r for r in rows if r["op"] in math]
    return {
        "counts": dict(collections.Counter(r["preferred"] for r in rows)),
        "math_counts": dict(collections.Counter(r["preferred"] for r in expensive)),
        "math_all_ane": bool(expensive)
        and all("NeuralEngine" in r["preferred"] for r in expensive),
        "ops": rows,
    }


def build(
    module, examples, input_names, output_names, folder, bits, granularity, provenance
):
    folder.mkdir(parents=True, exist_ok=True)
    pkg = folder / f"lut{bits}_{granularity}.mlpackage"
    compiled = pkg.with_suffix(".mlmodelc")
    state = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        state.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        state.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
    build_inputs = {
        "provenance": provenance,
        "module": str(module),
        "module_class": type(module).__name__,
        "state_sha256": state.hexdigest(),
        "inputs": [
            {"name": name, "shape": list(t.shape), "dtype": str(t.dtype)}
            for name, t in zip(input_names, examples)
        ],
        "outputs": list(output_names),
        "bits": bits,
        "granularity": granularity,
        "group_size": 16,
        "deployment": "macOS15",
        "precision": "FLOAT16",
        "compute_units": "CPU_AND_NE",
    }
    artifacts = {"package": pkg, "compiled": compiled}
    info = load_cached(folder / "receipt.json", build_inputs, artifacts)
    if info is not None:
        return info
    started = time.perf_counter()
    module = module.eval().to(torch.float16)
    print(f"[trace] {folder.name}", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(module, examples, strict=False)
    print(f"[convert] {folder.name}", flush=True)
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        inputs=[
            ct.TensorType(name=n, shape=t.shape, dtype=np.float16)
            for n, t in zip(input_names, examples)
        ],
        outputs=[ct.TensorType(name=n, dtype=np.float16) for n in output_names],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    if bits:
        print(f"[palettize] {folder.name}: LUT{bits} {granularity}", flush=True)
        model = cto.palettize_weights(
            model,
            cto.OptimizationConfig(
                global_config=cto.OpPalettizerConfig(
                    nbits=bits, mode="kmeans", granularity=granularity, group_size=16
                )
            ),
        )
    model.save(str(pkg))
    print(f"[compile] {folder.name}", flush=True)
    ct.models.utils.compile_model(str(pkg), destination_path=str(compiled))
    info = {
        "package": str(pkg),
        "compiled": str(compiled),
        "build_s": time.perf_counter() - started,
        "bits": bits,
        "granularity": granularity,
        "placement": placement(compiled),
    }
    # Successful prediction and real inputs are checked in the benchmark, separately from the plan.
    info = save_receipt(folder / "receipt.json", info, build_inputs, artifacts)
    print(f"[placement] {folder.name}: {info['placement']['math_counts']}", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target", type=Path, default=bench.REPO / ".asahi/models/m1-full-ane-target"
    )
    ap.add_argument(
        "--draft", type=Path, default=bench.REPO / ".asahi/models/m1-full-ane-draft"
    )
    ap.add_argument("--out", type=Path, default=bench.REPO / ".asahi/m4-recipe-m1")
    ap.add_argument("--capacity", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=14)
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--granularity", default="per_grouped_channel")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.capacity < 8 or args.chunk < 1:
        ap.error("Need capacity>=8 and chunk>=1")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.target / "config.json").read_text())
    draft_cfg = json.loads((args.draft / "config.json").read_text())
    dc = draft_cfg["transformer_layer_config"]
    result = {
        "target": str(args.target),
        "draft": str(args.draft),
        "capacity": args.capacity,
        "block": 8,
        "bits": args.bits,
        "granularity": args.granularity,
        "status": "waiting_for_devices",
        "config": cfg,
        "draft_config": draft_cfg,
        "artifacts": [],
        "source_sha256": {
            p: bench.sha256(bench.REPO / "scripts" / p)
            for p in (
                "convert_macos_m4_recipe.py",
                "macos_m4_coreml_models.py",
                "macos_m4_artifact_cache.py",
                "m1_common.py",
                "m1_lut6_weights.py",
            )
        },
    }
    manifest = args.out / ("probe.json" if args.probe else "manifest.json")
    progress = manifest.with_name(manifest.stem + ".building.json")

    def save():
        write_json(progress, result)

    torch.set_num_threads(4)
    with bench.device_locks():
        compiler = Path(
            subprocess.check_output(
                ["xcrun", "--find", "coremlcompiler"], text=True
            ).strip()
        )
        provenance = {
            "source_sha256": result["source_sha256"],
            "config": cfg,
            "draft_config": draft_cfg,
            "target_weights_sha256": bench.sha256(args.target / "model.safetensors"),
            "draft_weights_sha256": bench.sha256(args.draft / "model.safetensors"),
            "capacity": args.capacity,
            "block": 8,
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "coremltools": ct.__version__,
                "numpy": np.__version__,
                "machine": platform.machine(),
                "macos": platform.mac_ver()[0],
                "coremlcompiler_sha256": bench.sha256(compiler),
            },
        }
        result["build_provenance"] = provenance
        result["status"] = "converting"
        save()
        input_names = ["hidden", "cos", "sin", "cache_k", "cache_v", "mask"]
        if args.probe:
            m = TargetChunk(args.target, 0, 1)
            result["artifacts"].append(
                build(
                    m,
                    cache_examples(cfg, 1, args.capacity),
                    input_names,
                    ["hidden_out", "new_k", "new_v", "captures"],
                    args.out / "probe_target0_stable",
                    args.bits,
                    args.granularity,
                    provenance,
                )
            )
        else:
            embed = args.out / "embedding.npy"
            embed_inputs = {
                "kind": "embedding",
                "provenance": provenance,
                "dtype": "float16",
            }
            embedding_receipt = args.out / "embedding.receipt.json"
            if (
                load_cached(embedding_receipt, embed_inputs, {"embedding": embed})
                is None
            ):
                reader = Tensors(args.target / "model.safetensors")
                try:
                    np.save(embed, reader.read("model.embed_tokens.weight", np.float16))
                finally:
                    reader.close()
                save_receipt(embedding_receipt, {}, embed_inputs, {"embedding": embed})
            for start in range(0, cfg["num_hidden_layers"], args.chunk):
                count = min(args.chunk, cfg["num_hidden_layers"] - start)
                module = TargetChunk(args.target, start, count)
                outs = ["hidden_out", "new_k", "new_v"] + (
                    ["captures"] if module.capture_ids else []
                )
                info = build(
                    module,
                    cache_examples(cfg, count, args.capacity),
                    input_names,
                    outs,
                    args.out / f"target_{start}_{count}",
                    args.bits,
                    args.granularity,
                    provenance,
                )
                info.update(
                    role="target", start=start, count=count, captures=module.capture_ids
                )
                result["artifacts"].append(info)
                save()
                del module
                gc.collect()
            projector = DraftProjector(args.draft)
            examples = (
                torch.randn(1, 8, 3 * cfg["hidden_size"], dtype=torch.float16),
                torch.ones(1, 1, 8, cfg["head_dim"], dtype=torch.float16),
                torch.zeros(1, 1, 8, cfg["head_dim"], dtype=torch.float16),
            )
            info = build(
                projector,
                examples,
                ["features", "cos", "sin"],
                ["new_k", "new_v"],
                args.out / "draft_projector",
                args.bits,
                "per_tensor",
                provenance,
            )
            info["role"] = "projector"
            result["artifacts"].append(info)
            save()
            del projector
            gc.collect()
            body = DraftBody(args.draft)
            info = build(
                body,
                cache_examples(dc, dc["num_hidden_layers"], args.capacity),
                input_names,
                ["hidden_out"],
                args.out / "draft_body",
                args.bits,
                "per_tensor",
                provenance,
            )
            info["role"] = "draft"
            result["artifacts"].append(info)
            save()
            del body
            gc.collect()
            head = VocabularyHead(args.target)
            examples = (torch.randn(1, 8, cfg["hidden_size"], dtype=torch.float16),)
            info = build(
                head,
                examples,
                ["hidden"],
                [f"logits{i}" for i in range(len(head.tiles))],
                args.out / "head",
                args.bits,
                args.granularity,
                provenance,
            )
            info["role"] = "head"
            result["artifacts"].append(info)
            save()
            del head
            gc.collect()
        result["status"] = "complete"
        result["math_all_ane"] = all(
            a["placement"]["math_all_ane"] for a in result["artifacts"]
        )
        write_json(manifest, result)
        progress.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
