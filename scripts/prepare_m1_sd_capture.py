"""Prepare matched real-input M1 ANE capture cases for execution on macOS.

MIL forms follow Orion's pinned builders. Host references use FP32 arithmetic
with FP16 projection boundaries; they are diagnostics, not Apple goldens.
No Apple compiler, weights, or generated ANE program is committed to Git.
"""
import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np

from asahi_ane import REPO
from asahi_dflash import Tensors, rms
from bench_asahi_mlx import TARGET, TARGET_REV, model_snapshot, sha256


def tensor(shape):
    return "tensor<fp16, [" + ",".join(map(str, shape)) + "]>"


def const(name, kind, value):
    return f'    {kind} {name} = const()[name=string("{name}"), val={kind}({value})];\n'


def op(name, shape, operation, args):
    return f'    {tensor(shape)} {name} = {operation}({args})[name=string("{name}")];\n'


def blob(directory, name, data):
    data = np.ascontiguousarray(data, dtype="<f2").tobytes()
    header = bytearray(128)
    struct.pack_into("<II", header, 0, 1, 2)
    struct.pack_into("<IIQQ", header, 64, 0xdeadbeef, 1, len(data), 128)
    relative = f"weights/{name}.bin"
    (directory / relative).write_bytes(header + data)
    return relative


def blob_const(name, shape):
    kind = tensor(shape)
    return const(name, kind, f'BLOBFILE(path=string("@model_path/weights/{name}.bin"), offset=uint64(64))')


def norm_mil(prefix, x, channels, seq, eps):
    shape, reduced = [1, channels, 1, seq], [1, 1, 1, seq]
    return (op(prefix + "_square", shape, "mul", f"x={x}, y={x}")
            + const(prefix + "_axis", "tensor<int32, [1]>", "[1]")
            + const(prefix + "_keep", "bool", "true")
            + op(prefix + "_sum", reduced, "reduce_sum", f"x={prefix}_square, axes={prefix}_axis, keep_dims={prefix}_keep")
            + const(prefix + "_inv", "fp16", repr(1 / channels))
            + op(prefix + "_mean", reduced, "mul", f"x={prefix}_sum, y={prefix}_inv")
            + const(prefix + "_eps", "fp16", repr(float(eps)))
            + op(prefix + "_plus", reduced, "add", f"x={prefix}_mean, y={prefix}_eps")
            + const(prefix + "_half", "fp16", "-0.5")
            + op(prefix + "_scale", reduced, "pow", f"x={prefix}_plus, y={prefix}_half")
            + op(prefix + "_normalized", shape, "mul", f"x={x}, y={prefix}_scale")
            + blob_const(prefix + "_weight", [1, channels, 1, 1])
            + op(prefix + "_out", shape, "mul", f"x={prefix}_normalized, y={prefix}_weight"))


def linear_mil(prefix, x, inputs, outputs, seq):
    return (const(prefix + "_padding", "string", '"valid"')
            + const(prefix + "_stride", "tensor<int32, [2]>", "[1,1]")
            + const(prefix + "_pad", "tensor<int32, [4]>", "[0,0,0,0]")
            + const(prefix + "_dilation", "tensor<int32, [2]>", "[1,1]")
            + const(prefix + "_groups", "int32", "1")
            + blob_const(prefix + "_weight", [outputs, inputs, 1, 1])
            + op(prefix + "_out", [1, outputs, 1, seq], "conv",
                 f"dilations={prefix}_dilation, groups={prefix}_groups, pad={prefix}_pad, "
                 f"pad_type={prefix}_padding, strides={prefix}_stride, weight={prefix}_weight, x={x}"))


def attention_mil(heads, dim, seq, context):
    qshape, kshape = [1, heads, dim, seq], [1, heads, dim, context]
    body = ""
    for name, shape in (("q", qshape), ("k", kshape), ("v", kshape)):
        body += const(name + "_shape", "tensor<int32, [4]>", str(shape))
        body += op(name + "4", shape, "reshape", f"x={name}, shape={name}_shape")
    perm = [0, 1, 3, 2]
    body += const("permutation", "tensor<int32, [4]>", str(perm))
    body += op("queries", [1, heads, seq, dim], "transpose", "x=q4, perm=permutation")
    body += op("values", [1, heads, context, dim], "transpose", "x=v4, perm=permutation")
    body += const("no_transpose", "bool", "false")
    body += op("dot", [1, heads, seq, context], "matmul", "x=queries, y=k4, transpose_x=no_transpose, transpose_y=no_transpose")
    body += const("scale", "fp16", repr(dim**-0.5))
    body += op("scaled", [1, heads, seq, context], "mul", "x=dot, y=scale")
    body += blob_const("mask", [1, 1, seq, context])
    body += op("masked", [1, heads, seq, context], "add", "x=scaled, y=mask")
    body += const("softmax_axis", "int32", "-1")
    body += op("probabilities", [1, heads, seq, context], "softmax", "x=masked, axis=softmax_axis")
    body += op("attended", [1, heads, seq, dim], "matmul", "x=probabilities, y=values, transpose_x=no_transpose, transpose_y=no_transpose")
    body += op("transposed", [1, heads, dim, seq], "transpose", "x=attended, perm=permutation")
    shape = [1, heads * dim, 1, seq]
    body += const("output_shape", "tensor<int32, [4]>", str(shape))
    return body + op("output", shape, "reshape", "x=transposed, shape=output_shape")


def nchw(rows, seq=32):
    out = np.zeros((1, rows.shape[1], 1, seq), np.float16)
    out[0, :, 0, :len(rows)] = rows.astype(np.float16).T
    return out


def reference_linear(x, weight):
    return (x.astype(np.float16).astype(np.float32) @ weight.astype(np.float32).T).astype(np.float16).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, default=REPO / "asahi/fixtures/m1-sd",
                    help="Pinned real inputs shipped in Git; override for freshly profiled inputs")
    ap.add_argument("--out", type=Path, default=REPO / ".asahi/m1-sd-capture")
    ap.add_argument("--include-target", action="store_true", help="Copy the pinned 1.2 GB target for a matched Metal probe")
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("--out must be absent or empty")
    source = json.loads((args.fixtures / "SOURCE.json").read_text())
    draft_path = REPO / ".asahi/models/microcycle-dflash"
    if sha256(draft_path / "model.safetensors") != source["fixture_source"]["weights_sha256"]:
        raise ValueError("Draft weight provenance differs")
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / "asahi/macos", args.out / "tools",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(REPO / "asahi/vendor/Orion", args.out / "Orion")
    config = json.loads((draft_path / "config.json").read_text())["transformer_layer_config"]
    channels, seq = config["hidden_size"], 32
    weights = Tensors(draft_path / "model.safetensors")
    manifest = {"schema": 1, "description": __doc__, "fixture_source": source["fixture_source"],
                "preparer_sha256": sha256(__file__),
                "reference_kind": "FP32 host diagnostic with FP16 projection boundaries; not Apple output",
                "target_architecture": "h13g", "cases": []}
    bench = json.loads((REPO / "notes/asahi/dflash_bf16_serial.json").read_text())
    manifest["target"] = bench["target"]
    manifest["teacher_traces"] = [{"name": row["name"], "prompt": row["prompt"],
                                   "tokens": row["baseline"]["tokens"]} for row in bench["rows"][:4]]
    if args.include_target:
        target_path, target_meta = model_snapshot(TARGET, TARGET_REV)
        if target_meta != manifest["target"]:
            raise ValueError("Pinned target differs")
        (args.out / "target").mkdir()
        for file in target_meta["files"]:
            shutil.copyfile(target_path / file, args.out / "target" / file)
    def case(label, inputs, body, output_name, expected, blobs):
        directory = args.out / "cases" / label
        (directory / "weights").mkdir(parents=True)
        declarations, bindings = [], []
        for name, data in inputs.items():
            data = np.ascontiguousarray(data, dtype="<f2")
            declarations.append(f"{tensor(data.shape)} {name}")
            (directory / (name + ".bin")).write_bytes(data.tobytes())
            bindings.append({"name": name, "shape": list(data.shape), "file": name + ".bin"})
        mil = "program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n  func main<ios18>(" + ", ".join(declarations) + ") {\n" + body + f"  }} -> ({output_name});\n}}\n"
        (directory / "model.mil").write_text(mil)
        blob_paths = [blob(directory, name, data) for name, data in blobs.items()]
        expected = np.ascontiguousarray(expected, dtype="<f2")
        (directory / "host-reference.bin").write_bytes(expected.tobytes())
        descriptor = {"label": label, "inputs": bindings, "blobs": blob_paths,
                      "outputs": [{"name": output_name, "shape": list(expected.shape), "reference": "host-reference.bin"}],
                      "reference_kind": manifest["reference_kind"]}
        (directory / "case.json").write_text(json.dumps(descriptor, indent=2) + "\n")
        manifest["cases"].append({"directory": "cases/" + label,
                                  "files": {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}})
    try:
        for name, meta in source["fixtures"].items():
            fixture_path = args.fixtures / meta["file"]
            if sha256(fixture_path) != meta["sha256"]:
                raise ValueError("Real-input fixture hash differs")
            with np.load(fixture_path, allow_pickle=False) as f:
                nweight = weights.read("layers.0.input_layernorm.weight")
                x = f["norm_input"].astype(np.float16).astype(np.float32)
                norm = rms(x, nweight.astype(np.float16).astype(np.float32), config["rms_norm_eps"])
                case(name + "-rmsnorm", {"x": nchw(x)}, norm_mil("norm", "x", channels, seq, config["rms_norm_eps"]),
                     "norm_out", nchw(norm), {"norm_weight": nweight})
                x = f["ffn_input"].astype(np.float16).astype(np.float32)
                nweight = weights.read("layers.0.post_attention_layernorm.weight")
                gate = weights.read("layers.0.mlp.gate_proj.weight", np.float16)
                up = weights.read("layers.0.mlp.up_proj.weight", np.float16)
                down = weights.read("layers.0.mlp.down_proj.weight", np.float16)
                intermediate = config["intermediate_size"]
                normalized = rms(x, nweight.astype(np.float16).astype(np.float32), config["rms_norm_eps"])
                g, u = reference_linear(normalized, gate), reference_linear(normalized, up)
                hidden = g * np.exp(-np.logaddexp(0, -g)) * u
                expected = x + reference_linear(hidden, down)
                body = (norm_mil("norm", "x", channels, seq, config["rms_norm_eps"])
                        + linear_mil("gate", "norm_out", channels, intermediate, seq)
                        + linear_mil("up", "norm_out", channels, intermediate, seq)
                        + op("sigmoid_gate", [1, intermediate, 1, seq], "sigmoid", "x=gate_out")
                        + op("silu_gate", [1, intermediate, 1, seq], "mul", "x=gate_out, y=sigmoid_gate")
                        + op("activated", [1, intermediate, 1, seq], "mul", "x=silu_gate, y=up_out")
                        + linear_mil("down", "activated", intermediate, channels, seq)
                        + op("residual", [1, channels, 1, seq], "add", "x=x, y=down_out"))
                case(name + "-fused-ffn", {"x": nchw(x)}, body, "residual", nchw(expected),
                     {"norm_weight": nweight, "gate_weight": gate, "up_weight": up, "down_weight": down})
                heads, dim, block = config["num_attention_heads"], config["head_dim"], len(f["norm_input"])
                context = ((f["k"].shape[-1] + 31) // 32) * 32
                inputs = {}
                for label in ("q", "k", "v"):
                    width = seq if label == "q" else context
                    data = np.zeros((heads, dim, width), np.float16)
                    data[..., :f[label].shape[-1]] = f[label]
                    inputs[label] = data.reshape(1, heads * dim, 1, width)
                committed = int(f["context_tokens"])
                mask = np.full((1, 1, seq, context), -1e4, np.float16)
                mask[..., :committed] = 0
                for i in range(block):
                    mask[0, 0, i, committed:committed + i + 1] = 0
                q = inputs["q"].reshape(heads, dim, seq).astype(np.float32)
                k = inputs["k"].reshape(heads, dim, context).astype(np.float32)
                v = inputs["v"].reshape(heads, dim, context).astype(np.float32)
                scores = (q.transpose(0, 2, 1) @ k) * dim**-0.5 + mask[0]
                probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
                probabilities /= probabilities.sum(axis=-1, keepdims=True)
                attended = (probabilities @ v.transpose(0, 2, 1)).transpose(0, 2, 1)
                case(name + "-cached-attention", inputs, attention_mil(heads, dim, seq, context), "output",
                     attended.reshape(1, heads * dim, 1, seq), {"mask": mask})
                for chunk in (4096, 8192):
                    head = weights.read("lm_head.weight", np.float16, slice(0, chunk))
                    x = f["head_input"]
                    case(f"{name}-head{chunk}", {"x": nchw(x)}, linear_mil("head", "x", channels, chunk, seq),
                         "head_out", nchw(reference_linear(x, head)), {"head_weight": head})
    finally:
        weights.close()
    manifest["tools"] = {str(p.relative_to(args.out)): sha256(p) for base in (args.out / "tools", args.out / "Orion")
                         for p in sorted(base.rglob("*")) if p.is_file()}
    if args.include_target:
        manifest["tools"].update({"target/" + name: digest for name, digest in manifest["target"]["files"].items()})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest['cases'])} real-input cases in {args.out}")


if __name__ == "__main__":
    main()
