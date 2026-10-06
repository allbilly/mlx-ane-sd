"""Public Speculators DFlash through ANEForge's ANE-only macOS runtime.

The fused body includes the projector, all three Qwen3 layers and final norm.
It recomputes context K/V from committed features, so speculative tokens never
enter draft state. Fixed capacity is explicit; exceeding it is an error.
MLX is used only for the GPU target and optionally the draft vocabulary head.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from asahi_dflash import Tensors

REPO = Path(__file__).resolve().parent.parent


def setup_aneforge(reference):
    reference = Path(reference).resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference))
    os.environ["ANEFORGE_NO_AUTOBUILD"] = "1"
    os.environ["ANEFORGE_CACHE_DIR"] = str(REPO / ".asahi/macos-ane-sd/cache")
    import aneforge as af
    from aneforge._runtime import _find_dylib
    library = _find_dylib()
    return af, {"path": str(reference), "commit": subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip(),
        "runtime_library": str(library), "runtime_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}


class DFlashANE:
    def __init__(self, path, reference, capacity=256, head="gpu"):
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text())
        cfg = self.config
        c = cfg["transformer_layer_config"]
        if (cfg["speculators_config"]["algorithm"] != "dflash" or c["model_type"] != "qwen3"
                or cfg["speculators_config"]["verifier"]["name_or_path"] != "Qwen/Qwen3-0.6B"
                or cfg.get("sliding_window_non_causal", False)
                or cfg.get("shift_label", False) or cfg.get("sample_from_anchor", False)
                or cfg.get("pure_draft_prefix_len", 1) != 1):
            raise ValueError("Expected the pinned causal Qwen3-0.6B DFlash")
        self.feature_ids = tuple(cfg["aux_hidden_state_layer_ids"])
        self.capacity, self.head_mode = capacity, head
        self.dim, self.block_size = c["hidden_size"], cfg["block_size"]
        self.physical_block = 32
        self.h, self.kvh, self.dh = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.eps = c["rms_norm_eps"]
        self.theta = c.get("rope_theta", c.get("rope_parameters", {}).get("rope_theta", 1e6))
        if capacity > c["sliding_window"] or head not in ("gpu", "ane"):
            raise ValueError("Invalid capacity or head backend")
        self.af, self.provenance = setup_aneforge(reference)
        weights = Tensors(self.path / "model.safetensors")
        try:
            self.embedding = weights.read("embed_tokens.weight", np.float16)
            body = {name: weights.read(name, np.float16) for name in weights.header
                    if name not in ("__metadata__", "embed_tokens.weight", "lm_head.weight")}
            head_weight = weights.read("lm_head.weight", np.float16)
        finally:
            weights.close()
        self._build_body(body)
        self.head_tiles = []
        if head == "ane":
            # M1's per-axis limit is 16384; use the already-captured 8192 width.
            self._build_head(head_weight)
            self.gpu_weight = None
        else:
            import mlx.core as mx
            self.gpu_weight = mx.array(head_weight)
            mx.eval(self.gpu_weight)
        self.reset()

    def _build_body(self, weights):
        af = self.af
        from aneforge.llm import rope, _repeat_kv
        self.body_inputs = {}
        def inp(name, shape):
            node = af.input(shape)
            self.body_inputs[name] = node
            return node
        C, B, H, KV, D, W = self.capacity, self.physical_block, self.h, self.kvh, self.dh, self.dim
        features = inp("features", (C, len(self.feature_ids) * W))
        x = inp("noise", (B, W))
        cc, sc = inp("cos_context", (1, 1, C, D)), inp("sin_context", (1, 1, C, D))
        cq, sq = inp("cos_noise", (1, 1, B, D)), inp("sin_noise", (1, 1, B, D))
        mask = inp("mask", (1, 1, B, C + B))
        def norm(values, name):
            # M1 reduce_l2_norm overflows on real DFlash residuals (up to ~8000),
            # turning x / inf into an all-zero draft. Power-of-two rescaling
            # preserves RMSNorm and bounds the reduction. The initial embedding
            # is small, so retain its scale to avoid fp16 underflow.
            scale = 1.0 if name == "layers.0.input_layernorm.weight" else (
                1 / 1024 if values.shape[-1] == W else 1 / 16)
            return (values * scale).rms_norm(weights[name], self.eps * scale * scale)
        context = norm(features.linear(weights["fc.weight"]), "hidden_norm.weight")
        def project(values, name, rows, heads, normalize=False, cos=None, sin=None):
            projected = values.linear(weights[name + "_proj.weight"]).reshape(rows, heads, D)
            if normalize:
                projected = norm(projected.reshape(rows * heads, D), name + "_norm.weight").reshape(rows, heads, D)
            projected = projected.transpose([1, 0, 2]).reshape(1, heads, rows, D)
            return rope(projected, cos, sin) if cos is not None else projected
        for i in range(len(self.config["transformer_layer_config"]["layer_types"])):
            p = f"layers.{i}."
            z = norm(x, p + "input_layernorm.weight")
            a = p + "self_attn."
            q = project(z, a + "q", B, H, True, cq, sq)
            k = af.concat([project(context, a + "k", C, KV, True, cc, sc),
                           project(z, a + "k", B, KV, True, cq, sq)], axis=2)
            v = af.concat([project(context, a + "v", C, KV), project(z, a + "v", B, KV)], axis=2)
            k, v = _repeat_kv(k, H // KV), _repeat_kv(v, H // KV)
            scores = ((q @ k.transpose([0, 1, 3, 2])) * D**-0.5 + mask).softmax(-1)
            attended = (scores @ v).reshape(H, B, D).transpose([1, 0, 2]).reshape(B, H * D)
            x = x + attended.linear(weights[a + "o_proj.weight"])
            z = norm(x, p + "post_attention_layernorm.weight")
            x = x + (z.linear(weights[p + "mlp.gate_proj.weight"]).silu()
                     * z.linear(weights[p + "mlp.up_proj.weight"])).linear(weights[p + "mlp.down_proj.weight"])
        out = norm(x, "norm.weight")
        self.graph_out = out
        build = REPO / f".asahi/macos-ane-sd/body-c{C}"
        print("[ANE compile] complete three-layer draft body", flush=True)
        self.body = af.compile(out, build_dir=build, opt=0)
        if self.body._prog._device_mask != 4:
            raise RuntimeError("Draft body is not restricted to ANE")
        self.body_ports = dict(zip(self.body_inputs, (name for name, _ in self.body._inputs)))
        self.body_out = self.body._out_name

    def _build_head(self, weight):
        from aneforge._compile import compile_multi
        B = self.physical_block
        source = self.af.input((B, self.dim))
        outs = [source.linear(weight[start:start + 8192]) for start in range(0, len(weight), 8192)]
        print("[ANE compile] tiled full-vocabulary draft head", flush=True)
        self.head = compile_multi(outs, build_dir=REPO / ".asahi/macos-ane-sd/head8192")
        if self.head.prog._device_mask != 4:
            raise RuntimeError("Draft head is not restricted to ANE")
        self.head_input = self.head.input_ports[0][1]
        self.head_tiles = [(i * 8192, name) for i, (_, name) in enumerate(self.head.output_ports)]

    def reset(self):
        self.features = np.zeros((self.capacity, len(self.feature_ids) * self.dim), np.float16)
        self.offset = 0
        self.ane_evaluations = 0
        self.profile = {key: 0.0 for key in ("context", "body", "head")}

    def append(self, features):
        import mlx.core as mx
        start = time.perf_counter()
        if isinstance(features, mx.array):
            features = np.array(features.astype(mx.float16))
        features = np.asarray(features, np.float16)
        if features.ndim != 2 or features.shape[1] != self.features.shape[1]:
            raise ValueError("Wrong committed feature shape")
        end = self.offset + len(features)
        if end > self.capacity:
            raise ValueError("Committed context exceeds fixed ANE capacity; increase --capacity")
        self.features[self.offset:end] = features
        self.offset = end
        self.profile["context"] += time.perf_counter() - start

    def _rope(self, positions):
        half = self.dh // 2
        angles = positions[:, None] * self.theta**(-np.arange(half, dtype=np.float32) / half)
        angles = np.concatenate((angles, angles), axis=-1)
        return np.cos(angles).astype(np.float16)[None, None], np.sin(angles).astype(np.float16)[None, None]

    def propose(self, anchor):
        start = time.perf_counter()
        B, C, BS = self.physical_block, self.capacity, self.block_size
        noise = np.zeros((B, self.dim), np.float16)
        noise[:BS] = self.embedding[[anchor] + [self.config["mask_token_id"]] * (BS - 1)]
        cc, sc = self._rope(np.arange(C))
        cq, sq = self._rope(np.arange(self.offset, self.offset + B))
        mask = np.full((1, 1, B, C + B), -10000, np.float16)
        mask[:, :, :, :self.offset] = 0
        mask[:, :, :BS, C:C + BS] = np.where(np.tri(BS, dtype=bool), 0, -10000)
        feed = {"features": self.features, "noise": noise, "cos_context": cc, "sin_context": sc,
                "cos_noise": cq, "sin_noise": sq, "mask": mask}
        self.last_feed = feed
        hidden = self.body._prog.eval({self.body_ports[k]: v for k, v in feed.items()})[self.body_out].copy()
        self.ane_evaluations += 1
        if not np.isfinite(hidden[:BS]).all():
            raise RuntimeError("ANE draft returned non-finite hidden states")
        if not np.any(hidden[1:BS]):
            raise RuntimeError("ANE draft returned all-zero hidden states")
        self.profile["body"] += time.perf_counter() - start
        start = time.perf_counter()
        if self.head_mode == "ane":
            logits = self.head.prog.eval({self.head_input: hidden})
            self.ane_evaluations += 1
            ids = np.zeros(BS - 1, np.int32)
            values = np.full(BS - 1, -np.inf, np.float32)
            for base, port in self.head_tiles:
                scores = logits[port][1:BS]
                local = scores.argmax(-1)
                maxima = scores[np.arange(BS - 1), local]
                better = maxima > values
                ids[better], values[better] = local[better] + base, maxima[better]
            proposals = ids.tolist()
        else:
            import mlx.core as mx
            proposals = mx.argmax(mx.array(hidden[1:BS]) @ self.gpu_weight.T, axis=-1).tolist()
        self.profile["head"] += time.perf_counter() - start
        return proposals, hidden[1:BS]

    def close(self):
        self.body.release()
        if self.head_mode == "ane":
            self.head.release()
