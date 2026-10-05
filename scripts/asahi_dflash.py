"""Speculators-format Qwen3 DFlash on native M1 ANE + NumPy.

The linear projections and vocabulary head execute on ANE. Normalization,
RoPE, attention, residuals and SwiGLU execute on CPU. This is a Linux port of
the block-diffusion architecture, not CoreML graph execution. Only committed
target features enter the draft cache; speculative block K/V never do.
"""
from __future__ import annotations

import json
import mmap
import struct
import time
from pathlib import Path

import numpy as np

from asahi_ane import ANE


class Tensors:
    """Read BF16 safetensors without importing or executing model code."""
    def __init__(self, path):
        self.file = open(path, "rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        size = struct.unpack_from("<Q", self.map)[0]
        self.header = json.loads(self.map[8:8 + size])
        self.offset = 8 + size

    def read(self, name, dtype=np.float32, rows=None):
        meta = self.header[name]
        if meta["dtype"] != "BF16":
            raise ValueError(f"Expected BF16 weights: {name}")
        begin, end = meta["data_offsets"]
        shape = meta["shape"]
        if end - begin != int(np.prod(shape)) * 2 or self.offset + end > len(self.map):
            raise ValueError(f"Invalid tensor extent: {name}")
        bits = np.ndarray(shape, dtype="<u2", buffer=self.map, offset=self.offset + begin)
        if rows is not None:
            bits = bits[rows]
        return (bits.astype(np.uint32) << 16).view(np.float32).astype(dtype)

    def close(self):
        self.map.close()
        self.file.close()


def rms(x, weight, eps):
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * weight


def rope(x, positions, theta):
    half = x.shape[-1] // 2
    angles = np.asarray(positions, np.float32)[:, None] * (
        theta ** (-np.arange(half, dtype=np.float32) / half))[None]
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1)


class DFlash:
    def __init__(self, path, ane: ANE | None):
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text())
        cfg = self.config
        c = cfg["transformer_layer_config"]
        if cfg["speculators_config"]["algorithm"] != "dflash" or c["model_type"] != "qwen3":
            raise ValueError("This port supports Speculators DFlash with dense Qwen3 layers")
        if cfg.get("draft_vocab_size", c["vocab_size"]) != c["vocab_size"]:
            raise ValueError("Reduced-vocabulary draft mappings are not supported")
        if cfg.get("shift_label", False) or cfg.get("sample_from_anchor", False):
            raise ValueError("This port expects unshifted labels with one known anchor")
        self.block_size = cfg["block_size"]
        if not 2 <= self.block_size <= 32:
            raise ValueError("The ANE batch limit is 32")
        # These are HF/vLLM hidden-state IDs, i.e. 1 means after decoder layer 0.
        self.feature_ids = tuple(cfg["aux_hidden_state_layer_ids"])
        self.dim, self.heads = c["hidden_size"], c["num_attention_heads"]
        self.kv_heads, self.head_dim = c["num_key_value_heads"], c["head_dim"]
        self.eps = c["rms_norm_eps"]
        self.theta = c.get("rope_theta", c.get("rope_parameters", {}).get("rope_theta", 1e6))
        self.ane = ane
        self.weights = {}
        self.linears = {}
        self.window = c.get("sliding_window")
        self.layer_types = c.get("layer_types", ["full_attention"] * c["num_hidden_layers"])
        self.noncausal_window = cfg.get("sliding_window_non_causal", False)
        self.layers = c["num_hidden_layers"]
        tensors = Tensors(self.path / "model.safetensors")
        try:
            self.embedding = tensors.read("embed_tokens.weight", np.float16)
            for name in tensors.header:
                if name == "__metadata__" or name in {"embed_tokens.weight", "lm_head.weight"}:
                    continue
                if name.endswith("_proj.weight") or name == "fc.weight":
                    w = tensors.read(name, np.float16)
                    self.linears[name] = self._linear(w)
                else:
                    self.weights[name] = tensors.read(name)
            # A 151936-token head exceeds the register program's N limit.
            self.head_chunks = []
            for start in range(0, c["vocab_size"], 4096):
                w = tensors.read("lm_head.weight", np.float16, slice(start, start + 4096))
                self.head_chunks.append((start, self._linear(w)))
        finally:
            tensors.close()
        self.reset()

    def _linear(self, w):
        if self.ane is not None:
            return self.ane.linear(w)
        # Explicit FP16 input/output rounding reference for the ANE linears.
        def run(x):
            return (np.asarray(x, np.float16).astype(np.float32) @ w.astype(np.float32).T
                    ).astype(np.float16).astype(np.float32)
        return run

    def reset(self):
        self.cache = [None] * self.layers
        self.offset = 0
        self.profile = {k: 0.0 for k in ("context", "body", "head")}

    def norm(self, x, name):
        return rms(x, self.weights[name + ".weight"], self.eps)

    def linear(self, x, name):
        return self.linears[name + ".weight"](x)

    def append(self, features):
        """Append only target features already committed at absolute offset."""
        features = np.asarray(features, np.float32)
        if features.ndim != 2 or features.shape[1] != len(self.feature_ids) * self.dim:
            raise ValueError("Target features have the wrong shape")
        start = time.perf_counter()
        for begin in range(0, len(features), 32):
            f = features[begin:begin + 32]
            ctx = self.norm(self.linear(f, "fc"), "hidden_norm")
            positions = np.arange(self.offset, self.offset + len(f))
            for layer in range(self.layers):
                name = f"layers.{layer}.self_attn"
                k = self.linear(ctx, name + ".k_proj").reshape(-1, self.kv_heads, self.head_dim)
                k = rope(self.norm(k, name + ".k_norm"), positions, self.theta)
                v = self.linear(ctx, name + ".v_proj").reshape(-1, self.kv_heads, self.head_dim)
                previous = self.cache[layer]
                self.cache[layer] = (k, v, positions) if previous is None else tuple(
                    np.concatenate((old, new), axis=0) for old, new in zip(previous, (k, v, positions)))
                if self.layer_types[layer] == "sliding_attention":
                    # Training's window applies relative to the block anchor.
                    keep = self.cache[layer][2] >= self.offset + len(f) - self.window
                    self.cache[layer] = tuple(a[keep] for a in self.cache[layer])
            self.offset += len(f)
        self.profile["context"] += time.perf_counter() - start

    def propose(self, anchor):
        if not self.offset:
            raise ValueError("Draft needs committed target context")
        start = time.perf_counter()
        ids = [anchor] + [self.config["mask_token_id"]] * (self.block_size - 1)
        x = self.embedding[ids].astype(np.float32)
        positions = np.arange(self.offset, self.offset + self.block_size)
        repeat = self.heads // self.kv_heads
        for layer in range(self.layers):
            name = f"layers.{layer}"
            z = self.norm(x, name + ".input_layernorm")
            attn = name + ".self_attn"
            q = self.linear(z, attn + ".q_proj").reshape(-1, self.heads, self.head_dim)
            q = rope(self.norm(q, attn + ".q_norm"), positions, self.theta)
            k = self.linear(z, attn + ".k_proj").reshape(-1, self.kv_heads, self.head_dim)
            k = rope(self.norm(k, attn + ".k_norm"), positions, self.theta)
            v = self.linear(z, attn + ".v_proj").reshape(-1, self.kv_heads, self.head_dim)
            context_k, context_v, context_pos = self.cache[layer]
            k = np.concatenate((context_k, k), axis=0)
            v = np.concatenate((context_v, v), axis=0)
            k = np.repeat(k, repeat, axis=1).transpose(1, 0, 2)
            v = np.repeat(v, repeat, axis=1).transpose(1, 0, 2)
            scores = (q.transpose(1, 0, 2) @ k.transpose(0, 2, 1)) * self.head_dim**-0.5
            if self.layer_types[layer] == "sliding_attention":
                visible = np.concatenate((np.ones((self.block_size, len(context_pos)), bool),
                                          np.ones((self.block_size, self.block_size), bool)
                                          if self.noncausal_window else
                                          np.tri(self.block_size, dtype=bool)), axis=1)
                scores = np.where(visible[None], scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            out = (probabilities @ v).transpose(1, 0, 2).reshape(self.block_size, -1)
            x += self.linear(out, attn + ".o_proj")
            z = self.norm(x, name + ".post_attention_layernorm")
            gate = self.linear(z, name + ".mlp.gate_proj")
            up = self.linear(z, name + ".mlp.up_proj")
            # Stable sigmoid avoids overflow for negative gates.
            sigmoid = np.exp(-np.logaddexp(0, -gate))
            x += self.linear(gate * sigmoid * up, name + ".mlp.down_proj")
        hidden = self.norm(x, "norm")[1:]
        self.profile["body"] += time.perf_counter() - start
        start = time.perf_counter()
        best_values = np.full(len(hidden), -np.inf, np.float32)
        best_ids = np.zeros(len(hidden), np.int32)
        for base, linear in self.head_chunks:
            logits = linear(hidden)
            ids = np.argmax(logits, axis=-1)
            values = logits[np.arange(len(hidden)), ids]
            replace = values > best_values
            best_ids[replace] = ids[replace] + base
            best_values[replace] = values[replace]
        self.profile["head"] += time.perf_counter() - start
        return best_ids.tolist(), hidden
