"""M4 full-ANE recipe adapted to the pinned public Qwen3-0.6B DFlash pair.

Same offload/compression method: cached Core ML target chunks, captured hidden
states, cached draft context, and a full-vocabulary LUT6 head. No downloads and
no changes to the original M4 or Asahi implementations.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from asahi_dflash import Tensors


class Norm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.from_numpy(weight.copy()))
        self.eps = eps

    def forward(self, x):
        # Bound squares/reductions on M1; the algebra is RMSNorm, including eps.
        # A floor on the scale also keeps all-zero padding finite.
        a = x.float()
        scale = a.abs().amax(dim=-1, keepdim=True).clamp(min=0.001)
        scaled = a / scale
        variance = scaled.square().mean(dim=-1, keepdim=True)
        # sqrt(eps) is representable in fp16; eps/scale**2 can be rewritten
        # through the unrepresentable constant 1/eps by the conversion passes.
        eps_scaled = (self.eps**0.5 / scale).square()
        normalized = scaled * torch.rsqrt(variance + eps_scaled)
        return normalized.to(x.dtype) * self.weight


def linear(weight):
    result = nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=torch.float16)
    result.weight = nn.Parameter(torch.from_numpy(weight.copy()))
    return result


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


class Block(nn.Module):
    def __init__(self, weights, prefix, config, block):
        super().__init__()
        self.B, self.W = block, config["hidden_size"]
        self.H, self.KV, self.D = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
        eps = config["rms_norm_eps"]
        def read(name): return weights.read(prefix + name, np.float16)
        self.input_ln = Norm(read("input_layernorm.weight"), eps)
        self.post_ln = Norm(read("post_attention_layernorm.weight"), eps)
        for role in ("q", "k", "v", "o"):
            setattr(self, role, linear(read("self_attn." + role + "_proj.weight")))
        self.q_norm = Norm(read("self_attn.q_norm.weight"), eps)
        self.k_norm = Norm(read("self_attn.k_norm.weight"), eps)
        for role in ("gate", "up", "down"):
            setattr(self, role, linear(read("mlp." + role + "_proj.weight")))

    def key_value(self, context, cos, sin):
        k = self.k(context).reshape(1, self.B, self.KV, self.D).transpose(1, 2)
        v = self.v(context).reshape(1, self.B, self.KV, self.D).transpose(1, 2)
        return rotate(self.k_norm(k), cos, sin), v

    def forward(self, x, cos, sin, cache_k, cache_v, mask):
        z = self.input_ln(x)
        q = self.q(z).reshape(1, self.B, self.H, self.D).transpose(1, 2)
        q = rotate(self.q_norm(q), cos, sin)
        new_k, new_v = self.key_value(z, cos, sin)
        k = torch.cat((cache_k, new_k), dim=2).repeat_interleave(self.H // self.KV, dim=1)
        v = torch.cat((cache_v, new_v), dim=2).repeat_interleave(self.H // self.KV, dim=1)
        scores = q @ k.transpose(-2, -1) * self.D**-0.5 + mask
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, self.B, self.H * self.D)
        x = x + self.o(attended)
        z = self.post_ln(x)
        x = x + self.down(F.silu(self.gate(z)) * self.up(z))
        return x, new_k, new_v


class TargetChunk(nn.Module):
    def __init__(self, path, start, count, block=8, captures=(1, 13, 25)):
        super().__init__()
        self.config = json.loads((Path(path) / "config.json").read_text())
        self.start, self.count, self.B = start, count, block
        self.capture_ids = [i for i in captures if start < i <= start + count]
        reader = Tensors(Path(path) / "model.safetensors")
        try:
            self.layers = nn.ModuleList([Block(reader, f"model.layers.{i}.", self.config, block)
                                         for i in range(start, start + count)])
            self.norm = Norm(reader.read("model.norm.weight", np.float16), self.config["rms_norm_eps"]) \
                if start + count == self.config["num_hidden_layers"] else None
        finally:
            reader.close()

    def forward(self, hidden, cos, sin, cache_k, cache_v, mask):
        keys, values, captures = [], [], []
        for local, layer in enumerate(self.layers):
            hidden, k, v = layer(hidden, cos, sin, cache_k[local], cache_v[local], mask)
            keys.append(k); values.append(v)
            if self.start + local + 1 in self.capture_ids: captures.append(hidden)
        result = (self.norm(hidden) if self.norm is not None else hidden,
                  torch.stack(keys), torch.stack(values))
        return result + (torch.stack(captures),) if captures else result


class DraftProjector(nn.Module):
    def __init__(self, path, block=8):
        super().__init__()
        cfg = json.loads((Path(path) / "config.json").read_text())
        c = cfg["transformer_layer_config"]
        self.B, self.KV, self.D = block, c["num_key_value_heads"], c["head_dim"]
        reader = Tensors(Path(path) / "model.safetensors")
        try:
            self.fc = linear(reader.read("fc.weight", np.float16))
            self.norm = Norm(reader.read("hidden_norm.weight", np.float16), c["rms_norm_eps"])
            self.k = nn.ModuleList([linear(reader.read(f"layers.{i}.self_attn.k_proj.weight", np.float16))
                                    for i in range(c["num_hidden_layers"])])
            self.v = nn.ModuleList([linear(reader.read(f"layers.{i}.self_attn.v_proj.weight", np.float16))
                                    for i in range(c["num_hidden_layers"])])
            self.k_norm = nn.ModuleList([Norm(reader.read(f"layers.{i}.self_attn.k_norm.weight", np.float16), c["rms_norm_eps"])
                                         for i in range(c["num_hidden_layers"])])
        finally: reader.close()

    def forward(self, features, cos, sin):
        context = self.norm(self.fc(features))
        keys, values = [], []
        for k, v, norm in zip(self.k, self.v, self.k_norm):
            projected = k(context).reshape(1, self.B, self.KV, self.D).transpose(1, 2)
            keys.append(rotate(norm(projected), cos, sin))
            values.append(v(context).reshape(1, self.B, self.KV, self.D).transpose(1, 2))
        return torch.stack(keys), torch.stack(values)


class DraftBody(nn.Module):
    def __init__(self, path, block=8):
        super().__init__()
        cfg = json.loads((Path(path) / "config.json").read_text())
        c = cfg["transformer_layer_config"]
        reader = Tensors(Path(path) / "model.safetensors")
        try:
            self.layers = nn.ModuleList([Block(reader, f"layers.{i}.", c, block)
                                         for i in range(c["num_hidden_layers"])])
            self.norm = Norm(reader.read("norm.weight", np.float16), c["rms_norm_eps"])
        finally: reader.close()

    def forward(self, hidden, cos, sin, cache_k, cache_v, mask):
        for i, layer in enumerate(self.layers):
            hidden, _, _ = layer(hidden, cos, sin, cache_k[i], cache_v[i], mask)
        return self.norm(hidden)


class VocabularyHead(nn.Module):
    def __init__(self, path, tile=8192):
        super().__init__()
        reader = Tensors(Path(path) / "model.safetensors")
        try:
            # The public target/draft embeddings and frozen heads are identical.
            weight = reader.read("model.embed_tokens.weight", np.float16)
        finally: reader.close()
        self.tiles = nn.ModuleList([linear(weight[i:i+tile]) for i in range(0, len(weight), tile)])

    def forward(self, hidden):
        return tuple(layer(hidden) for layer in self.tiles)


def cache_examples(config, count, capacity, block=8):
    H, KV, D = config["hidden_size"], config["num_key_value_heads"], config["head_dim"]
    return (torch.randn(1, block, H, dtype=torch.float16) * .1,
            torch.ones(1, 1, block, D, dtype=torch.float16),
            torch.zeros(1, 1, block, D, dtype=torch.float16),
            torch.zeros(count, 1, KV, capacity, D, dtype=torch.float16),
            torch.zeros(count, 1, KV, capacity, D, dtype=torch.float16),
            torch.zeros(1, 1, block, capacity + block, dtype=torch.float16))
