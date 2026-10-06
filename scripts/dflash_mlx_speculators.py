"""Native MLX inference for a publicly released Speculators Qwen3 DFlash.

Implements the checkpoint's causal sliding-window block mask and HF hidden
state numbering. Context K/V comes only from committed target features; the
noise block never enters this cache. Architecture reference:
https://github.com/vllm-project/speculators/tree/main/src/speculators/models/dflash
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import ModelArgs, TransformerBlock


class DraftNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = dict(config["transformer_layer_config"])
        c["rope_theta"] = c.get("rope_theta", c.get("rope_parameters", {}).get("rope_theta", 1e6))
        args = ModelArgs.from_dict(c)
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.fc = nn.Linear(len(config["aux_hidden_state_layer_ids"]) * args.hidden_size,
                            args.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)


class DFlash:
    def __init__(self, path: Path, proposal_count=7, precision="bf16", shared_embedding=None):
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text())
        cfg = self.config
        c = cfg["transformer_layer_config"]
        if cfg["speculators_config"]["algorithm"] != "dflash" or c["model_type"] != "qwen3":
            raise ValueError("Expected a dense Qwen3 Speculators DFlash checkpoint")
        if cfg.get("projector_type", "dflash") != "dflash" or cfg.get("pure_draft_prefix_len", 1) != 1:
            raise ValueError("Expected the standard DFlash projector and one known anchor")
        if cfg.get("shift_label", False) or cfg.get("sample_from_anchor", False):
            raise ValueError("Only unshifted labels with one anchor are supported")
        if cfg.get("draft_vocab_size", c["vocab_size"]) != c["vocab_size"]:
            raise ValueError("Reduced vocabulary mappings are not supported")
        if not 1 <= proposal_count < cfg["block_size"]:
            raise ValueError("Proposal count must be within the trained block size")
        self.feature_ids = tuple(cfg["aux_hidden_state_layer_ids"])
        self.proposal_count = proposal_count
        self.window = c.get("sliding_window")
        self.layer_types = c.get("layer_types", ["full_attention"] * c["num_hidden_layers"])
        self.noncausal = cfg.get("sliding_window_non_causal", False)
        # Causal intra-block attention allows dropping unused later mask tokens.
        self.block_size = proposal_count + 1 if not self.noncausal and all(
            t == "sliding_attention" for t in self.layer_types) else cfg["block_size"]
        self.network = DraftNetwork(cfg)
        weights = mx.load(str(self.path / "model.safetensors"))
        self.shared_weights = False
        if shared_embedding is not None:
            # Validate on the loaded checkpoint, rather than assuming frozen heads.
            identical = mx.array_equal(weights["embed_tokens.weight"], shared_embedding) & mx.array_equal(
                weights["lm_head.weight"], shared_embedding)
            if identical.item():
                weights["embed_tokens.weight"] = weights["lm_head.weight"] = shared_embedding
                self.shared_weights = True
        self.network.load_weights(list(weights.items()), strict=True)
        self.network.set_dtype({"bf16": mx.bfloat16, "fp16": mx.float16}[precision])
        if self.shared_weights:
            self.network.lm_head.weight = self.network.embed_tokens.weight
        self.network.eval()
        mx.eval(self.network.parameters())
        self.reset()

    def reset(self):
        self.cache = [None] * len(self.network.layers)
        self.offset = 0
        self.profile = {k: 0.0 for k in ("context", "body", "head")}

    def append(self, features):
        if features.ndim != 2 or features.shape[1] != len(self.feature_ids) * self.network.fc.weight.shape[0]:
            raise ValueError("Wrong target feature shape")
        start = time.perf_counter()
        features = features.astype(self.network.fc.weight.dtype)[None]
        context = self.network.hidden_norm(self.network.fc(features))
        length = context.shape[1]
        for i, layer in enumerate(self.network.layers):
            a = layer.self_attn
            k = a.k_norm(a.k_proj(context).reshape(1, length, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            k = a.rope(k, offset=self.offset)
            v = a.v_proj(context).reshape(1, length, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            if self.cache[i] is not None:
                k = mx.concatenate((self.cache[i][0], k), axis=2)
                v = mx.concatenate((self.cache[i][1], v), axis=2)
            if self.layer_types[i] == "sliding_attention" and self.window:
                k, v = k[:, :, -self.window:], v[:, :, -self.window:]
            self.cache[i] = (k, v)
        self.offset += length
        mx.eval(self.cache)
        self.profile["context"] += time.perf_counter() - start

    def propose(self, anchor):
        if not self.offset:
            raise ValueError("Draft needs committed target context")
        start = time.perf_counter()
        n = self.network
        ids = mx.array([[anchor] + [self.config["mask_token_id"]] * (self.block_size - 1)], mx.uint32)
        x = n.embed_tokens(ids)
        length = x.shape[1]
        for i, layer in enumerate(n.layers):
            a = layer.self_attn
            z = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(z).reshape(1, length, a.n_heads, -1)).transpose(0, 2, 1, 3)
            k = a.k_norm(a.k_proj(z).reshape(1, length, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = a.v_proj(z).reshape(1, length, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q, k = a.rope(q, offset=self.offset), a.rope(k, offset=self.offset)
            k = mx.concatenate((self.cache[i][0], k), axis=2)
            v = mx.concatenate((self.cache[i][1], v), axis=2)
            mask = "causal" if self.layer_types[i] == "sliding_attention" and not self.noncausal else None
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=mask)
            x = x + a.o_proj(out.transpose(0, 2, 1, 3).reshape(1, length, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        hidden = n.norm(x)[:, 1:self.proposal_count + 1]
        mx.eval(hidden)
        self.profile["body"] += time.perf_counter() - start
        start = time.perf_counter()
        ids = mx.argmax(n.lm_head(hidden), axis=-1)[0]
        mx.eval(ids)
        self.profile["head"] += time.perf_counter() - start
        return ids.tolist(), hidden[0]
