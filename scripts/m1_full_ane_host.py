"""Portable host loop for the exact M1 full-ANE stack.

The macOS CoreML and Linux DRM adapters share cache, acceptance and EOS logic.
A program factory supplies execution; no CoreML or Metal dependency here.
"""

from __future__ import annotations
import json
import hashlib
import time
from pathlib import Path
import numpy as np


def rope(offset, block, dim, theta):
    inv = theta ** (-np.arange(dim // 2, dtype=np.float32) / (dim // 2))
    angles = np.arange(offset, offset + block, dtype=np.float32)[:, None] * inv
    angles = np.concatenate((angles, angles), axis=-1)
    return (
        np.cos(angles).astype(np.float16)[None, None],
        np.sin(angles).astype(np.float16)[None, None],
    )


def causal_mask(offset, capacity, block):
    result = np.full((1, 1, block, capacity + block), -10000, np.float16)
    result[:, :, :, :offset] = 0
    result[:, :, :, capacity:] = np.where(np.tri(block, dtype=bool), 0, -10000)
    return result


def cache_prefix(value, count):
    read = getattr(value, "read_prefix", None)
    return (
        read(count)
        if read is not None
        else np.asarray(value, np.float16)[:, :, :, :count]
    )


def trace_matches(actual, expected):
    """Unused predictions may be omitted; decisions and cache state must match."""
    if len(actual) != len(expected):
        return False
    for row, golden in zip(actual, expected):
        if any(
            row[key] != golden[key]
            for key in ("candidates", "accepted", "committed", "offset")
        ):
            return False
        if len(row["predictions"]) < row["accepted"] + 1:
            return False
        if row["predictions"] != golden["predictions"][: len(row["predictions"])]:
            return False
    return True


class FullANEStack:
    def __init__(self, directory, embedding, factory, manifest_name="stack.json"):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / manifest_name).read_text())
        m = self.manifest
        self.cfg, self.dc = m["config"], m["draft_config"]["transformer_layer_config"]
        self.B, self.C = m["block"], m["capacity"]
        self.D, self.KV = self.cfg["head_dim"], self.cfg["num_key_value_heads"]
        self.W = self.cfg["hidden_size"]
        self.feature_ids = tuple(m["draft_config"]["aux_hidden_state_layer_ids"])
        self.embedding = embedding
        positional = m.get("positional_tables")
        self.positional = None
        if positional:
            path = self.directory / positional["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != positional["sha256"]:
                raise ValueError("Changed fixed RoPE table")
            with np.load(path, allow_pickle=False) as table:
                self.positional = (table["cos"].copy(), table["sin"].copy())
            if any(
                values.shape != (self.C + self.B, self.D // 2)
                or values.dtype != np.float16
                or not np.isfinite(values).all()
                for values in self.positional
            ):
                raise ValueError("Wrong fixed RoPE table")
        self.profile = {}
        self.calls = {}
        self.chunks = []
        if embedding.shape != (self.cfg["vocab_size"], self.W):
            raise ValueError("Wrong embedding shape")
        for info in m["artifacts"]:
            program = factory(info)
            role = info["role"]
            if role == "target":
                self.chunks.append((info, program))
            elif role == "head":
                self.head = program
            elif role == "projector":
                self.projector = program
            elif role == "draft":
                self.draft = program
        self.reset()

    def positions(self, offset):
        if self.positional is None:
            return rope(offset, self.B, self.D, self.cfg["rope_theta"])
        return tuple(
            np.concatenate(
                (values[offset : offset + self.B], values[offset : offset + self.B]),
                axis=-1,
            )[None, None]
            for values in self.positional
        )

    def reset(self):
        self.offset = self.draft_offset = 0
        self.cache_k = np.zeros(
            (self.cfg["num_hidden_layers"], 1, self.KV, self.C, self.D), np.float16
        )
        self.cache_v = np.zeros_like(self.cache_k)
        self.draft_k = np.zeros(
            (self.dc["num_hidden_layers"], 1, self.KV, self.C, self.D), np.float16
        )
        self.draft_v = np.zeros_like(self.draft_k)
        self.profile = {}
        self.calls = {}
        for program in [model for _, model in self.chunks] + [self.draft]:
            reset = getattr(program, "reset_transport", None)
            if reset is not None:
                reset()

    def ids(self, hidden, start=0, end=None, role="target_head", candidates=None):
        end = self.B if end is None else end
        verify = getattr(self.head, "predict_verified_ids", None)
        if candidates is not None and verify is not None:
            return verify({"hidden": hidden}, candidates, start, end, role)
        read = getattr(self.head, "predict_token_ids", None)
        if read is not None:
            return read({"hidden": hidden}, start, end, role)
        outputs = self.head.predict({"hidden": hidden}, role)
        begin = time.perf_counter()
        rows = end - start
        best = np.full(rows, -np.inf, np.float32)
        ids = np.zeros(rows, np.int32)
        for i in range((self.cfg["vocab_size"] + 8191) // 8192):
            # Only reduce rows actually requested. fp32 NumPy argmax is faster
            # than its fp16 path, and converting preserves every fp16 value.
            logits = np.asarray(outputs[f"logits{i}"])[0, start:end].astype(np.float32)
            index = logits.argmax(-1)
            values = logits[np.arange(rows), index]
            better = values > best
            ids[better] = index[better] + i * 8192
            best[better] = values[better]
        if not np.isfinite(best).all():
            raise RuntimeError("Invalid ANE vocabulary logits")
        self.profile["host_argmax"] = (
            self.profile.get("host_argmax", 0) + time.perf_counter() - begin
        )
        return ids.tolist()

    def forward(self, token_ids, logits=True, candidates=None):
        n = len(token_ids)
        if not 1 <= n <= self.B or self.offset + n > self.C:
            raise ValueError("Invalid target block/context length")
        hidden = np.zeros((1, self.B, self.W), np.float16)
        hidden[0, :n] = self.embedding[token_ids]
        cos, sin = self.positions(self.offset)
        mask = causal_mask(self.offset, self.C, self.B)
        pending = []
        captures = {}
        for info, model in self.chunks:
            start, count = info["start"], info["count"]
            out = model.predict(
                {
                    "hidden": hidden,
                    "cos": cos,
                    "sin": sin,
                    "mask": mask,
                    "cache_k": self.cache_k[start : start + count],
                    "cache_v": self.cache_v[start : start + count],
                }
            )
            hidden = np.asarray(out["hidden_out"], np.float16)
            if not np.isfinite(hidden[0, :n]).all():
                raise RuntimeError("Non-finite target hidden state")
            pending.append((start, count, out["new_k"], out["new_v"]))
            if info["captures"]:
                for local, index in enumerate(info["captures"]):
                    captures[index] = np.asarray(out["captures"])[local, 0, :n]
        features = np.concatenate(
            [captures[i] for i in self.feature_ids], axis=-1
        ).astype(np.float16)
        self.last_hidden = hidden
        predictions = self.ids(hidden, end=n, candidates=candidates) if logits else None
        return predictions, features, pending

    def commit(self, pending, count):
        for start, layers, k, v in pending:
            k, v = cache_prefix(k, count), cache_prefix(v, count)
            self.cache_k[
                start : start + layers, :, :, self.offset : self.offset + count
            ] = k
            self.cache_v[
                start : start + layers, :, :, self.offset : self.offset + count
            ] = v
            program = next(
                model for info, model in self.chunks if info["start"] == start
            )
            update = getattr(program, "commit_cache", None)
            if update is not None:
                update(k, v, self.offset, count)
        self.offset += count

    def append_draft(self, features):
        for begin in range(0, len(features), self.B):
            active = features[begin : begin + self.B]
            n = len(active)
            if self.draft_offset + n > self.C:
                raise ValueError("Draft cache capacity exceeded")
            padded = np.zeros((1, self.B, len(self.feature_ids) * self.W), np.float16)
            padded[0, :n] = active
            cos, sin = self.positions(self.draft_offset)
            outputs = self.projector.predict(
                {"features": padded, "cos": cos, "sin": sin}
            )
            k, v = cache_prefix(outputs["new_k"], n), cache_prefix(outputs["new_v"], n)
            if (
                not np.isfinite(k[:, :, :, :n]).all()
                or not np.isfinite(v[:, :, :, :n]).all()
            ):
                raise RuntimeError("Invalid draft context projection")
            self.draft_k[:, :, :, self.draft_offset : self.draft_offset + n] = k[
                :, :, :, :n
            ]
            self.draft_v[:, :, :, self.draft_offset : self.draft_offset + n] = v[
                :, :, :, :n
            ]
            update = getattr(self.draft, "commit_cache", None)
            if update is not None:
                update(k[:, :, :, :n], v[:, :, :, :n], self.draft_offset, n)
            self.draft_offset += n

    def propose(self, anchor, count=None):
        if self.offset != self.draft_offset:
            raise RuntimeError("Draft/target cache offset mismatch")
        count = self.B - 1 if count is None else count
        if not 1 <= count < self.B:
            raise ValueError("Invalid draft lookahead")
        mask_id = self.manifest["draft_config"]["mask_token_id"]
        noise = self.embedding[[anchor] + [mask_id] * (self.B - 1)][None]
        cos, sin = self.positions(self.draft_offset)
        out = self.draft.predict(
            {
                "hidden": noise,
                "cos": cos,
                "sin": sin,
                "cache_k": self.draft_k,
                "cache_v": self.draft_v,
                "mask": causal_mask(self.draft_offset, self.C, self.B),
            }
        )
        hidden = np.asarray(out["hidden_out"], np.float16)
        if not np.isfinite(hidden).all() or not np.any(hidden):
            raise RuntimeError("Invalid ANE draft hidden state")
        return self.ids(hidden, start=1, end=count + 1, role="draft_head"), hidden[
            0, 1 : count + 1
        ]


def generate(stack, tok, prompt, limit, speculative, num_draft=None):
    stack.reset()
    num_draft = stack.B - 1 if num_draft is None else num_draft
    if not 1 <= num_draft < stack.B:
        raise ValueError("Draft lookahead exceeds the compiled block")
    prompt_ids = tok.encode(prompt)
    if len(prompt_ids) + limit > stack.C:
        raise ValueError("Prompt/generation exceed compiled context capacity")
    start = time.perf_counter()
    for begin in range(0, len(prompt_ids) - 1, stack.B):
        ids = prompt_ids[begin : min(begin + stack.B, len(prompt_ids) - 1)]
        _, features, pending = stack.forward(ids, False)
        stack.commit(pending, len(ids))
        if speculative:
            stack.append_draft(features)
    prefill = time.perf_counter() - start
    stack.profile = {}
    stack.calls = {}
    start = time.perf_counter()
    first, features, pending = stack.forward(prompt_ids[-1:])
    stack.commit(pending, 1)
    if speculative:
        stack.append_draft(features)
    tokens = [first[0]]
    traces = []
    accepted = proposed = 0
    eos = tok.eos_token_ids if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}
    while len(tokens) < limit and tokens[-1] not in eos:
        wanted = min(num_draft, max(0, limit - len(tokens) - 1))
        candidates = (
            stack.propose(tokens[-1], wanted)[0] if speculative and wanted else []
        )
        predictions, features, pending = stack.forward(
            [tokens[-1], *candidates], candidates=candidates
        )
        count = 0
        for candidate, prediction in zip(candidates, predictions):
            if candidate != prediction:
                break
            count += 1
        committed = [*candidates[:count], predictions[count]]
        for i, token in enumerate(committed):
            if token in eos:
                committed = committed[: i + 1]
                break
        stack.commit(pending, len(committed))
        if speculative:
            stack.append_draft(features[: len(committed)])
        tokens.extend(committed)
        traces.append(
            {
                "candidates": candidates,
                "predictions": predictions,
                "accepted": count,
                "committed": committed,
                "offset": stack.offset,
            }
        )
        accepted += min(count, len(committed))
        proposed += len(candidates)
    elapsed = time.perf_counter() - start
    return {
        "tokens": tokens,
        "text": tok.decode(tokens),
        "decode_s": elapsed,
        "prefill_s": prefill,
        "tok_per_s_decode": len(tokens) / elapsed,
        "cycles": len(traces),
        "accepted": accepted,
        "proposed": proposed,
        "profile_s": dict(stack.profile),
        "calls": dict(stack.calls),
        "trace": traces,
    }
