"""Check lossless SD cache commits on a small deterministic Metal Qwen3.

Cover complete, partial and zero acceptance, truncated final blocks and EOS.
Committed features must match scalar decoding, not merely the returned tokens.
"""
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3 import Model, ModelArgs

from bench_macos_dflash import device_locks, generate, target_forward


class Tokens:
    eos_token_ids = []

    def encode(self, _):
        return [2, 7, 4]

    def decode(self, ids):
        return str(ids)


class Draft:
    feature_ids = (1, 2)

    def __init__(self, gold, mode):
        self.gold, self.mode = gold, mode
        self.reset()

    def reset(self):
        self.offset = self.cycles = 0
        self.features = []
        self.profile = {"context": 0.0, "body": 0.0, "head": 0.0}

    def append(self, features):
        self.offset += len(features)
        self.features.append(np.array(features.astype(mx.float32)))

    def propose(self, _):
        index = self.offset - 3 + 1
        ids = (self.gold[index:index + 3] + [0, 0, 0])[:3]
        if self.mode == "reject":
            ids[0] = (ids[0] + 1) % 32
        elif self.mode == "partial" and self.cycles % 2 == 0:
            ids[1] = (ids[1] + 1) % 32
        self.cycles += 1
        return ids, None


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.set_default_device(mx.gpu)
        mx.random.seed(16)
        cls.model = Model(ModelArgs(model_type="qwen3", hidden_size=32,
                          num_hidden_layers=2, intermediate_size=64, num_attention_heads=2,
                          rms_norm_eps=1e-6, vocab_size=32, num_key_value_heads=1,
                          max_position_embeddings=128, rope_theta=1e6,
                          head_dim=16, tie_word_embeddings=True))
        mx.eval(cls.model.parameters())
        cls.tok = Tokens()
        cls.gold = generate(cls.model, cls.tok, None, "prompt", 14, False)["tokens"]
        cache = make_prompt_cache(cls.model)
        cls.features = []
        for token in cls.tok.encode("") + cls.gold[:-1]:
            _, f = target_forward(cls.model, [token], cache, Draft.feature_ids)
            cls.features.append(np.array(f))

    def check(self, mode, verification="batch"):
        draft = Draft(self.gold, mode)
        result = generate(self.model, self.tok, draft, "prompt", 14, True, verification)
        self.assertEqual(result["tokens"], self.gold)
        self.assertEqual(draft.offset, 3 + 14 - 1)
        np.testing.assert_allclose(np.concatenate(draft.features),
                                   np.concatenate(self.features), atol=2e-5, rtol=2e-5)
        return result

    def test_accept_all_and_short_tail(self):
        self.assertGreater(self.check("accept")["accepted"], 0)

    def test_reject_first(self):
        self.assertEqual(self.check("reject")["accepted"], 0)

    def test_partial_and_full_blocks(self):
        result = self.check("partial")
        self.assertGreater(result["accepted"], 0)
        self.assertLess(result["accepted"], result["proposed"])

    def test_serial_reference(self):
        for mode in ("accept", "reject", "partial"):
            with self.subTest(mode=mode):
                self.check(mode, "serial")

    def test_eos_inside_accepted_block(self):
        class Cache:
            offset = 0

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.offset -= n
                return n

        class EOSTokens(Tokens):
            eos_token_ids = [9]

        gold = [4, 8, 9, 10, 11, 12]
        state = Cache()

        def forward(_model, ids, cache, feature_ids=(), logits=True):
            begin = cache[0].offset
            cache[0].offset += len(ids)
            features = mx.arange(begin, begin + len(ids), dtype=mx.float32)[:, None]
            predictions = [gold[pos - 2] if pos >= 2 else 0 for pos in range(begin, begin + len(ids))]
            return predictions if logits else None, features

        draft = Draft(gold, "accept")
        with patch("bench_macos_dflash.target_forward", forward), patch(
                "mlx_lm.models.cache.make_prompt_cache", lambda _: [state]):
            result = generate(self.model, EOSTokens(), draft, "prompt", 14, True)
        self.assertEqual(result["tokens"], [4, 8, 9])
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(state.offset, 5)
        self.assertEqual(draft.offset, 5)
        np.testing.assert_array_equal(np.concatenate(draft.features)[:, 0], np.arange(5))


class SmallMatmulTests(unittest.TestCase):
    def test_nonmultiple_dimensions_and_all_block_widths(self):
        from macos_small_matmul import linear
        mx.random.seed(72)
        for dtype in (mx.bfloat16, mx.float16):
            for rows in (2, 4, 8):
                with self.subTest(dtype=dtype, rows=rows):
                    x = mx.random.normal((1, rows, 73)).astype(dtype)
                    w = (mx.random.normal((67, 73)) / 73**0.5).astype(dtype)
                    reference = (x.astype(mx.float32) @ w.astype(mx.float32).T).astype(dtype)
                    actual = linear(x, w)
                    mx.eval(actual, reference)
                    np.testing.assert_allclose(np.array(actual.astype(mx.float32)),
                                               np.array(reference.astype(mx.float32)),
                                               atol=3e-5, rtol=0.008)

    def test_install_disable_preserves_original_linear(self):
        import mlx.nn as nn
        from macos_small_matmul import install, set_enabled
        m = nn.Sequential(nn.Linear(73, 67, bias=True), nn.ReLU(), nn.Linear(67, 17, bias=False))
        m.set_dtype(mx.bfloat16)
        x = mx.random.normal((4, 73)).astype(mx.bfloat16)
        expected = m(x)
        mx.eval(expected)
        modules = install(m)
        self.assertEqual(len(modules), 2)
        set_enabled(modules, False)
        self.assertTrue(mx.array_equal(m(x), expected).item())


if __name__ == "__main__":
    with device_locks():
        unittest.main()
