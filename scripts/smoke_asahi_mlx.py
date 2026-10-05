"""GPU primitives and SD cache paths; untrained fixture, no quality/rate claim."""
import mlx.core as mx

from mlx_lm.generate import generate_step, speculative_generate_step
from mlx_lm.models.qwen3 import Model, ModelArgs


def main():
    assert mx.default_device() == mx.Device(mx.gpu)
    x = mx.arange(32 * 32, dtype=mx.float32).reshape(32, 32) / 1024
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        a = x.astype(dtype)
        y = a @ mx.ones((32, 16), dtype=dtype)
        mx.eval(y)
        assert y.shape == (32, 16)
        assert mx.all(mx.isfinite(y)).item()
        reference = mx.sum(a.astype(mx.float32), axis=1)[:, None]
        error = mx.max(mx.abs(y.astype(mx.float32) - reference)).item()
        assert error < 0.2, error
        norm = mx.fast.rms_norm(a, mx.ones((32,), dtype=dtype), 1e-6)
        rope = mx.fast.rope(a.reshape(1, 1, 32, 32), 32, traditional=False,
                            base=1000000, scale=1, offset=0)
        attention = mx.fast.scaled_dot_product_attention(
            rope, rope, rope, scale=32 ** -0.5, mask="causal")
        mx.eval(norm, rope, attention)
        assert mx.all(mx.isfinite(norm)).item()
        assert mx.all(mx.isfinite(attention)).item()
        print(f"PASS {dtype}: matmul max_abs_error={error}, RMSNorm, RoPE, causal SDPA",
              flush=True)

    # Deliberately untrained, small shape fixture: exercise both acceptance and
    # rejection without a model download. It is not a quantization quality test.
    args = ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=2, rms_norm_eps=1e-6,
        vocab_size=256, num_key_value_heads=1, max_position_embeddings=512,
        rope_theta=1000000, head_dim=32, tie_word_embeddings=False,
    )
    mx.random.seed(42)
    target = Model(args)
    target.apply(lambda a: a.astype(mx.bfloat16))
    mx.eval(target.parameters())
    draft = Model(args)
    draft.update(target.parameters())
    mx.eval(draft.parameters())
    prompt = mx.array([65, 66, 67, 68])
    baseline = [x[0] for x in generate_step(prompt, target, max_tokens=16)]
    for different in (False, True):
        if different:
            draft.lm_head.weight = -draft.lm_head.weight
        responses = list(speculative_generate_step(
            prompt, target, draft, num_draft_tokens=3, max_tokens=16))
        tokens = [x[0] for x in responses]
        accepted = sum(x[2] for x in responses)
        assert tokens == baseline, (baseline, tokens)
        assert accepted < len(tokens) if different else accepted > 0
        print(f"PASS {'different' if different else 'identical'} draft: "
              f"token_ids_equal=True, accepted={accepted}/{len(tokens)}", flush=True)
    mx.synchronize()
    print("PASS GPU primitives and untrained SD acceptance/rejection/cache fixture",
          flush=True)


if __name__ == "__main__":
    main()
