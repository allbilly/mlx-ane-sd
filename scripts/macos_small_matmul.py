"""Experimental small-block Metal linear: reuse each weight across 2–8 rows.

Weights and activations retain their original dtype; dot products accumulate
in FP32. Kernel reduction order can differ from stock MLX, so end-to-end token
identity must be checked before reporting a lossless speedup.
"""
import math

import mlx.core as mx
import mlx.nn as nn

_kernel = mx.fast.metal_kernel(
    name="dflash_small_block_linear",
    input_names=["x", "w"],
    output_names=["out"],
    source="""
        uint lane = thread_index_in_simdgroup;
        uint row = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
        if (row >= N) return;
        float accum[M];
        for (uint m = 0; m < M; ++m) accum[m] = 0.0f;
        for (uint k = lane; k < K; k += 32) {
            float weight = float(w[row * K + k]);
            for (uint m = 0; m < M; ++m)
                accum[m] = metal::fma(float(x[m * K + k]), weight, accum[m]);
        }
        for (uint m = 0; m < M; ++m) {
            float total = metal::simd_sum(accum[m]);
            if (lane == 0) out[m * N + row] = T(total);
        }
    """,
)


def linear(x, weight):
    rows = math.prod(x.shape[:-1])
    if not 2 <= rows <= 8 or x.dtype != weight.dtype or x.dtype not in (mx.bfloat16, mx.float16):
        return x @ weight.T
    n, k = weight.shape
    return _kernel(inputs=[x, weight], template=[("T", x.dtype), ("M", rows), ("N", n), ("K", k)],
                   grid=(((n + 3) // 4) * 128, 1, 1), threadgroup=(128, 1, 1),
                   output_shapes=[(*x.shape[:-1], n)], output_dtypes=[x.dtype])[0]


class SmallLinear(nn.Linear):
    def __call__(self, x):
        if not self._small_enabled:
            return super().__call__(x)
        out = linear(x, self.weight)
        return out + self.bias if "bias" in self else out


class SmallEmbedding(nn.Embedding):
    def as_linear(self, x):
        return linear(x, self.weight) if self._small_enabled else x @ self.weight.T


def install(model):
    """Change only linear evaluation, keeping each existing parameter array."""
    installed = []
    def visit(module):
        if isinstance(module, nn.Linear):
            module.__class__ = SmallLinear
            module._small_enabled = True
            installed.append(module)
            return
        if isinstance(module, nn.Embedding):
            module.__class__ = SmallEmbedding
            module._small_enabled = True
            installed.append(module)
            return
        for name, child in list(module.children().items()):
            if isinstance(child, list):
                for member in child:
                    visit(member)
            elif isinstance(child, nn.Module):
                visit(child)
    visit(model)
    return installed


def set_enabled(modules, enabled):
    for module in modules:
        module._small_enabled = enabled
