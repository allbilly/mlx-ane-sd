"""Diagnostic linear routes for small speculative verification blocks.

Only multirow inputs with 2..31 rows change. The scalar decode baseline keeps
the original route. Neither route promises full-model numerical equivalence.
"""
from contextlib import contextmanager


def matmul(x, weight, mode):
    import mlx.core as mx

    rows = x.size // x.shape[-1]
    if mode == "stock" or not 2 <= rows < 32:
        return x @ weight.T
    flat = x.reshape(rows, x.shape[-1])
    if mode == "row-gemv":
        # A rank-2 RHS lets MLX flatten the batch back into one GEMM.
        # Explicit stride-zero broadcasting preserves independent GEMVs.
        rhs = mx.broadcast_to(weight.T, (rows, weight.shape[1], weight.shape[0]))
        out = mx.matmul(flat[:, None, :], rhs)[:, 0, :]
    elif mode == "pad32":
        padded = mx.concatenate((flat, mx.zeros((32 - rows, flat.shape[-1]), flat.dtype)))
        out = (padded @ weight.T)[:rows]
    else:
        raise ValueError(mode)
    return out.reshape(*x.shape[:-1], weight.shape[0])


@contextmanager
def linear_route(mode):
    import mlx.nn as nn

    if mode not in ("stock", "row-gemv", "pad32"):
        raise ValueError(mode)
    if mode == "stock":
        yield
        return
    original_linear, original_head = nn.Linear.__call__, nn.Embedding.as_linear

    def call(layer, x):
        out = matmul(x, layer.weight, mode)
        return out + layer.bias if "bias" in layer else out

    def head(layer, x):
        return matmul(x, layer.weight, mode)

    nn.Linear.__call__, nn.Embedding.as_linear = call, head
    try:
        yield
    finally:
        nn.Linear.__call__, nn.Embedding.as_linear = original_linear, original_head
