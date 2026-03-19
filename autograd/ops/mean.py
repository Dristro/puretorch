import numpy as np

from ..context import Context
from ..function import Function


class Mean(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, dim=None, keepdims=False):
        if dim is None:
            norm_dim = None
            denom = a.size
        else:
            if isinstance(dim, int):
                dim = (dim,)
            nd = a.ndim
            norm_dim = tuple(sorted(d if d >= 0 else d + nd for d in dim))
            denom = 1
            for d in norm_dim:
                denom *= a.shape[d]
        ctx.save_for_backward(a.shape, norm_dim, keepdims, denom)
        return np.mean(a, axis=norm_dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        shape, dim, keepdims, denom = ctx.saved_tensors

        if dim is None:
            # grad_output is scalar if keepdims=False
            return (np.ones(shape, dtype=grad_output.dtype) * grad_output) / denom

        if not keepdims:
            shp = list(shape)
            for d in dim:
                shp[d] = 1
            grad_output = grad_output.reshape(shp)

        return (np.ones(shape, dtype=grad_output.dtype) * grad_output) / denom
