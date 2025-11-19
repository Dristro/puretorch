from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np

from .context import Context
from .function import Function


class Mean(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable", dim=None, keepdims=False):
        if dim is None:
            norm_dim = None
            denom = a.data.size
        else:
            if isinstance(dim, int):
                dim = (dim,)
            nd = a.ndim
            norm_dim = tuple(sorted(d if d >= 0 else d + nd for d in dim))
            denom = 1
            for d in norm_dim:
                denom *= a.shape[d]
        self.ctx.save_for_backward(a, norm_dim, keepdims, denom)
        return np.mean(a.data, axis=norm_dim, keepdims=keepdims)

    def backward(self, grad_output: np.ndarray):
        item, dim, keepdims, denom = self.ctx.saved_data
        shape = item.shape

        if dim is None:
            # grad_output is scalar if keepdims=False
            out = np.ones(shape, dtype=grad_output.dtype) * grad_output
            out = (out) / denom
            return out

        if not keepdims:
            shp = list(shape)
            for d in dim:
                shp[d] = 1
            grad_output = grad_output.reshape(shp)

        return (np.ones(shape, dtype=grad_output.dtype) * grad_output) / denom
