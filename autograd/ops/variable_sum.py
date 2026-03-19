import numpy as np

from ..context import Context
from ..function import Function


class VariableSum(Function):
    ###
    # Used chatgpt to fix bugs in this code, leading to issues with ce-loss
    ###
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, dim=None, keepdims=False):
        """
        Args:
            a: np.ndarray
            dim: int or tuple of ints or None
            keepdims: if True, result shape matches 'a' in rank (PyTorch-style)
        """
        # Normalize dim to a sorted tuple of positive axes or None
        if dim is None:
            norm_dim = None
        else:
            if isinstance(dim, int):
                dim = (dim,)
            nd = a.ndim
            norm_dim = tuple(sorted(d if d >= 0 else d + nd for d in dim))
        ctx.save_for_backward(a.shape, norm_dim, keepdims)
        return np.sum(a, axis=norm_dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        shape, dim, keepdims = ctx.saved_tensors

        # Sum over all elements
        if dim is None:
            # grad_output shape is () if keepdims=False, or shape of ones if keepdims=True
            return np.ones(shape, dtype=grad_output.dtype) * grad_output

        # We reduced some axes. If keepdims=False, reinsert singleton dims at reduced positions
        if not keepdims:
            shp = list(shape)
            for d in dim:
                shp[d] = 1
            grad_output = grad_output.reshape(shp)  # now broadcastable to 'shape'

        return np.ones(shape, dtype=grad_output.dtype) * grad_output
