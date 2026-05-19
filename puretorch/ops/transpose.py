import numpy as np

from ..context import Context
from ..function import Function


class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a.shape)
        return a.T

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        # (orig_shape,) = ctx.saved_tensors  # DEBUG
        return grad_output.T
