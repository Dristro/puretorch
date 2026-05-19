import numpy as np

from ..context import Context
from ..function import Function


class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, shape):
        ctx.save_for_backward(a.shape)
        return a.reshape(shape)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        (orig_shape,) = ctx.saved_tensors
        return grad_output.reshape(orig_shape)
