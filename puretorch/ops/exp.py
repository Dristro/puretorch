import numpy as np

from ..context import Context
from ..function import Function


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray):
        out = np.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        out = ctx.saved_tensors
        return grad_output * out
