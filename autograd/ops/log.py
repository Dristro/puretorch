import numpy as np

from ..context import Context
from ..function import Function


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray):
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a = ctx.saved_tensors
        return grad_output / a
