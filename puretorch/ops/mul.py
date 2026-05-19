import numpy as np

from ..context import Context
from ..function import Function


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a
