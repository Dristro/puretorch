import numpy as np

from ..context import Context
from ..function import Function


class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray):
        ctx.save_for_backward(None, None)  # we will not need the tensors
        return a + b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return grad_output, grad_output
