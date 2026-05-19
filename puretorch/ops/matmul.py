import numpy as np

from ..context import Context
from ..function import Function


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a, b = ctx.saved_tensors
        da = grad_output @ b.T
        db = a.T @ grad_output
        return da, db
