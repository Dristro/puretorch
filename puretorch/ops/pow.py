import numpy as np

from ..context import Context
from ..function import Function


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, exponent: int | float):
        assert isinstance(exponent, (int, float)), (
            f"Exponent must be int or float, got: {exponent}"
        )
        ctx.save_for_backward(a, exponent)
        return a**exponent

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a, exponent = ctx.saved_tensors
        return grad_output * exponent * (a ** (exponent - 1))
