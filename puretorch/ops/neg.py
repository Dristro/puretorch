import numpy as np

from ..context import Context
from ..function import Function


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray):
        return -a

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return -grad_output
