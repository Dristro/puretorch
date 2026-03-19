import numpy as np

from puretorch import Tensor
from autograd.context import Context
from autograd.function import Function
from autograd import enforce_tensor, _wrap_forward


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray):
        ctx.save_for_backward(a)
        return np.maximum(a, 0)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        (a,) = ctx.saved_tensors
        grad = grad_output * (a > 0).astype(float)
        return grad


def relu(x: Tensor) -> Tensor:
    x = enforce_tensor(x)
    return _wrap_forward(ReLU, x)
