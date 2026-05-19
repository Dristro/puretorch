import numpy as np

from .context import Context


class Function:
    """Base class for autograd supported functions."""

    _parents = []

    @staticmethod
    def forward(ctx: Context, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        raise NotImplementedError
