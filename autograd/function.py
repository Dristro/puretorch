import numpy as np

from .context import Context


class Function:
    """Base class for autograd functions."""

    _parents = []

    @staticmethod
    def forward(ctx: Context, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def barkward(ctx: Context, grad_output: np.ndarray):
        raise NotImplementedError
