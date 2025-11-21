import numpy as np
from .context import Context


class Function:
    """Base class for autograd functions."""

    def __init__(self, ctx: Context):
        """
        Initialize Function instance with context.
        Child class is expected to perform ctx.save_for_later
        for proper functionality. The Variable class doesn't
        work as expected if ctx.save_for_backward isnt applied.
        """
        self._ctx = ctx

    @property
    def ctx(self):
        return self._ctx

    def forward(self) -> np.ndarray:
        """
        Operation/Function forward logic.
        Child class is expected to save tensors for bakcward
        in the Context (`ctx`) variable and return output.

        Calling forward on Function will result in `NotImplementedError`
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Operation/Function backward logic.
        Child class is expected to used saved tensors from
        Context (`ctx`) and return gradient.

        Calling backward on Function will result in `NotImplementedError`
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n"\
               f"\tctx={self.ctx}\n"\
               ")"
