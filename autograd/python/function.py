import numpy as np
from .context import Context


class Function:
    """Base class for autograd functions."""

    def __init__(self):
        """
        Initialize Function instance with context.
        """
        self._ctx = Context()

    @property
    def ctx(self):
        return self._ctx

    def forward(self, *args, **kwargs) -> np.ndarray:
        """
        Operation/Function forward logic.
        Child class is expected to save tensors for bakcward
        in the Context (`ctx`) variable and return output.

        Calling forward on Function will result in `NotImplementedError`
        """
        raise NotImplementedError

    def barkward(self, grad_output):
        """
        Operation/Function backward logic.
        Child class is expected to used saved tensors from
        Context (`ctx`) and return gradient.

        Calling backward on Function will result in `NotImplementedError`
        """
        raise NotImplementedError
