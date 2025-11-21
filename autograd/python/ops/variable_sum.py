from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np
from typing import Union, Optional, Tuple

from .function import Function


class VariableSum(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(
        self,
        a: "Variable",
        dim: Optional[Union[Tuple[int], int]] = None,
        keepdims: bool = False,
    ):
        """
        Args:
            a: "Variable"
            dim: int or tuple of ints or None
            keepdims: if True, result shape matches 'a' in rank (PyTorch-style)
        """
        # Normalize dim to a sorted tuple of positive axes or None
        if isinstance(dim, int):
            dim = (dim,)
        self.ctx.save_for_backward(a, dim, keepdims)
        return np.sum(a.data, axis=dim, keepdims=keepdims)

    def backward(self, grad_output: np.ndarray):
        item, dim, keepdims = self.ctx.saved_data
        shape = item.shape
        grad = np.ones(shape, dtype=grad_output.dtype)

        # Sum over all elements
        if dim is None:
            # grad_output shape is () if keepdims=False, or shape of ones if keepdims=True
            return grad * grad_output

        # We reduced some axes. If keepdims=False, reinsert singleton dims at reduced positions
        if not keepdims:
            shp = list(shape)
            for d in dim:
                shp[d] = 1
            # now broadcastable to 'shape'
            grad_output = grad_output.reshape(shp)

        return grad * grad_output
