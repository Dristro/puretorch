from typing import TYPE_CHECKING

from numpy.random import normal
if TYPE_CHECKING:
    from variable import Variable

import numpy as np
from typing import Union, Tuple, Optional

from .context import Context
from .function import Function


class ReLU(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a)
        return np.maximum(a.data, 0)

    def backward(self, grad_output: np.ndarray):
        a, = self.ctx.saved_data
        grad = grad_output * (a.data > 0).astype(float)
        return grad
