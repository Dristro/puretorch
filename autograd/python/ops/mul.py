from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np

from .context import Context
from .function import Function


class Mul(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable", b: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a, b)
        return a.data * b.data

    def backward(self, grad_output: np.ndarray):
        a, b = self.ctx.saved_data
        return grad_output * b.data, grad_output * a.data
