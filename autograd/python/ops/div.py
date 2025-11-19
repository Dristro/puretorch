from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np

from .context import Context
from .function import Function


class Div(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable", b: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a, b)
        return a.data / b.data

    def backward(self, grad_output: np.ndarray):
        a, b = self.ctx.saved_data
        out1 = grad_output / b.data
        out2 = -grad_output * a.data / (b.data ** 2)
        return out1, out2
