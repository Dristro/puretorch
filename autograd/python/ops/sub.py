from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np

from .function import Function


class Sub(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable", b: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a, b)
        return a.data - b.data

    def backward(self, grad_output: np.ndarray):
        return grad_output, -grad_output
