from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np

from .function import Function


class Exp(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable") -> np.ndarray:
        out = np.exp(a.data)
        self.ctx.save_for_backward(a, out)
        return out

    def backward(self, grad_output: np.ndarray):
        _, out = self.ctx.saved_data
        return grad_output * out
