from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np

from .function import Function


class MatMul(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable", b: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a, b)
        return a.data @ b.data

    def backward(self, grad_output: np.ndarray):
        a, b = self.ctx.saved_data
        da = grad_output @ b.data.T
        db = a.data.T @ grad_output
        return da, db
