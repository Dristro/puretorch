from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np

from .context import Context
from .function import Function


class Transpose(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a)
        return a.data.T

    def backward(self, grad_output: np.ndarray):
        return grad_output.T
