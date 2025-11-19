from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np

from .context import Context
from .function import Function


class Log(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a)
        return np.log(a.data)

    def backward(self, grad_output: np.ndarray):
        a, = self.ctx.saved_data
        return grad_output / a.data
