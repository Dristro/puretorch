from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np
from typing import Tuple

from .function import Function


class Reshape(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable", shape: Tuple[int, ...]):
        self.ctx.save_for_backward(a)
        return a.data.reshape(shape)

    def backward(self, grad_output: np.ndarray):
        orig_shape, = self.ctx.saved_data
        orig_shape = orig_shape.shape
        return grad_output.reshape(orig_shape)
