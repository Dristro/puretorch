"""
This function is not exposed in Variable.
Its sole purpose is to be used for Tensor.
I dont want to 'mix' autograd and puretorch's
functionality, so puretorch.nn.functional.relu
will call this implementation of ReLU.
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np

from .function import Function


class ReLU(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a)
        return np.maximum(a.data, 0)

    def backward(self, grad_output: np.ndarray):
        a, = self.ctx.saved_data
        return grad_output * (a > 0).astype(float)
