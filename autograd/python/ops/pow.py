from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np
from typing import Union

from .context import Context
from .function import Function


class Pow(Function):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, a: "Variable", exponent: Union[int, float]):
        assert isinstance(exponent, (int, float)), f"Exponent must be int or float, got: {exponent}"
        self.ctx.save_for_backward(a, exponent)
        return a.data ** exponent

    def backward(self, grad_output: np.ndarray):
        a, exponent = self.ctx.saved_data
        return grad_output * exponent * (a.data ** (exponent - 1))
