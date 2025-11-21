import numpy as np
from typing import Union, List, Tuple
from autograd import Variable
from puretorch import Tensor


class Parameter(Tensor):
    """Simple alias to mark a Tensor as trainable parameter."""

    def __init__(
        self,
        data: Union[int, float, List, Tuple, np.ndarray, Variable],
        requires_grad: bool = True,
    ):
        super().__init__(
            data,
            requires_grad=requires_grad,
            is_leaf=True
        )
