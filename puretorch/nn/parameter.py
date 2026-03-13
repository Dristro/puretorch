import numpy as np

from puretorch import Tensor
from autograd import Variable
from typing import Literal, get_args

AllowedName = Literal[
    "w_linear",
    "b_linear",
]
_ALLOWED_NAMES = get_args(AllowedName)


class Parameter(Tensor):
    """
    Alias to mark a Tensor as trainable parameter.

    Additional to Tensor, Parameter also stores information like
    `name`. This will be particularly helpful for Module specific
    operations.
    """

    def __init__(
        self,
        data: int | float | list | tuple | np.ndarray | Tensor | Variable,
        name: AllowedName,
        requires_grad: bool = True,
    ):
        """
        Args:
            data (see Tensor docs for dtype): data for the tensor
            name (str): name given to parameter
            requires_grad (bool, default=True): requires grad
        """
        super().__init__(
            data,
            requires_grad=requires_grad,
            is_leaf=True,
            device="cpu",
        )

        if name not in _ALLOWED_NAMES:
            raise ValueError(
                f"Unsupported `name`, expected one of "
                f"({', '.join(_ALLOWED_NAMES)}), got '{name}'."
            )
        self.name = name
