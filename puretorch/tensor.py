import numpy as np
from typing import Union, Optional
from autograd import Variable, Function


class Tensor(Variable):
    def __init__(
        self,
        data: Union[int, float, list, tuple, np.ndarray, Variable],
        requires_grad: bool = False,
        grad_fn: Optional[Function] = None,
        is_leaf: bool = True,
        device: str = "cpu",
    ):
        """
        Creates a Tensor instance with data.

        Args:
            data: data in tensor
            requires_grad: tracks gradient if `True`
            grad_fn: function used to arrive to current tensor
            is_leaf: is tensor a leaf-tensor
            device: device the tensor lives on (cpu only for now)
        """
        super().__init__(
            data=data,
            requires_grad=requires_grad,
            grad_fn=grad_fn,
            is_leaf=is_leaf,
        )
        self._device = device

    @property
    def device(self):
        return self._device

    def item(self):
        """
        Returns data within tensor.
        Same as accessing `.data`.
        """
        return self.data

    def relu(self) -> "Tensor":
        """
        **Deprication warning**:
        This function will be removed in the next release.
        Please use `nn.functional.relu()`.

        Computes relu and returns new instance.
        Returns:
            Tensor
        """
        return Tensor(
            data=self._relu(),
            requires_grad=self.requires_grad,
            is_leaf=False,
        )

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __repr__(self):
        formatted_data = np.array2string(
            self.data,
            precision=4,
            suppress_small=True,
            separator=', ',
            prefix=' ' * 7
        )
        statement = f"tensor({formatted_data}, requires_grad={self.requires_grad})" if self.requires_grad else f"tensor({formatted_data})"
        return statement
