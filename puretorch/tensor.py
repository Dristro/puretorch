import numpy as np

from autograd import Variable, Function
from .device import Device
from .dtype import DTYPES


class Tensor(Variable):
    def __init__(
        self,
        data: int | float | list | tuple | np.ndarray | Variable,
        requires_grad: bool = False,
        grad_fn: Function | None = None,
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

    def _to(self, device: str | Device, dtype: DTYPES):  # pyright: ignore
        # Device
        Warning("[WARNING] tensor.device only supports `cpu`.")
        if isinstance(device, str):
            device = Device(device)  # pyright: ignore
        self._device = device

        # dtype
        self._dtype = dtype
        self._data = self._data.astype(dtype)

    @property
    def device(self):
        return self._device

    def item(self):
        return self.data

    def __repr__(self):
        # return f"tensor({self.data}, requires_grad={self.requires_grad}, device={self.device}, dtype={self.dtype})"
        formatted_data = np.array2string(
            self.data, precision=4, suppress_small=True, separator=", ", prefix=" " * 7
        )
        statement = (
            f"tensor({formatted_data}, requires_grad={self.requires_grad})"
            if self.requires_grad
            else f"tensor({formatted_data})"
        )
        return statement
