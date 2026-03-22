from puretorch.device import Device
from puretorch.dtype import DType
from puretorch.tensor import Tensor
from puretorch import nn
from puretorch import optim
from autograd import no_grad, enable_grad
from puretorch.nn import functional
from puretorch.utils.tensor_utils import (
    tensor,
    allclose,
    all,
    equal,
    zeros_like,
    linspace,
)
from puretorch.utils.viz import make_dot

DTYPES = list(DType.__members__.keys())

globals().update(
    {name: member.value for name, member in DType.__members__.items()}
)

__version__ = "1.2.0+dev"

__all__ = [
    "Device",
    "DType",
    *DTYPES,
    "Tensor",
    "nn",
    "optim",
    "no_grad",
    "enable_grad",
    "functional",
    "tensor",
    "allclose",
    "all",
    "equal",
    "zeros_like",
    "linspace",
    "make_dot",
]
