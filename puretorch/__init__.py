from puretorch.device import Device
from puretorch.dtype import DType
from puretorch.tensor import Tensor, _unbroadcast, enforce_tensor, _wrap_forward
from puretorch import nn
from puretorch import optim
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
from .context import Context
from .function import Function
from . import ops
from .grad_mode import no_grad, enable_grad


DTYPES = list(DType.__members__.keys())

globals().update(
    {name: member.value for name, member in DType.__members__.items()}
)

__version__ = "1.3.0+dev"

__all__ = [
    "Device",
    "DType",
    *DTYPES,
    "Tensor",
    "_unbroadcast",
    "enforce_tensor",
    "_wrap_forward",
    "Context",
    "Function",
    "ops",
    "no_grad",
    "enable_grad",
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
