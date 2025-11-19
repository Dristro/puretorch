from puretorch import Tensor

from autograd.python.ops import ReLU
from .functional_utils import _wrap_forward, _enforce_tensor


def relu(x: Tensor) -> Tensor:
    x = _enforce_tensor(x)
    out = _wrap_forward(ReLU, x)
    return out
