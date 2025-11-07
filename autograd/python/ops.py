from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

import numpy as np
from typing import Union, Tuple

from .context import Context
from .function import Function


class Add(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable", b: "Variable") -> np.ndarray:
        return a.data + b.data

    def backward(self, grad_output: np.ndarray):
        return grad_output, grad_output


class Sub(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable", b: "Variable") -> np.ndarray:
        return a.data - b.data

    def backward(self, grad_output: np.ndarray):
        return grad_output, -grad_output


class Mul(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable", b: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a, b)
        return a.data * b.data

    def backward(self, grad_output: np.ndarray):
        a, b = self.ctx.saved_data
        return grad_output * b.data, grad_output * a.data


class Div(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable", b: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a, b)
        return a.data / b.data

    def backward(self, grad_output: np.ndarray):
        a, b = self.ctx.saved_data
        out1 = grad_output / b.data
        out2 = -grad_output * a.data / (b.data ** 2)
        return out1, out2


class Neg(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable") -> np.ndarray:
        return -a.data

    def backward(self, grad_output: np.ndarray):
        return -grad_output


class MatMul(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable", b: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a, b)
        return a.data @ b.data

    def backward(self, grad_output: np.ndarray):
        a, b = self.ctx.saved_data
        da = grad_output @ b.data.T
        db = a.data.T @ grad_output
        return da, db


class Transpose(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a.shape)
        return a.data.T

    def backward(self, grad_output: np.ndarray):
        return grad_output.T


class Reshape(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable", shape: Tuple[int, ...]):
        self.ctx.save_for_backward(a.shape)
        return a.data.reshape(shape)

    def backward(self, grad_output: np.ndarray):
        (orig_shape,) = self.ctx.saved_data
        return grad_output.reshape(orig_shape)


class ReLU(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a)
        return np.maximum(a.data, 0)

    def backward(self, grad_output: np.ndarray):
        (a,) = self.ctx.saved_data
        grad = grad_output * (a.data > 0).astype(float)
        return grad


class Pow(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable", exponent: Union[int, float]):
        assert isinstance(exponent, (int, float)), f"Exponent must be int or float, got: {exponent}"
        self.ctx.save_for_backward(a, exponent)
        return a.data ** exponent

    def backward(self, grad_output: np.ndarray):
        a, exponent = self.ctx.saved_data
        return grad_output * exponent * (a.data ** (exponent - 1))


class VariableSum(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable", dim=None, keepdims=False):
        """
        Args:
            a: "Variable"
            dim: int or tuple of ints or None
            keepdims: if True, result shape matches 'a' in rank (PyTorch-style)
        """
        # Normalize dim to a sorted tuple of positive axes or None
        if dim is None:
            norm_dim = None
        else:
            if isinstance(dim, int):
                dim = (dim,)
            nd = a.data.ndim
            norm_dim = tuple(sorted(d if d >= 0 else d + nd for d in dim))
        self.ctx.save_for_backward(a.shape, norm_dim, keepdims)
        return np.sum(a.data, axis=norm_dim, keepdims=keepdims)

    def backward(self, grad_output: np.ndarray):
        shape, dim, keepdims = self.ctx.saved_data

        # Sum over all elements
        if dim is None:
            # grad_output shape is () if keepdims=False, or shape of ones if keepdims=True
            return np.ones(shape, dtype=grad_output.dtype) * grad_output

        # We reduced some axes. If keepdims=False, reinsert singleton dims at reduced positions
        if not keepdims:
            shp = list(shape)
            for d in dim:
                shp[d] = 1
            # now broadcastable to 'shape'
            grad_output = grad_output.reshape(shp)

        return np.ones(shape, dtype=grad_output.dtype) * grad_output


class Mean(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable", dim=None, keepdims=False):
        if dim is None:
            norm_dim = None
            denom = a.data.size
        else:
            if isinstance(dim, int):
                dim = (dim,)
            nd = a.ndim
            norm_dim = tuple(sorted(d if d >= 0 else d + nd for d in dim))
            denom = 1
            for d in norm_dim:
                denom *= a.shape[d]
        self.ctx.save_for_backward(a.shape, norm_dim, keepdims, denom)
        return np.mean(a.data, axis=norm_dim, keepdims=keepdims)

    def backward(self, grad_output: np.ndarray):
        shape, dim, keepdims, denom = self.ctx.saved_data

        if dim is None:
            # grad_output is scalar if keepdims=False
            out = np.ones(shape, dtype=grad_output.dtype) * grad_output
            out = (out) / denom
            return out

        if not keepdims:
            shp = list(shape)
            for d in dim:
                shp[d] = 1
            grad_output = grad_output.reshape(shp)

        return (np.ones(shape, dtype=grad_output.dtype) * grad_output) / denom


class Exp(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable") -> np.ndarray:
        out = np.exp(a.data)
        self.ctx.save_for_backward(out)
        return out

    def backward(self, grad_output: np.ndarray):
        out = self.ctx.saved_data
        return grad_output * out


class Log(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: "Variable") -> np.ndarray:
        self.ctx.save_for_backward(a)
        return np.log(a.data)

    def backward(self, grad_output: np.ndarray):
        a = self.ctx.saved_data
        return grad_output / a.data
