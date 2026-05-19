import numpy as np

from ..context import Context
from ..function import Function


class Index(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, idx: int | list | tuple):
        # Tensor
        def unwrap(i):
            if hasattr(i, "_data"):
                return i._data
            return i

        if isinstance(idx, tuple):
            idx = tuple(unwrap(i) for i in idx)
        elif isinstance(idx, list):
            idx = [unwrap(i) for i in idx]
        else:
            idx = unwrap(idx)

        ctx.save_for_backward(x.shape, idx)

        out = x[idx]
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        grad_input = np.zeros(ctx.saved_tensors[0])
        grad_input[ctx.saved_tensors[1]] += grad_output

        return grad_input, None
