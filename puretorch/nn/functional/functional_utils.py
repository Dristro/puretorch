"""
All util functions for the funtional module of puretorch.

The most important utility here is the wrap forward function,
it allows for autograd-safe function forward for tensors.
That is, manually casting a tensor into a variable for forward,
then cast the output back into a tensor.
This is an inefficient system, autograd will later be chenged to
make this process smoother for tensors.
"""
import numpy as np
from typing import Type, Any

from puretorch import Tensor
from autograd.python.ops import Function, Context


def _enforce_tensor(x: Any) -> Tensor:
    """
    Converts 'x' into a Tensor instance if not already.
    Used for tensor-ops.
    Output variable properties:
        requires_grad = False
        is_leaf = True
        dtype: float
    """
    if isinstance(x, Tensor):
        return x
    return Tensor(
        data=np.array(x, dtype=float),
        requires_grad=False,
        is_leaf=True,
        grad_fn=None,
        device="cpu",  # manually set device
    )


def _wrap_forward(
    fn_cls: Type[Function],
    *parents: Tensor,
    **kwargs
) -> Tensor:
    """
    **NOTE**: Same function as found in Variable, its used for functional where
        some function's are defined in autograd.ops, but not integrated to
        variable. One such function is ReLU, its not bound to Variable/Tensor,
        its part of nn.functional, so we manually bind it to tensors/variables
        in functional.

    **Important**: Unlike wrap_forward in variable, this function expects
                   Tensors only, Variables are not allowed.

    autograd safe tensor/variable operations. This function is used to
    manually bind and manage context for a given numerical function
    defined for tensors/variables.

    Args:
        fn_cls (Function): class of the function called
        parents (Variable or Tensor): ts/vs participating in function
        **kwargs: all the kwargs for function forward call
    Returns:
        Variable or Tensor: output tensor with data.
    """

    # Fresh context for function
    ctx = Context()
    versions = [p._version for p in parents]
    ctx.version_snapshot = tuple(versions)
    fn = fn_cls(ctx)

    # Compute function output
    out_data = fn.forward(*parents, **kwargs)

    # Create output as tensor
    devices = [p.device for p in parents if hasattr(p, "device")]
    device = devices[-1]  # ISSUE: device presedence needs to be provided
    out = Tensor(
            data=out_data,
            requires_grad=any(p.requires_grad for p in parents),
            grad_fn=None,  # Will be assigned after
            is_leaf=False,
            device=device,  # pyright: ignore
        )
    # Binding a grad_fn
    out.grad_fn = fn

    return out
