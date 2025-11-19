from unittest import result
import weakref
import numpy as np
from typing import (
    Any,
    Type,
    List,
    Union,
    Tuple,
    Callable,
    Optional,
)

from .grad_mode import is_grad_enabled
from .ops.context import Context
from .ops.function import Function
from .ops import (
    Add,
    Mul,
    Sub,
    Div,
    Neg,
    Pow,
    Exp,
    Log,
    Mean,
    MatMul,
    Reshape,
    Transpose,
    VariableSum,
)

_data_dtype = Union[int, float, list, tuple, np.ndarray, 'Variable']


def logger(message: str, type_: str = "INFO"):
    print(f"[{type_}]: {message}")


class Variable:
    """Auto grad variable class."""

    def __init__(
        self,
        data: _data_dtype,
        requires_grad: bool = False,
        grad_fn: Optional[Function] = None,
        is_leaf: bool = True,
    ):
        """
        Variable constructor.

        Args:
            data: data stored in variable
            requires_grad: if variable instance requires gradient tracking
            grad_fn: function used to arrive at variable (None if leaf)
            is_leaf: is insstance leaf variable or not
        Returns:
            None
        """

        self._data: np.ndarray = data if isinstance(data, np.ndarray) else np.array(data)
        self._grad: Optional[np.ndarray] = None
        self.grad_fn: Optional[Function] = grad_fn  # function producing self
        self.requires_grad = bool(requires_grad and is_grad_enabled())
        self.is_leaf = is_leaf
        self._version: int = 0

        self._shape = self._data.shape
        self._ndim = self._data.ndim
        self._dtype = self._data.dtype
        self._backward_hooks = []
        self._self_weakref = weakref.ref(self)  # saved_tensors avoid cycles

    def __repr__(self):
        return f"Variable(shape={self.shape}, dtype={self._data.dtype}"\
               f", requires_grad={self.requires_grad})"

    def zero_grad(
        self,
        set_to_none: bool = True,
    ) -> None:
        """
        Args:
            set_to_none: sets current grad to none if True
        """
        if set_to_none:
            self._grad = None
        else:
            self._grad = np.zeros_like(self.data)

    def detach(self) -> "Variable":
        """Returns new instance of Variable, not part of graph."""
        return type(self)(self._data.copy(), requires_grad=False, grad_fn=None, is_leaf=True)
        # return Variable(self._data.copy(), requires_grad=False, grad_fn=None, is_leaf=True)
        # FIX: remove in next version ^

    def requires_grad_(self, val=True) -> None:
        """In-place requires-grad change"""
        self.requires_grad = val

    @property
    def data(self) -> np.ndarray:
        """
        Data in Variable instance.\n
        Returned data is immutable.
        """
        v = self._data.view()
        v.setflags(write=False)
        return v

    @data.setter
    def data(self, new: _data_dtype):
        """
        WARNING; this operation will bump the version of Variable instance,
        which may result in unexpected behaviour (errors too) during backprop.
        Please use this this carefully.
        """
        self._check_inplace_ok()
        new_arr = new if isinstance(new, np.ndarray) else np.array(new)
        self._data = new_arr
        self._bump_version()

    @property
    def grad(self) -> Optional[np.ndarray]:
        return self._grad

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def numpy(self) -> np.ndarray:
        """Return as numpy.ndarray"""
        return self._data

    def tolist(self) -> List:
        """Convert into python-list"""
        return list(self._data)

    def backward(self, gradient: Optional[np.ndarray] = None):
        r"""
        Compute gradients of the graph w.r.t current tensor.
        If `gradient = None`, then its assumed to be one.

        For custom gradients, ensure that: `gradient.shape == tensor.shape`

        Args:
            gradient (Optional[np.ndarray]): gradient of tensor
        """
        if not self.requires_grad:
            raise RuntimeError("Called .backward() on a tensor that doesn't require grad.")

        # assume grad is one if not provided
        if gradient is None:
            gradient = np.ones_like(self._data)

        # build topo sort
        topo = []
        visited = set()

        def build(_tensor: "Variable"):
            if id(_tensor) in visited:
                return
            visited.add(id(_tensor))

            grad_fn = _tensor.grad_fn
            if grad_fn is not None:
                for child in grad_fn.ctx.saved_data:  # parents
                    if _isinstance(child, "Variable", "Tensor"):
                        build(child)

            topo.append(_tensor)

        build(self)

        # init the grads
        grads = {id(self): gradient.copy()}

        # use topo to propagate
        gen = (_t for _t in reversed(topo) if _t.requires_grad)
        for _tensor in gen:
            grad = grads.get(id(_tensor))
            if grad is None:
                continue

            # processing a leaf
            if _tensor.is_leaf:
                if _tensor.grad is None:
                    _tensor._grad = grad  # init grad
                else:                     # (or)
                    _tensor._grad += grad  # accumilate grad
                for hook in _tensor._backward_hooks:
                    hook(_tensor)  # call all hooks (if given)

            # processing intermediate tensors
            else:
                # version safety check (detects illegal in-place between forward and backward)
                ctx = _tensor.grad_fn.ctx
                snap = ctx.version_snapshot
                if snap is not None:
                    for parent, seen in zip(ctx.saved_data, snap):
                        if parent._version != seen:
                            raise RuntimeError(
                                f"One of the Variables needed for backward was modified in-place: "\
                                f"saved version {seen}, current version {parent._version} for: {repr(parent)}"
                            )

                grad_out = _tensor.grad_fn.backward(grad)  # op-wise backward

                # logic for any function (even custom ones, defined by user)
                if not isinstance(grad_out, tuple):
                    grad_out = (grad_out,)

                for parent, g_out in zip(ctx.saved_data, grad_out):
                    if g_out is None:
                        continue

                    # inverse-brodcasing, reduce gradient to parent shape (if needed)
                    shape = parent  # assumes parent is Tuple
                    if _isinstance(parent, "Variable", "Tensor") or isinstance(parent, np.ndarray):
                        shape = parent.shape
                    g_out = _unbroadcast(g_out, shape)

                    if id(parent) not in grads:
                        grads[id(parent)] = g_out  # put new tensor+grad in grads

                    else:
                        grads[id(parent)] += g_out  # accumulate grad

    def _bump_version(self):
        self._version += 1

    def _check_inplace_ok(self):
        if self.requires_grad and not self.is_leaf:
            raise RuntimeError("In-place modification on a non-leaf Variable that requires grad.")

    # hooks
    def register_hook(self, fn: Callable[['Variable'], None]):
        self._backward_hooks.append(fn)

    # math operators
    def __add__(self, other) -> "Variable":
        return add(self, other)

    def __radd__(self, other) -> "Variable":
        return add(other, self)

    def __mul__(self, other) -> "Variable":
        return mul(self, other)

    def __rmul__(self, other) -> "Variable":
        return mul(other, self)

    def __sub__(self, other) -> "Variable":
        return sub(self, other)

    def __rsub__(self, other) -> "Variable":
        return sub(other, self)

    def __truediv__(self, other) -> "Variable":
        return div(self, other)

    def __rtruediv__(self, other) -> "Variable":
        return div(other, self)

    def __neg__(self) -> "Variable":
        return neg(self)

    def __matmul__(self, other) -> "Variable":
        return matmul(self, other)

    def __pow__(self, exp: Union[int, float]) -> "Variable":
        return pow(self, exp)

    # variable comparisons
    def _coerce_other(self, other):
        """Helper to extract data from Variable or plain number/array."""
        if isinstance(other, Variable):  # works for subclasses too
            return other.data
        elif isinstance(other, (int, float, np.ndarray, list)):
            return other
        else:
            return NotImplementedError(f"other must be either: (int, float, np.ndarray, list), got: {type(other)}")

    def __eq__(self, other) -> bool:
        other = self._coerce_other(other)
        return np.equal(self.data, other)

    def __lt__(self, other) -> bool:
        other = self._coerce_other(other)
        return np.less(self.data, other)

    def __gt__(self, other) -> bool:
        other = self._coerce_other(other)
        return np.greater(self.data, other)

    def __le__(self, other) -> bool:
        other = self._coerce_other(other)
        return np.less_equal(self.data, other)

    def __ge__(self, other) -> bool:
        other = self._coerce_other(other)
        return np.greater_equal(self.data, other)

    # inplace operators

    def __iadd__(self, other) -> "Variable":
        self._check_inplace_ok()
        if isinstance(other, Variable):
            np.add(self._data, other._data, out=self._data, casting="unsafe")
        else:
            np.add(self._data, other, out=self._data, casting="unsafe")
        self._bump_version()
        return type(self)(self)
        return self

    def __isub__(self, other) -> "Variable":
        self._check_inplace_ok()
        if isinstance(other, Variable):
            np.subtract(self._data, other._data, out=self._data, casting="unsafe")
        else:
            np.subtract(self._data, other, out=self._data, casting="unsafe")
        self._bump_version()
        return type(self)(self)
        return self

    def __imul__(self, other) -> "Variable":
        self._check_inplace_ok()
        if isinstance(other, Variable):
            np.multiply(self._data, other._data, out=self._data, casting="unsafe")
        else:
            np.multiply(self._data, other, out=self._data, casting="unsafe")
        self._bump_version()
        return type(self)(self)

    def __itruediv__(self, other) -> "Variable":
        self._check_inplace_ok()
        if isinstance(other, Variable):
            np.true_divide(self._data, other._data, out=self._data, casting="unsafe")
        else:
            np.true_divide(self._data, other, out=self._data, casting="unsafe")
        self._bump_version()
        return type(self)(self)

    def __ipow__(self, other) -> "Variable":
        self._check_inplace_ok()
        if isinstance(other, Variable):
            np.power(self._data, other._data, out=self._data, casting="unsafe")
        else:
            np.power(self._data, other, out=self._data, casting="unsafe")
        self._bump_version()
        return type(self)(self)

    # data logic

    def __setitem__(self, key, value):
        self._check_inplace_ok()
        if isinstance(value, Variable):
            self._data[key] = value._data
        else:
            self._data[key] = value
        self._bump_version()

    def squeeze(
        self,
        dim: Optional[int] = None,
        in_place: bool = True,
    ) -> Optional["Variable"]:
        """
        Removes dims with no entries
        Args:
            dim (int): dimention to squeeze (if none, all dims are squeezed)
            in_place (bool): squeeze in-place
        Returns:
            Variable, if in_place = False
        """
        out = self.data.squeeze(axis=dim) if not in_place else self._data.squeeze(axis=dim)
        return type(self)(out)

    # ====== Tensor functions ====== #

    def reshape(self, shape: tuple) -> "Variable":
        return reshape(self, shape=shape)

    @property
    def T(self) -> "Variable":
        return transpose(self)

    def sum(self, dim: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> "Variable":
        return tensor_sum(a=self, dim=dim, keepdims=keepdims)

    def mean(self, dim: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> "Variable":
        return mean(a=self, dim=dim, keepdims=keepdims)

    def exp(self) -> "Variable":
        return exp(a=self)

    def log(self) -> "Variable":
        return log(a=self)

    # private, will be accessed in Tensor (not needed for Variable)
    def _relu(self) -> "Variable":
        return relu(a=self)

    def add_(
        self,
        other: _data_dtype,
    ):
        """
        In-place add operator.
        WARNING: using `add_` will bump-version,
        will break backward-graph.

        Args:
            other (_data_dtype): operand
        """
        self._check_inplace_ok()
        if isinstance(other, Variable):
            self._data += other._data
        else:
            self._data += other
        self._bump_version()

    def mul_(self, other):
        """
        In-place multiply operator.
        WARNING: using `mul_` will bump-version.

        Args:
            other (_data_dtype): operand
        """
        self._check_inplace_ok()
        if isinstance(other, Variable):
            self._data *= other._data
        else:
            self._data *= other
        self._bump_version()

    def zero_(self):
        """
        In-place zero operator, sets self.data to zeros.
        WARNING: using `zero_` will bump-version.

        Args:
            other (_data_dtype): operand
        """
        self._check_inplace_ok()
        self._data[...] = 0
        self._bump_version()


# Helpers for the tensor class (not accessed outside this file.)

def _isinstance(x: Any, *class_name: str):
    return x.__class__.__name__ in class_name


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce grad to the target shape (inverse of broadcasting)."""
    if grad.shape == shape:
        return grad
    # sum over leading dims
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    # sum dims of size 1
    for i, (gdim, sdim) in enumerate(zip(grad.shape, shape)):
        if sdim == 1 and gdim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def enforce_tensor(x) -> Variable:
    """
    Converts 'x' into a Tensor instance if not already.
    Used for tensor-ops.
    """
    if isinstance(x, Variable):
        return x
    return Variable(
        np.array(x, dtype=float),
        requires_grad=False,
        is_leaf=True
    )


def _wrap_forward(
    fn_cls: Type[Function],
    *parents: Variable,
    result_cls=None,
    **kwargs
) -> Variable:
    # Fresh context for function
    ctx = Context()
    versions = [p._version for p in parents]
    ctx.version_snapshot = tuple(versions)
    fn = fn_cls(ctx)

    # Compute function output
    out_data = fn.forward(*parents, **kwargs)

    # Pick output class from parents (tensor has more importance)
    if result_cls is None:
        classes = {type(p) for p in parents}
        if len(classes) == 1:
            result_cls = classes.pop()
        else:
            tensor_presedence = (type(p) for p in parents
                                 if p.__class__.__name__ == "Tesnsor")
            result_cls = next(tensor_presedence, Variable)

    # Create output as tensor
    if result_cls.__name__ == "Tensor":
        devices = [p.device for p in parents if hasattr(p, "device")]
        device = devices[-1]  # ISSUE: device presedence needs to be provided
        out = result_cls(
                data=out_data,
                requires_grad=any(p.requires_grad for p in parents),
                grad_fn=None,  # Will be assigned after
                is_leaf=False,
                device=device,  # pyright: ignore
            )
    # Create output as Variable
    else:
        out = Variable(
            data=out_data,
            requires_grad=any(p.requires_grad for p in parents),
            grad_fn=None,  # Will be assigned after
            is_leaf=False,
        )

    # Binding a grad_fn
    out.grad_fn = fn

    return out


def add(a, b):
    a = enforce_tensor(a)
    b = enforce_tensor(b)
    return _wrap_forward(Add, a, b)


def sub(a, b):
    a = enforce_tensor(a)
    b = enforce_tensor(b)
    return _wrap_forward(Sub, a, b)


def mul(a, b):
    a = enforce_tensor(a)
    b = enforce_tensor(b)
    return _wrap_forward(Mul, a, b)


def div(a, b):
    a = enforce_tensor(a)
    b = enforce_tensor(b)
    return _wrap_forward(Div, a, b)


def neg(a):
    a = enforce_tensor(a)
    return _wrap_forward(Neg, a)


def matmul(a, b):
    a = enforce_tensor(a)
    b = enforce_tensor(b)
    return _wrap_forward(MatMul, a, b)


def tensor_sum(
    a,
    dim: Optional[Union[int, tuple]] = None,
    keepdims: bool = False
):
    a = enforce_tensor(a)
    return _wrap_forward(VariableSum, a, dim=dim, keepdims=keepdims)


def mean(a, dim=None, keepdims=False):
    a = enforce_tensor(a)
    return _wrap_forward(Mean, a, dim=dim, keepdims=keepdims)


def transpose(a):
    a = enforce_tensor(a)
    return _wrap_forward(Transpose, a)


def reshape(a, shape: tuple):
    a = enforce_tensor(a)
    return _wrap_forward(Reshape, a, shape=shape)


def relu(a):
    a = enforce_tensor(a)
    return _wrap_forward(ReLU, a)


def pow(a, exponent):
    a = enforce_tensor(a)
    return _wrap_forward(Pow, a, exponent=exponent)


def exp(a):
    a = enforce_tensor(a)
    return _wrap_forward(Exp, a)


def log(a):
    a = enforce_tensor(a)
    return _wrap_forward(Log, a)


if __name__ == "__main__":
    a = Variable(30, requires_grad=True)
    b = Variable(2, requires_grad=True)
    c = a * b
    c.backward()
