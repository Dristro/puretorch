from puretorch import Tensor
from .utils import _as_cls_const


def softmax(logits: Tensor, dim: int = -1) -> Tensor:
    shift_np = logits.data.max(axis=dim, keepdims=True)
    shift = _as_cls_const(shift_np, logits)
    exps = (logits - shift).exp()
    denom = exps.sum(dim=dim, keepdims=True)
    out: Tensor = exps / denom  # pyright: ignore
    return out
