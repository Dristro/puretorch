from puretorch import Tensor
from . import _as_cls_const


def softmax(logits: Tensor, dim: int = -1) -> Tensor:
    # Numerically stable softmax: subtract per-row max
    shift_np = logits.data.max(axis=dim, keepdims=True)
    shift = _as_cls_const(shift_np, logits)
    exps = (logits - shift).exp()
    denom = exps.sum(dim=dim, keepdims=True)
    return exps / denom
