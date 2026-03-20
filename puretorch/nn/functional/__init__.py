from .utils import _as_cls_const
from .relu import relu
from .tanh import tanh
from .softmax import softmax
from .log_softmax import log_softmax
from .cross_entropy import cross_entropy

__all__ = [
    "_as_cls_const",
    "relu",
    "tanh",
    "softmax",
    "log_softmax",
    "cross_entropy"
]
