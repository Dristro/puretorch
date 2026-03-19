from puretorch import Tensor
from . import softmax


def log_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    return softmax(logits, dim=dim).log()
