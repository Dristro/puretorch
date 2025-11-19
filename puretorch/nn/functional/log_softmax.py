from puretorch import Tensor
from .softmax import softmax


def log_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    out: Tensor = softmax(logits, dim=dim).log()  # pyright: ignore
    return out
