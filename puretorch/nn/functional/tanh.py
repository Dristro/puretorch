from puretorch import Tensor


def tanh(x: Tensor) -> Tensor:
    pos = x.exp()
    neg = x.__neg__().exp()
    out: Tensor = (pos - neg) / (pos + neg)  # pyright: ignore
    return out
