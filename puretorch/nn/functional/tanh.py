from puretorch import Tensor


def tanh(x: Tensor) -> Tensor:
    pos = x.exp()
    neg = x.__neg__().exp()
    return (pos - neg) / (pos + neg)
