import numpy as np

from typing import Literal

from puretorch import Tensor
from . import log_softmax


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    ignore_idx: int = -100,
    weight: list | np.ndarray | Tensor | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> Tensor:
    """
    Cross Entropy loss over logits for targets.
    Targets are expected to be ordinal, i.e. index of
    class label where true.

    Args:
        logits (Tensor): logits tensor, shape [B, n_classes]
        targets (Tensor): targets tensor, shape [B,]
        ignore_idx (int, default=-100): target value ignored for loss
        weight (list | np.ndarray | Tensor | None, default=None): rescale
            loss for given class. Expected shape [n_classes], if `None`,
            then all classes are equally weighted (as 1).
        reduction ("mean", "sum", default="mean"): reduction applied to
            result.

    Returns:
        loss (Tensor): loss value as Tensor
    """
    assert logits.shape[0] == targets.shape[0], (
        f"Incorrect batch size. Expected logits and targets to have "
        f"batch-first and of same size. Got logits {logits.shape} and "
        f"targets {targets.shape}."
    )

    # ignore idx
    mask = targets.data != ignore_idx
    targets[~mask] = 0
    mask = Tensor(mask.astype(float))

    # log-probs and nll
    log_probs = log_softmax(logits, dim=-1)
    nll = -log_probs[(np.arange(len(targets)), targets)]

    # loss
    if weight is not None:
        if not isinstance(weight, Tensor):
            weight = Tensor(weight)
        weight.requires_grad_(False)
        wei = weight[targets]
        loss = nll * wei * mask
    else:
        loss = nll * mask

    # reduction
    if reduction.lower() == "mean":
        loss = loss.sum() / mask.sum()
    elif reduction.lower() == "sum":
        loss = loss.sum()
    else:
        raise ValueError(f"Support reductions are [mean, sum], got: {reduction}")

    return loss  # pyright: ignore
