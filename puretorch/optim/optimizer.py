"""
Base class for all optimizer classes, be it custom or defaults
"""

from collections import defaultdict
from abc import abstractmethod
from typing import Iterable, Any

from puretorch import nn


class Optimizer:
    """
    Base class for all optimizers

    Once initialized, following attrs are availible:
    - param_groups (list[dict[str, Any]])

    Args:
        params: iterable of PureTorch.Tensor's. Parameters to optimized
    """

    def __init__(self, params: Iterable[nn.Parameter], defaults: dict):
        """
        Initializes param_groups as a list of dicts. Where each dict
        contains a single kv pair with key `params` associated with a
        list of parameters.

        Args:
            params (Iterable): an iterable of nn.Parameter. Parameters
            to be optimized
            defaults (dict): dict containing default optimization opts
        """
        self.defaults = defaults

        self.state = defaultdict(dict)
        self.param_groups: list[dict[str, Any]] = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("Got empty parameter list.")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group: dict[str, Any]):
        """
        Add a param group to Optimizer.

        Args:
            param_group (dict)

        """
        assert isinstance(param_group, dict), f"param group must be a dict, got {
            type(param_group)
        }"

        params = param_group["params"]
        if isinstance(params, nn.Parameter):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "Optimizer params are expected to be ordered. Got a set, whose order changes bw runs..."
            )
        else:
            param_group["params"] = list(params)

        for param in param_group["params"]:
            if not isinstance(param, nn.Parameter):
                raise TypeError(f"Expected nn.Parameter, got {type(param)}.")
            if not param.is_leaf:
                raise ValueError("Optimizing a leaf Tensor is not allowed.")

            for name, default in self.defaults.items():
                param_group.setdefault(name, default)

            # Ensure unique params per param-group
            param_set = set()
            for group in self.param_groups:
                param_set.update(set(group["params"]))
            if not param_set.isdisjoint(set(param_group["params"])):
                raise ValueError("Some params appear in more than one param group")

            self.param_groups.append(param_group)

    def zero_grad(self, set_to_none: bool = False):
        """
        Sets the gradients of all the parameters given to the optimizer.
        Please note that all params in param_groups are set to zero-grad.

        Args:
            set_to_none (bool, default=False): sets the grads to None if True,
            zero otherwise.
        """
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        try:
                            param.zero_grad()
                        except AttributeError:
                            raise TypeError(
                                "Parameter does not support 'zero_grad'; ensure all parameters are "
                                "puretorch.Tensors."
                            )

    # The step method must be defined for each optimizer
    @abstractmethod
    def step(self):
        """
        Performs a single optimization step on all given params (updates the parameters)
        """
        raise NotImplementedError
