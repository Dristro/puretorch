import numpy as np
from puretorch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        """
        Learnable-linear layer.
        Applies: y = wx+b
        Args:
            in_features (int): input size/dim
            out_features (int): output size/dim
            bias (bool): apply learable-bias term if True
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weights (normalized, -1 to 1)
        weights_data = np.random.randn(out_features, in_features)
        weights_norm = np.linalg.norm(weights_data)
        epsilon = 1e-8
        if weights_norm != 0:
            weights_data /= (weights_norm + epsilon)
        self.weights = nn.Parameter(weights_data, requires_grad=True)

        # Bias (normalized, -1 to 1)
        if bias:
            bias_data = np.random.randn(out_features)
            bias_norm = np.linalg.norm(bias_data)
            if bias_norm != 0:
                bias_data /= (bias_norm + epsilon)
            self.bias = nn.Parameter(bias_data, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of nn.Linear

        Args:
            x (Tensor): input to layer
        Returns:
            Tensor: tensor after applying weights and bias
        """
        # assert isinstance(x, Tensor), "Input x must be a Tensor."
        assert x.data.shape[-1] == self.in_features, (
            f"Expected input features {self.in_features}, "
            f"but got {x.data.shape[-1]}."
        )
        out = x @ self.weights.T
        if self.bias is not None:
            out = out + self.bias
        return out  # pyright: ignore

    # def parameters(self):
    #     yield self.weights
    #     if self.bias is not None:
    #         yield self.bias

    # def __call__(self, x):
    #     return self.forward(x)
