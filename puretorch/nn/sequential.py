from puretorch import Tensor, nn


class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module, name: str = "Sequential model"):
        """
        Creates a new Sequential instance with the given layers.

        Args:
            layers: List of layers to propagate during the forward pass
            name: Name of the model
        """
        super().__init__()
        for i, m in enumerate(modules):
            self.add_module(f"{str(m.__class__)}{str(i)}", m)
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass on the input Tensor and returns a Tensor
        with the output of the model.

        Args:
            x: Tensor to propagate on.

        Returns:
            A Tensor with the output of the model on the input Tensor (x).
        """
        for m in self.children():
            x = m(x)
        return x

    def __repr__(self):
        return "[TODO] add this later, not that important"
