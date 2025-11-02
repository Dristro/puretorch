from typing import Tuple


class Context:
    """Context passed to forward to save variables for backward."""

    def __init__(self):
        self._saved_tensors: Tuple = ()
        self._version_snapshot: int = 0

    def save_for_backward(self, *tensors):
        """Save tensors for backward."""
        self._saved_tensors = tuple(tensors)

    @property
    def saved_tensors(self):
        """Access saved tensors for instance of Context."""
        return self._saved_tensors
