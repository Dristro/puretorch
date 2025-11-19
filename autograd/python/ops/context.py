from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from variable import Variable

from typing import Tuple, Any


def _isinstance(x: Any, class_name: str):
    return x.__class__.__name__ == class_name


class Context:
    """Context passed to forward to save variables for backward."""

    def __init__(self):
        """
        Init Context. No args required.
        saved_tensors and version_snapshot are empty tuples by default.
        """
        self._saved_tensors: Tuple = ()
        self._version_snapshot: Tuple[int, ...] = ()

    @property
    def version_snapshot(self) -> Tuple[int, ...]:
        """Returns the `version_snapshot` of `Context`"""
        return self._version_snapshot

    @version_snapshot.setter
    def version_snapshot(self, value: Tuple[int, ...]):
        """
        Manual override for `version_snapshot`.\n
        Will be used for Variable.backward()

        WARNING: don't touch this unless you know what your doing
                 updating this property may result in broken backward().
        """
        self._version_snapshot = value

    def version_snapshot_(self, value: Tuple[int, ...]):
        """
        Manual override for `version_snapshot`.\n
        Will be used for Variable.backward()

        WARNING: don't touch this unless you know what your doing
                 updating this property may result in broken backward().
        """
        self._version_snapshot = value

    def save_for_backward(self, *data: Any):
        """
        Save data/information relevant for backward.
        Args:
            *data (Any): data to store for backward
        """
        """Save data for backward."""
        self._saved_tensors = tuple(data)
        versions = [d._version for d in data if _isinstance(d, "Variable")]
        self.version_snapshot_(tuple(versions))

    @property
    def saved_data(self):
        """
        Access saved data for instance of Context.\n
        This property does not have static-ordering,\n
        the order is defined by the way you store info.

        For example:
        ```python
        # Saving some data di
        ctx.save_for_backward(d1, d2, d3, ..., dn)

        # Accessing data (same order as saved)
        d1, d2, d3, ..., dn = ctx.saved_for_backward
        ```
        """
        return self._saved_tensors

    def __repr__(self) -> str:
        return f"Context(saved_data={self.saved_data},\n"\
               f"\tversion_snapshot={self.version_snapshot}\n"\
               "\t)"
