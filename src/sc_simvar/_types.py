"""Types for sc_simvar module."""

from typing import Protocol, runtime_checkable

from numpy import ndarray


@runtime_checkable
class HasToArray(Protocol):
    """Protocol for objects that can be converted to a numpy array."""

    def toarray(self) -> ndarray:
        """Convert the object to a numpy array."""
        raise NotImplementedError("Subclasses must implement this method.")
