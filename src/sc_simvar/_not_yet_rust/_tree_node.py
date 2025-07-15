"""The TreeNode class is imported from ete3 if available, otherwise a dummy class is used for type checking."""

from typing import TYPE_CHECKING, Any, Protocol


class _DummyTreeNode(Protocol):
    """Dummy class for type checking."""

    @property
    def name(self) -> str:
        """Return the name."""
        ...

    @property
    def children(self) -> Any:
        """Return the children."""
        ...

    @property
    def up(self) -> Any:
        """Return the parent."""
        ...

    def is_root(self) -> bool:
        """Return True if the node is the root."""
        ...

    def is_leaf(self) -> bool:
        """Return True if the node is a leaf."""
        ...

    def __next__(self) -> "_DummyTreeNode":
        """Return an iterator."""
        ...

    def __iter__(self) -> "_DummyTreeNode":
        """Return an iterator."""
        ...


if TYPE_CHECKING:
    TreeNode = _DummyTreeNode
else:
    try:
        from ete3 import TreeNode  # type: ignore
    except ImportError:
        TreeNode = _DummyTreeNode  # type: ignore
