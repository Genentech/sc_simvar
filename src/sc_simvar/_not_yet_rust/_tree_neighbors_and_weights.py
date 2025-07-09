"""Compute the knn graph from a tree."""

from numpy import float64, ones_like, uint64, zeros
from numpy.random import rand
from numpy.typing import NDArray
from pandas import Series
from tqdm import tqdm

from ._tree_node import TreeNode


def _search(current_node: TreeNode, previous_node: TreeNode | None, distance: float) -> dict[str, float]:
    """Search the tree for the nearest neighbors.

    Parameters
    ----------
    current_node : TreeNode
        The current node.
    previous_node : TreeNode | None
        The previous node, if any.
    distance : float
        The distance to the current node.

    Returns
    -------
    Dict[str, float]
        A `dict` where keys are the labels of the neighbors and values are the
        distances.

    """
    if current_node.is_root():
        nodes_to_search = current_node.children
    else:
        nodes_to_search = [*current_node.children, current_node.up]

    nodes_to_search = [x for x in nodes_to_search if x is not previous_node]

    if len(nodes_to_search) == 0:
        return {current_node.name: distance}

    result: dict[str, float] = {}
    for new_node in nodes_to_search:
        result.update(_search(new_node, current_node, distance + 1))

    return result


def _knn(leaf: TreeNode, k: int) -> list[str]:
    """Compute the k nearest neighbors of a leaf.

    Parameters
    ----------
    leaf : TreeNode
        The leaf.
    k : int
        The number of neighbors to find.

    Returns
    -------
    List[str]
        The labels of the neighbors.

    """
    dists = Series(_search(leaf, None, 0))
    # to break ties randomly
    dists = dists + rand(len(dists)) * 0.9  # type: ignore

    return dists.sort_values().index[0:k].to_list()  # type: ignore


def tree_neighbors_and_weights(
    tree: TreeNode, cell_labels: list[str], n_neighbors: int = 30
) -> tuple[NDArray[uint64], NDArray[float64]]:
    """Compute the knn graph from a tree.

    Number of leaves must equal number of cells.

    Parameters
    ----------
    tree: ete3.TreeNode
        The root of the tree.
    cell_labels: List[str]
        The labels of the cells.
    n_neighbors: int, optional
        Number of neighbors to find. Defaults to 30

    Returns
    -------
    NDArray[uint64]
        The neighbors.
    NDArray[float64]
        The weights.

    """
    all_leaves = [x for x in tree if x.is_leaf()]
    all_neighbors = {leaf.name: _knn(leaf, n_neighbors) for leaf in tqdm(all_leaves)}
    label_index_map = {c: i for i, c in enumerate(cell_labels)}

    neighbors = zeros((len(all_neighbors), n_neighbors), dtype="uint64")
    for label, nbrs in all_neighbors.items():
        neighbors[label_index_map[label], :] = [label_index_map[nbr_label] for nbr_label in nbrs]

    return neighbors.astype("uint64"), ones_like(neighbors, dtype="float64")
