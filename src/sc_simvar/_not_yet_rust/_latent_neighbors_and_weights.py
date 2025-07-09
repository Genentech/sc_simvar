"""Compute the knn graph from a latent space."""

from typing import Any, cast

from hotspot.knn import compute_weights
from numpy import float64, uint64
from numpy.typing import NDArray
from pynndescent import NNDescent  # type: ignore
from sklearn.neighbors import NearestNeighbors


def latent_neighbors_and_weights(
    latent: NDArray[float64],
    n_neighbors: int = 30,
    neighborhood_factor: int = 3,
    approx_neighbors: bool = True,
) -> tuple[NDArray[uint64], NDArray[float64]]:
    """Compute the knn graph from a latent space.

    Parameters
    ----------
    latent : NDArray[float64]
        The latent space.
    n_neighbors : int, optional
        The number of neighbors to compute. Defaults to 30.
    neighborhood_factor : int, optional
        The neighborhood factor. Defaults to 3.
    approx_neighbors : bool, optional
        Whether to approximate the neighbors. Defaults to True.

    Returns
    -------
    NDArray[uint64]
        The neighbors.
    NDArray[float64]
        The weights.

    """
    if approx_neighbors:
        index = NNDescent(latent, n_neighbors=n_neighbors + 1)
        neighbors, dist = cast(tuple[NDArray[Any], NDArray[Any]], index.neighbor_graph)
        # pynndescent first neighbor is self, unlike sklearn
        neighbors, dist = neighbors[:, 1:], dist[:, 1:]
    else:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(latent)

        dist, neighbors = cast(tuple[NDArray[Any], NDArray[Any]], nbrs.kneighbors())

    neighbors = neighbors.astype("uint64")
    dist = dist.astype("float64")
    weights = compute_weights(dist, neighborhood_factor)

    return neighbors, weights
