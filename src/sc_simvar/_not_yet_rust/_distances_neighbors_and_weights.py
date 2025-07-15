"""Compute the knn graph from a distance matrix."""

from typing import Any, cast
from warnings import warn

from numpy import float64, uint64
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from sc_simvar._lib import compute_weights

# from ._compte_weights import compute_weights


def distances_neighbors_and_weights(
    distances: NDArray[float64], n_neighbors: int = 30, neighborhood_factor: int = 3
) -> tuple[NDArray[uint64], NDArray[float64]]:
    """Compute the knn graph from a distance matrix.

    Parameters
    ----------
    distances : NDArray[float64]
        The distances.
    n_neighbors : int, optional
        The number of neighbors to compute. Defaults to 30.
    neighborhood_factor : int, optional
        The neighborhood factor. Defaults to 3.

    Returns
    -------
    NDArray[uint64]
        The neighbors.
    NDArray[float64]
        The weights.

    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute", metric="precomputed").fit(  # type: ignore
        distances
    )

    try:
        dist, neighbors = cast(tuple[NDArray[Any], NDArray[Any]], nbrs.kneighbors())  # type: ignore
    # already is a neighbors graph
    except ValueError:
        num_nbrs = (distances[0] > 0).sum()

        warn(
            "Provided cell-cell distance graph is likely a "
            f"{num_nbrs}-neighbors graph. Trying as precomputed neighbors."
        )

        dist, neighbors = cast(
            tuple[NDArray[Any], NDArray[Any]],
            nbrs.kneighbors(n_neighbors=num_nbrs - 1),  # type: ignore
        )

    return neighbors.astype("uint64"), compute_weights(dist.astype("float64"), int(neighborhood_factor))
