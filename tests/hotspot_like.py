"""Functions for hotspot functionality that is functionized in sc_simvar."""

from typing import Tuple

from numpy import float64, uint64
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors


def hs_knn_from_distances(
    distances: NDArray[float64], n_neighbors: int = 30
) -> Tuple[NDArray[uint64], NDArray[float64]]:
    """Compute the knn from distances as hotspot does.

    Parameters
    ----------
    distances : NDArray[float64]
        The distances, must be 2D.
    n_neighbors : int
        The number of neighbors.

    Returns
    -------
    NDArray[uint64]
        The indices of the neighbors.
    NDArray[float64]
        The distances of the neighbors.

    """
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="brute", metric="precomputed"
    ).fit(distances)  # type: ignore
    dist, ind = nbrs.kneighbors()  # type: ignore

    return ind.astype(uint64), dist  # type: ignore


def hs_knn_from_latent(
    latent: NDArray[float64], n_neighbors: int = 30
) -> Tuple[NDArray[uint64], NDArray[float64]]:
    """Compute the knn from the latent space as hotspot does.

    Parameters
    ----------
    latent : NDArray[float64]
        The latent space, must be 2D.
    n_neighbors : int
        The number of neighbors.

    Returns
    -------
    NDArray[uint64]
        The indices of the neighbors.
    NDArray[float64]
        The distances of the neighbors.

    """
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="ball_tree", metric="euclidean"
    ).fit(latent)  # type: ignore
    dist, ind = nbrs.kneighbors()  # type: ignore

    return ind.astype(uint64), dist  # type: ignore
