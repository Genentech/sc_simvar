"""Test the knn functions."""

from compare import compare_ndarrays
from hotspot.knn import compute_weights as hs_compute_weights
from hotspot.knn import make_weights_non_redundant as hs_make_weights_non_redundant
from hotspot.knn import neighbors_and_weights as hs_latent_neighbors_and_weights
from hotspot.knn import (
    neighbors_and_weights_from_distances as hs_distances_neighbors_and_weights,
)
from hotspot_like import hs_knn_from_latent
from pandas import DataFrame
from simulate import sim_cell_labels, sim_distances, sim_latent, sim_latent_neighbors_and_weights
from utils import timer

from sc_simvar._lib import compute_weights as rust_compute_weights
from sc_simvar._lib import make_weights_non_redundant as rust_make_weights_non_redundant
from sc_simvar._not_yet_rust import distances_neighbors_and_weights as nyr_distances_neighbors_and_weights
from sc_simvar._not_yet_rust import latent_neighbors_and_weights as nyr_latent_neighbors_and_weights


def test_compute_weights() -> None:
    """Test the compute weights function."""
    _, distances = hs_knn_from_latent(latent=sim_latent(), n_neighbors=30)

    # for JIT compilation
    hs_compute_weights(distances=distances)
    rust_compute_weights(distances=distances, n_factor=3)

    hs_weights = timer(hs_compute_weights, distances=distances, prefix="hs_")

    rust_weights = timer(rust_compute_weights, distances=distances, n_factor=3)

    compare_ndarrays(hs_weights, rust_weights)


def test_distances() -> None:
    """Test the knn distance computations."""
    distances = sim_distances()

    hs_nbrs, hs_dist = timer(
        hs_distances_neighbors_and_weights, distances, cell_index=sim_cell_labels(), prefix="hs_"
    )

    nyr_nbrs, nyr_dist = timer(nyr_distances_neighbors_and_weights, distances)
    # sv_nbrs, sv_dist = timer(rust_make_neighbors_and_weights, distances, 30, 3, "distances")

    compare_ndarrays(hs_nbrs.to_numpy(dtype="uint64"), nyr_nbrs)
    compare_ndarrays(hs_dist.to_numpy(dtype="float64"), nyr_dist)

    # compare_ndarrays(hs_nbrs.to_numpy(dtype="uint64"), sv_nbrs)
    # compare_ndarrays(hs_dist.to_numpy(dtype="float64"), sv_dist)


def test_latent() -> None:
    """Test the knn distance computations."""
    latent = sim_latent()
    latent_df = DataFrame(latent)

    hs_nbrs, hs_dist = timer(hs_latent_neighbors_and_weights, latent_df, approx_neighbors=False, prefix="hs_")

    sv_nbrs, sv_dist = timer(nyr_latent_neighbors_and_weights, latent, approx_neighbors=False)

    compare_ndarrays(hs_dist.to_numpy(dtype="float64"), sv_dist)
    compare_ndarrays(hs_nbrs.to_numpy(dtype="uint64"), sv_nbrs)


# TODO: Test tree based neighbors and weights


def test_make_weights_non_redundant() -> None:
    """Test the make weights non redundant function."""
    neighbors, weights = sim_latent_neighbors_and_weights()

    # for JIT compilation
    hs_make_weights_non_redundant(neighbors=neighbors, weights=weights)

    hs_weights = timer(hs_make_weights_non_redundant, neighbors=neighbors, weights=weights, prefix="hs_")
    rust_weights = timer(rust_make_weights_non_redundant, neighbors=neighbors, weights=weights)

    compare_ndarrays(hs_weights, rust_weights)


if __name__ == "__main__":
    test_compute_weights()
    # test_distances()
    # test_latent()
    # test_make_weights_non_redundant()
