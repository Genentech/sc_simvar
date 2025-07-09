"""Simulate data for tests."""

from config import GENE_P, MEAN, MID, N_CELLS, N_DIMS, N_GENES, SCALE, SIZE
from numpy import array, exp, float64, floor, log, random, uint64, zeros
from numpy.random import normal, rand
from numpy.typing import NDArray
from pandas import Index, Series
from scipy.spatial.distance import pdist, squareform

from sc_simvar._not_yet_rust import latent_neighbors_and_weights


def sim_distances(n_cells: int = N_CELLS, n_dims: int = N_DIMS) -> NDArray[float64]:
    """Simulate distances.

    Parameters
    ----------
    n_cells : int, optional
        The number of cells, by default N_CELLS
    n_dims : int, optional
        The number of dimensions, by default N_DIMS

    Returns
    -------
    NDArray[float64]
        The distances.

    """
    # Generate random points
    points = rand(n_cells, n_dims)

    # Calculate pairwise distances
    distances = pdist(points)

    # Convert to a square matrix
    return squareform(distances)


def sim_latent(n_cells: int = N_CELLS, n_dims: int = N_DIMS) -> NDArray[float64]:
    """Simulate latent space.

    Parameters
    ----------
    n_cells : int, optional
        The number of cells, by default N_CELLS
    n_dims : int, optional
        The number of dimensions, by default N_DIMS

    Returns
    -------
    NDArray[float64]
        The latent space.

    """
    return normal(size=(n_cells, n_dims))


def sim_cell_labels(n_cells: int = N_CELLS) -> Index:
    """Simulate cell labels.

    Returns
    -------
    list[str]
        The cell labels.

    """
    return Index([f"Cell_{i}" for i in range(n_cells)])


def sim_gene_labels(n_genes: int = N_GENES) -> Series:
    """Simulate gene labels.

    Returns
    -------
    list[str]
        The gene labels.

    """
    return Series([f"Gene_{i}" for i in range(n_genes)])


def sim_latent_neighbors_and_weights(
    n_cells: int = N_CELLS, n_dims: int = N_DIMS
) -> tuple[NDArray[uint64], NDArray[float64]]:
    """Simulate neighbors and weights.

    Returns
    -------
    tuple[NDArray[uint64], NDArray[float64]]
        The neighbors and weights.

    """
    neighbors, weights = latent_neighbors_and_weights(sim_latent(n_cells, n_dims), approx_neighbors=False)

    return (neighbors, weights)


def sim_umi_counts(mid: int = MID, scale: int = SCALE, n_cells: int = N_CELLS) -> NDArray[uint64]:
    """Simulate umi counts.

    Returns
    -------
    NDArray[uint64]
        The umi counts.

    """
    mu = log(mid)
    sd = log(mid + scale) - mu

    vals = normal(loc=mu, scale=sd, size=n_cells)
    return floor(exp(vals)).astype(uint64)


def sim_danb_counts(
    mid: int = MID,
    scale: int = SCALE,
    n_genes: int = N_GENES,
    n_cells: int = N_CELLS,
    mean: int = MEAN,
    size: int = SIZE,
) -> tuple[NDArray[float64], NDArray[uint64]]:
    """Simulate danb counts.

    Returns
    -------
    NDArray[float64]
        The danb counts.
    NDArray[float64]
        The umi counts.

    """
    umi_counts = sim_umi_counts(mid, scale, n_cells)

    vals = zeros((n_genes, n_cells), dtype=float64)
    for i, val in enumerate(umi_counts / umi_counts.mean()):
        mean_i = mean * val

        sub_p = 1 - mean_i / (mean_i * (1 + mean_i / size))
        n = mean_i * (1 - sub_p) / sub_p

        vals[:, i] = random.negative_binomial(n=n, p=1 - sub_p, size=n_genes)

    return vals, umi_counts


def _sim_bernoulli_row(
    umi_counts: NDArray[uint64], gene_p: int = GENE_P, n_cells: int = N_CELLS
) -> NDArray[float64]:
    """Simulate bernoulli counts.

    Returns
    -------
    NDArray[float64]
        The bernoulli counts.

    """
    detect_p = 1 - (1 - gene_p / 10000) ** umi_counts

    return (rand(n_cells) < detect_p).astype(float64)


def sim_bernoulli_counts(
    mid: int = MID, scale: int = SCALE, n_cells: int = N_CELLS, n_genes: int = N_GENES
) -> tuple[NDArray[float64], NDArray[uint64]]:
    """Simulate bernoulli counts.

    Returns
    -------
    NDArray[float64]
        The bernoulli counts.
    NDArray[float64]
        The umi counts.

    """
    umi_counts = sim_umi_counts(mid, scale, n_cells)
    return (
        array([_sim_bernoulli_row(umi_counts) for _ in range(n_genes)]),
        umi_counts,
    )
