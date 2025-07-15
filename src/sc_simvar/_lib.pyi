"""The SCSimVar package."""

from typing import Literal

from numpy import float64, str_, uint64
from numpy.typing import NDArray

# TODO: make faster and accurate, unused currently
def make_neighbors_and_weights(
    data: NDArray[float64], k: int, n_factor: float, kind: Literal["latent", "distances"]
) -> tuple[NDArray[uint64], NDArray[float64]]:
    """Make the neighbors and weights.

    Parameters
    ----------
    data : NDArray[float64]
        The data values, must be 2D.
    k : int
        The number of neighbors.
    n_factor : float
        The neighborhood factor.
    kind : Literal["latent", "distances"]
        The kind of the data values

    Returns
    -------
    tuple[NDArray[uint64], NDArray[float64]]
        The indices of the neighbors and the weights.

    """
    ...

def make_weights_non_redundant(weights: NDArray[float64], neighbors: NDArray[uint64]) -> NDArray[float64]:
    """Make the weights non redundant.

    Parameters
    ----------
    weights : NDArray[float64]
        The weights of the neighbors, must be 2D.
    neighbors : NDArray[uint64]
        The indices of the neighbors, must be 2D.

    Returns
    -------
    NDArray[float64]
        The non redundant weights.

    """
    ...

def compute_simvar(
    counts: NDArray[float64],
    neighbors: NDArray[uint64],
    weights: NDArray[float64],
    umi_counts: NDArray[float64],
    features: NDArray[str_],
    model: Literal["normal", "danb", "bernoulli", "none"] = "danb",
    centered: bool = True,
) -> tuple[
    NDArray[str_],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
]:
    """Compute the similarity variance.

    Parameters
    ----------
    counts : NDArray[float64]
        The counts of the cells, must be 2D.
    neighbors : NDArray[uint64]
        The indices of the neighbors, must be 2D.
    weights : NDArray[float64]
        The weights of the neighbors, must be 2D.
    umi_counts : NDArray[float64]
        The UMI counts of the cells, must be 1D.
    features : NDArray[object_]
        The names of the genes, must be 1D.
    model : Literal["normal", "danb", "bernoulli", "none"], optional
        The model to use, by default `"danb"`.
    centered : bool, optional
        Whether to center the counts, by default `True`.

    Returns
    -------
    NDArray[object_]
        The ordered names of the genes, is 1D.
    NDArray[float64]
        The C values of the genes, is 1D.
    NDArray[float64]
        The Z values of the genes, is 1D.
    NDArray[float64]
        The p-values of the genes, is 1D.
    NDArray[float64]
        The FDRs of the genes, is 1D.

    """
    ...

def compute_simvar_pairs_centered_cond(
    counts: NDArray[float64],
    neighbors: NDArray[uint64],
    weights: NDArray[float64],
    umi_counts: NDArray[float64],
    model: Literal["normal", "danb", "bernoulli", "none"] = "danb",
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Compute the similarity variance pairs centered conditional symmetric.

    Parameters
    ----------
    counts : NDArray[float64]
        The counts of the cells, must be 2D.
    neighbors : NDArray[uint64]
        The indices of the neighbors, must be 2D.
    weights : NDArray[float64]
        The weights of the neighbors, must be 2D.
    umi_counts : NDArray[float64]
        The UMI counts of the cells, must be 1D.
    model : Literal["normal", "danb", "bernoulli", "none"], optional
        The model to use, by default `"danb"`.

    Returns
    -------
    NDArray[float64]
        The LCP values of the genes, is 2D.
    NDArray[float64]
        The Z values of the genes, is 2D.

    """
    ...

def compute_simvar_and_pairs(
    all_counts: NDArray[float64],
    sub_counts: NDArray[float64],
    neighbors: NDArray[uint64],
    weights: NDArray[float64],
    umi_counts: NDArray[float64],
    features: NDArray[str_],
    model: Literal["normal", "danb", "bernoulli", "none"] = "danb",
    centered: bool = True,
) -> tuple[
    NDArray[str_],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
]:
    """Compute the similarity variance and pairs.

    Parameters
    ----------
    all_counts : NDArray[float64]
        The counts of all cells, must be 2D.
    sub_counts : NDArray[float64]
        Thew counts of cells to compute local correlations for, must be 2D.
    neighbors : NDArray[uint64]
        The indices of the neighbors, must be 2D.
    weights : NDArray[float64]
        The weights of the neighbors, must be 2D.
    umi_counts : NDArray[float64]
        The UMI counts of the cells, must be 1D.
    features : NDArray[object_]
        The names of the genes.
    model : Literal["normal", "danb", "bernoulli", "none"], optional
        The model to use, by default `"danb"`.
    centered : bool, optional
        Whether to center the counts, by default `True`.

    Returns
    -------
    NDArray[object_]
        The ordered names of the genes, is 1D.
    NDArray[float64]
        The C values of the genes, is 1D.
    NDArray[float64]
        The Z values of the genes, is 1D.
    NDArray[float64]
        The p-values of the genes, is 1D.
    NDArray[float64]
        The FDRs of the genes, is 1D.
    NDArray[float64]
        The LCP values of the genes, is 2D.
    NDArray[float64]
        The Z values of the genes, is 2D.

    """
    ...

# Testing

# TODO: make faster and accurate, unused currently
def knn_from_latent(latent: NDArray[float64], k: uint64) -> tuple[NDArray[uint64], NDArray[float64]]:
    """Compute the k-nearest neighbors from the latent space.

    Parameters
    ----------
    latent : NDArray[float64]
        The latent space, must be 2D.
    k : uint64
        The number of neighbors.

    Returns
    -------
    tuple[NDArray[uint64], NDArray[float64]]
        The indices of the neighbors and the weights.

    """
    ...

def knn_from_distances(distances: NDArray[float64], k: int) -> tuple[NDArray[uint64], NDArray[float64]]:
    """Compute the k-nearest neighbors from the distances.

    Parameters
    ----------
    distances : NDArray[float64]
        The distances, must be 2D.
    k : uint64
        The number of neighbors.

    Returns
    -------
    NDArray[uint64]
        The indices of the neighbors.
    NDArray[float64]
        The distances of the neighbors.

    """
    ...

def compute_weights(distances: NDArray[float64], n_factor: int) -> NDArray[float64]:
    """Compute the weights.

    Parameters
    ----------
    distances : NDArray[float64]
        The distances, must be 2D.
    n_factor : int
        The neighborhood factor.

    Returns
    -------
    NDArray[float64]
        The weights.

    """
    ...

def fit_none_model(
    gene_counts: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Fit the none model.

    Parameters
    ----------
    gene_counts : NDArray[float64]
        The gene counts, must be 1D.

    Returns
    -------
    NDArray[float64]
        The means.
    NDArray[float64]
        The variances.
    NDArray[float64]
        The x2s.

    """
    ...

def fit_bernoulli_model(
    gene_counts: NDArray[float64], umi_counts: NDArray[float64]
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Fit the bernoulli model.

    Parameters
    ----------
    gene_counts : NDArray[float64]
        The gene counts, must be 1D.
    umi_counts : NDArray[float64]
        The UMI counts, must be 1D.

    Returns
    -------
    NDArray[float64]
        The means.
    NDArray[float64]
        The variances.
    NDArray[float64]
        The x2s.

    """
    ...

def fit_danb_model(
    gene_counts: NDArray[float64], umi_counts: NDArray[float64]
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Fit the danb model.

    Parameters
    ----------
    gene_counts : NDArray[float64]
        The gene counts, must be 1D.
    umi_counts : NDArray[float64]
        The UMI counts, must be 1D.

    Returns
    -------
    NDArray[float64]
        The means.
    NDArray[float64]
        The variances.
    NDArray[float64]
        The x2s.

    """
    ...

def fit_normal_model(
    gene_counts: NDArray[float64], umi_counts: NDArray[float64]
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Fit the normal model.

    Parameters
    ----------
    gene_counts : NDArray[float64]
        The gene counts, must be 1D.
    umi_counts : NDArray[float64]
        The UMI counts, must be 1D.

    Returns
    -------
    NDArray[float64]
        The means.
    NDArray[float64]
        The variances.
    NDArray[float64]
        The x2s.

    """
    ...

def center_values(vals: NDArray[float64], mu: NDArray[float64], var: NDArray[float64]) -> NDArray[float64]:
    """Center the values.

    Parameters
    ----------
    vals : NDArray[float64]
        The values, must be 1D.
    mu : NDArray[float64]
        The means, must be 1D.
    var : NDArray[float64]
        The variances, must be 1D.

    Returns
    -------
    NDArray[float64]
        The centered values.

    """
    ...

def compute_moments_weights(
    mu: NDArray[float64], x2: NDArray[float64], neighbors: NDArray[uint64], weights: NDArray[float64]
) -> tuple[float, float]:
    """Compute the moments of the weights.

    Parameters
    ----------
    mu : NDArray[float64]
        The means, must be 1D.
    x2 : NDArray[float64]
        The x2s, must be 1D.
    neighbors : NDArray[float64]
        The indices of the neighbors, must be 2D.
    weights : NDArray[float64]
        The weights of the neighbors, must be 2D.

    Returns
    -------
    float
        The first moment.
    float
        The second moment.

    """
    ...

def local_cov_weights(vals: NDArray[float64], neighbors: NDArray[uint64], weights: NDArray[float64]) -> float:
    """Compute the local covariance of the weights.

    Parameters
    ----------
    vals : NDArray[float64]
        The values, must be 1D.
    neighbors : NDArray[float64]
        The indices of the neighbors, must be 2D.
    weights : NDArray[float64]
        The weights of the neighbors, must be 2D.

    Returns
    -------
    float
        The local covariance.

    """
    ...

def compute_node_degree(neighbors: NDArray[uint64], weights: NDArray[float64]) -> NDArray[float64]:
    """Compute the node degrees.

    Parameters
    ----------
    neighbors : NDArray[float64]
        The indices of the neighbors, must be 2D.
    weights : NDArray[float64]
        The weights of the neighbors, must be 2D.

    Returns
    -------
    NDArray[float64]
        The node degrees.

    """
    ...

def compute_local_cov_max(node_degrees: NDArray[float64], row: NDArray[float64]) -> float:
    """Compute the local covariance max.

    Parameters
    ----------
    node_degrees : NDArray[float64]
        The node degrees, must be 1D.
    row : NDArray[float64]
        The row, must be 1D.

    Returns
    -------
    float
        The local covariance max.

    """
    ...

def create_centered_counts(
    counts: NDArray[float64], model: str, umi_counts: NDArray[float64]
) -> NDArray[float64]:
    """Create the centered counts.

    Parameters
    ----------
    counts : NDArray[float64]
        The counts, must be 2D.
    model : str
        The model to use.
    umi_counts : NDArray[float64]
        The UMI counts, must be 1D.

    Returns
    -------
    NDArray[float64]
        The centered counts.

    """
    ...

def calculate_conditional_eg2(
    counts: NDArray[float64], neighbors: NDArray[uint64], weights: NDArray[float64]
) -> NDArray[float64]:
    """Calculate the conditional eg2.

    Parameters
    ----------
    counts : NDArray[float64]
        The counts, must be 2D.
    neighbors : NDArray[uint64]
        The indices of the neighbors, must be 2D.
    weights : NDArray[float64]
        The weights of the neighbors, must be 2D.

    Returns
    -------
    NDArray[float64]
        The conditional eg2.

    """
    ...

def compute_simvar_pairs_inner_centered_cond_sym(
    combo: tuple[int, int],
    counts: NDArray[float64],
    neighbors: NDArray[uint64],
    weights: NDArray[float64],
    eg2s: NDArray[float64],
) -> tuple[float, float]:
    """Compute the similarity variance pairs inner centered conditional symmetric.

    Parameters
    ----------
    combo : tuple[int, int]
        The combo.
    counts : NDArray[float64]
        The counts, must be 2D.
    neighbors : NDArray[uint64]
        The indices of the neighbors, must be 2D.
    weights : NDArray[float64]
        The weights of the neighbors, must be 2D.
    eg2s : NDArray[float64]
        The eg2s, must be 2D.

    Returns
    -------
    float
        The LCP value.
    float
        The Z value.

    """
    ...

def compute_local_cov_pairs_max(node_degrees: NDArray[float64], counts: NDArray[float64]) -> NDArray[float64]:
    """Compute the local covariance pairs max.

    Parameters
    ----------
    node_degrees : NDArray[float64]
        The node degrees, must be 1D.
    counts : NDArray[float64]
        The counts, must be 2D.

    Returns
    -------
    NDArray[float64]
        The local covariance pairs max.

    """
    ...
