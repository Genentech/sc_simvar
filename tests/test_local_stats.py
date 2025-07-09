"""Test local stats functions."""

from compare import compare_data_frames, compare_ndarrays
from hotspot.local_stats import center_values as hs_center_values
from hotspot.local_stats import compute_hs as hs_compute_simvar
from hotspot.local_stats import compute_local_cov_max as hs_compute_local_cov_max
from hotspot.local_stats import compute_moments_weights as hs_compute_moments_weights
from hotspot.local_stats import compute_node_degree as hs_compute_node_degree
from hotspot.local_stats import local_cov_weights as hs_local_cov_weights
from numpy import array, float64, isclose
from numpy.typing import NDArray
from pandas import DataFrame, Series
from simulate import sim_bernoulli_counts, sim_gene_labels, sim_latent_neighbors_and_weights
from utils import timer

from sc_simvar._lib import (
    center_values,
    compute_local_cov_max,
    compute_moments_weights,
    compute_node_degree,
    compute_simvar,
    fit_bernoulli_model,
    local_cov_weights,
)

neighbors, weights = sim_latent_neighbors_and_weights()

df_neighbors = DataFrame(neighbors)
df_weights = DataFrame(weights)

gene_counts, umi_counts = sim_bernoulli_counts()
umi_counts = umi_counts.astype(float64)

s_umi_counts = Series(umi_counts)

gene_labels = sim_gene_labels()
a_gene_labels = array(gene_labels, dtype="U25")


def test_center_values() -> None:
    """Test the center_values function."""
    for row in gene_counts:
        sv_means, sv_vars, sv_x2s = fit_bernoulli_model(row, umi_counts)  # type: ignore

        # for JIT compilation
        hs_center_values(sv_means, sv_vars, sv_x2s)
        center_values(sv_means, sv_vars, sv_x2s)

        hs_centered_means = timer(hs_center_values, sv_means, sv_vars, sv_x2s, prefix="hs_")
        sv_centered_means = timer(center_values, sv_means, sv_vars, sv_x2s)

        compare_ndarrays(hs_centered_means, sv_centered_means)

        break


def test_compute_moments_weights() -> None:
    """Test the compute_moments_weights function."""
    for row in gene_counts:
        sv_means, sv_vars, sv_x2s = fit_bernoulli_model(row, umi_counts)  # type: ignore
        sv_centered_means = center_values(sv_means, sv_vars, sv_x2s)

        # for JIT compilation
        hs_compute_moments_weights(sv_centered_means, sv_x2s, neighbors, weights)
        compute_moments_weights(sv_centered_means, sv_x2s, neighbors, weights)

        hs_eg, hs_eg_2 = timer(
            hs_compute_moments_weights, sv_centered_means, sv_x2s, neighbors, weights, prefix="hs_"
        )
        sv_eg, sv_eg_2 = timer(compute_moments_weights, sv_centered_means, sv_x2s, neighbors, weights)

        assert isclose(hs_eg, sv_eg)
        assert isclose(hs_eg_2, sv_eg_2)

        break


def test_local_cov_weights() -> None:
    """Test the local_cov_weights function."""
    for row in gene_counts:
        sv_means, sv_vars, sv_x2s = fit_bernoulli_model(row, umi_counts)  # type: ignore
        sv_centered_means = center_values(sv_means, sv_vars, sv_x2s)

        # for JIT compilation
        hs_local_cov_weights(sv_centered_means, neighbors, weights)
        local_cov_weights(sv_centered_means, neighbors, weights)

        hs_g = timer(hs_local_cov_weights, sv_centered_means, neighbors, weights, prefix="hs_")
        sv_g = timer(local_cov_weights, sv_centered_means, neighbors, weights)

        assert isclose(hs_g, sv_g)

        break


def test_compute_node_degree() -> None:
    """Test the compute_node_degree function."""
    # for JIT compilation
    hs_compute_node_degree(neighbors, weights)

    hs_node_degrees = timer(hs_compute_node_degree, neighbors, weights, prefix="hs_")

    sv_node_degrees = timer(compute_node_degree, neighbors, weights)

    compare_ndarrays(hs_node_degrees, sv_node_degrees)


def test_compute_local_cov_max() -> None:
    """Test the compute_local_cov_max function."""
    node_degrees = compute_node_degree(neighbors, weights)

    row: NDArray[float64]
    for row in gene_counts:
        # for JIT compilation
        hs_compute_local_cov_max(node_degrees, row)

        hs_g = timer(hs_compute_local_cov_max, node_degrees, row, prefix="hs_")
        sv_g = timer(compute_local_cov_max, node_degrees, row)

        assert isclose(hs_g, sv_g)

        break


def test_compute_simvar_danb() -> None:
    """Test the compute_simvar function."""
    # for JIT compilation
    hs_compute_simvar(gene_counts, df_neighbors, df_weights, s_umi_counts, "danb", gene_labels, True)

    hs_simvar = timer(
        hs_compute_simvar,
        gene_counts,
        df_neighbors,
        df_weights,
        s_umi_counts,
        "normal",
        gene_labels,
        True,
    )

    sv_simvar = timer(
        compute_simvar, gene_counts, neighbors, weights, umi_counts, a_gene_labels, "normal", True
    )

    sv_simvar = DataFrame(dict(zip(["C", "Z", "Pval", "FDR"], sv_simvar[1:])), index=sv_simvar[0])
    sv_simvar.index.name = "Gene"

    compare_data_frames(hs_simvar, sv_simvar)


if __name__ == "__main__":
    test_compute_node_degree()
    test_center_values()
    test_local_cov_weights()
    test_compute_moments_weights()
    test_compute_local_cov_max()
    test_compute_simvar_danb()
