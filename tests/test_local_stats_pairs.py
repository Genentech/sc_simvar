"""Test local_stats_pairs methods."""

from itertools import combinations
from typing import Tuple, cast

from compare import compare_ndarrays
from hotspot.local_stats_pairs import (
    _compute_hs_pairs_inner_centered_cond_sym as hs_compute_simvar_pairs_inner_centered_cond_sym,
)
from hotspot.local_stats_pairs import compute_hs_pairs_centered_cond as hs_compute_simvar_pairs_centered_cond
from hotspot.local_stats_pairs import compute_local_cov_pairs_max as hs_compute_local_cov_pairs_max
from hotspot.local_stats_pairs import conditional_eg2 as hs_conditional_eg2
from hotspot.local_stats_pairs import create_centered_counts as hs_create_centered_counts
from numpy import array, float64, isclose
from numpy.typing import NDArray
from pandas import DataFrame, Series
from simulate import sim_danb_counts, sim_latent_neighbors_and_weights
from utils import timer

from sc_simvar._lib import (
    calculate_conditional_eg2,
    compute_local_cov_pairs_max,
    compute_node_degree,
    compute_simvar_pairs_centered_cond,
    compute_simvar_pairs_inner_centered_cond_sym,
    create_centered_counts,
)

neighbors, weights = sim_latent_neighbors_and_weights()

df_neighbors = DataFrame(neighbors)
df_weights = DataFrame(weights)

gene_counts, umi_counts = sim_danb_counts()
umi_counts = umi_counts.astype(float64)

df_gene_counts = DataFrame(gene_counts)
s_umi_counts = Series(umi_counts)


def test_create_centered_counts() -> None:
    """Test the create_centered_counts function."""
    # for JIT compilation
    hs_create_centered_counts(gene_counts, "danb", umi_counts)

    hs_centered_counts = timer(hs_create_centered_counts, gene_counts, "danb", umi_counts, prefix="hs_")
    sv_centered_counts = timer(create_centered_counts, gene_counts, "danb", umi_counts)

    compare_ndarrays(hs_centered_counts, sv_centered_counts)


def test_conditional_eg2() -> None:
    """Test the conditional_eg2 function."""

    def hs_calculate_conditional_eg2(
        counts: NDArray[float64], neighbors: DataFrame, weights: DataFrame
    ) -> NDArray[float64]:
        """Calculate the conditional hs eg2s."""
        return array(
            [
                hs_conditional_eg2(
                    row,
                    neighbors.to_numpy(),
                    weights.to_numpy(),
                )
                for row in counts
            ],
            dtype=float64,
        )

    # for JIT compilation
    hs_calculate_conditional_eg2(gene_counts[:2, :], df_neighbors, df_weights)

    hs_eg_2s = timer(hs_calculate_conditional_eg2, gene_counts, df_neighbors, df_weights, prefix="hs_")

    sv_eg_2s = timer(calculate_conditional_eg2, gene_counts, neighbors, weights)

    compare_ndarrays(hs_eg_2s, sv_eg_2s)


def test_compute_simvar_pairs_inner_centered_cond_sym() -> None:
    """Test the compute_simvar_pairs_inner_centered_cond_sym function."""
    eg2s = calculate_conditional_eg2(gene_counts, neighbors, weights)

    pair = next(combinations(range(gene_counts.shape[0]), 2))

    # for JIT compilation
    hs_compute_simvar_pairs_inner_centered_cond_sym(pair, gene_counts, neighbors, weights, eg2s)
    compute_simvar_pairs_inner_centered_cond_sym(pair, gene_counts, neighbors, weights, eg2s)

    hs_lcp, hs_z = cast(
        Tuple[float64, float64],
        timer(
            hs_compute_simvar_pairs_inner_centered_cond_sym,
            pair,
            gene_counts,
            neighbors,
            weights,
            eg2s,
        ),
    )

    sv_lcp, sv_z = timer(
        compute_simvar_pairs_inner_centered_cond_sym,
        pair,
        gene_counts,
        neighbors,
        weights,
        eg2s,
    )

    assert isclose(hs_lcp, sv_lcp)
    assert isclose(hs_z, sv_z)


def test_compute_local_cov_pairs_max() -> None:
    """Test the compute_local_cov_pairs_max function."""
    node_degrees = compute_node_degree(neighbors, weights)

    # for JIT compilation
    hs_compute_local_cov_pairs_max(node_degrees, gene_counts)

    hs_lcp = timer(
        hs_compute_local_cov_pairs_max,
        node_degrees,
        gene_counts,
    )

    sv_lcp = timer(
        compute_local_cov_pairs_max,
        node_degrees,
        gene_counts,
    )

    compare_ndarrays(hs_lcp, sv_lcp)


def test_compute_simvar_pairs_centered_cond() -> None:
    """Test the compute_simvar_pairs_centered_cond function."""
    # NOTE: no advantage to running pre for JIT comp

    hs_lcps, hs_zs = timer(
        hs_compute_simvar_pairs_centered_cond,
        df_gene_counts,
        df_neighbors,
        df_weights,
        s_umi_counts,
        "danb",
    )

    sv_lcps, sv_zs = timer(
        compute_simvar_pairs_centered_cond,
        gene_counts,
        neighbors,
        weights,
        umi_counts,
        "danb",
    )

    compare_ndarrays(hs_lcps.to_numpy(), sv_lcps)
    compare_ndarrays(hs_zs.to_numpy(), sv_zs)


if __name__ == "__main__":
    # test_create_centered_counts()
    # test_conditional_eg2()
    # test_compute_simvar_pairs_inner_centered_cond_sym()
    # test_compute_local_cov_pairs_max()
    test_compute_simvar_pairs_centered_cond()
