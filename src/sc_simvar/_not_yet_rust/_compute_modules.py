"""Assign modules from the gene pair-wise Z-scores."""

from typing import Any, cast

from numpy import fill_diagonal, float64, integer, nonzero, sort, zeros
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from ._assign_modules import assign_modules
from ._assign_modules_core import assign_modules_core
from ._calc_mean_dists import calc_mean_dists
from ._sort_linkage import sort_linkage


def compute_modules(
    Z_scores: DataFrame,
    min_gene_threshold: int = 10,
    fdr_threshold: float | None = None,
    z_threshold: float | None = None,
    core_only: bool | None = False,
) -> tuple["Series[int]", NDArray[Any]]:
    """Assign modules from the gene pair-wise Z-scores.

    Parameters
    ----------
    Z_scores : DataFrame
        Local correlations between genes
    min_gene_threshold : int, optional
        Minimum number of genes to create a module. Default is 10.
    fdr_threshold : float, optional
        Used to determine minimally significant z_score. Default is None.
    z_threshold : float, optional
        Used to determine minimally significant z_score. Default is None.
    core_only : bool, optional
        Whether or not to assign unassigned genes to a module. Default is False.

    Returns
    -------
    modules: Series[int]
        maps gene id to module id
    linkage: NDArray
        Linkage matrix in the format used by scipy.cluster.hierarchy.linkage

    """
    if z_threshold is None:
        if fdr_threshold is None:
            raise ValueError("If z_threshold is None, fdr_threshold must be provided.")

        allZ = squareform(  # just in case slightly not symmetric
            Z_scores.values / 2 + Z_scores.values.T / 2
        )
        allZ = sort(allZ.astype(float64, copy=False))
        allP = cast(NDArray[float64], norm.sf(allZ))

        allP_c = multipletests(allP, method="fdr_bh")[1]

        ii = nonzero(allP_c < fdr_threshold)[0]

        if ii.size > 0:
            i = ii[0]
            if not isinstance(i, integer):
                raise ValueError("This should not happen.")

            z_threshold = float(allZ[i])
        else:
            z_threshold = float(allZ[-1] + 1)

    # Compute the linkage matrix
    dd = Z_scores.copy().values
    fill_diagonal(dd, 0)
    condensed = squareform(dd) * -1
    offset = condensed.min() * -1
    condensed += offset
    Z = cast(NDArray[Any], linkage(condensed, method="average"))

    # Linkage -> Modules
    if core_only:
        out_clusters = assign_modules_core(
            Z,
            offset=offset,
            MIN_THRESHOLD=min_gene_threshold,
            leaf_labels=Z_scores.index,
            Z_THRESHOLD=z_threshold,
        )
    else:
        out_clusters = assign_modules(
            Z,
            offset=offset,
            MIN_THRESHOLD=min_gene_threshold,
            leaf_labels=Z_scores.index,
            Z_THRESHOLD=z_threshold,
        )

    # Sort the leaves of the linkage matrix (for plotting)
    mean_dists = zeros(Z.shape[0])
    calc_mean_dists(Z, Z.shape[0] - 1, mean_dists)
    linkage_out = Z.copy()
    sort_linkage(linkage_out, Z.shape[0] - 1, mean_dists)

    out_clusters.name = "Module"

    return out_clusters, linkage_out
