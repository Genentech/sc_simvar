"""Compute scores for each cell in the module using PCA on smoothed counts."""

from typing import Literal

from numpy import float64, uint64, zeros_like
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from ._create_centered_counts_row import create_centered_counts_row
from ._neighbor_smoothing_row import neighbor_smoothing_row


def compute_scores(
    counts_sub: NDArray[float64],
    model: Literal["bernoulli", "danb", "normal", "none"],
    num_umi: NDArray[float64],
    neighbors: NDArray[uint64],
    weights: NDArray[float64],
) -> NDArray[float64]:
    r"""Compute scores for each cell in the module using PCA on smoothed counts.

    Parameters
    ----------
    counts_sub : NDArray[float64]
        Row-subset of counts matrix with genes in the module
    model : Literal["normal", "danb", "bernoulli", "none"], optional
        The model to use, by default `"danb"`.

        Options:\n
            'danb': Depth-Adjusted Negative Binomial\n
            'bernoulli': Models probability of detection\n
            'normal': Depth-Adjusted Normal\n
            'none': Assumes data has been pre-standardized
    num_umi : NDArray[float64]
        Number of UMIs per cell
    neighbors : NDArray[float64]
        Nearest neighbors matrix
    weights : NDArray[float64]
        Weights for neighbors.

    Returns
    -------
    NDArray[float64]
        Scores for each cell in the module

    """
    cc_smooth = zeros_like(counts_sub, dtype=float64)

    for i in range(counts_sub.shape[0]):
        counts_row = counts_sub[i, :]
        # TODO: replace with rust implementation (needs some work?)
        centered_row = create_centered_counts_row(counts_row, model, num_umi)
        smooth_row = neighbor_smoothing_row(centered_row, neighbors, weights, _lambda=0.9)

        cc_smooth[i] = smooth_row

    pca = PCA(n_components=1)
    scores = pca.fit_transform(cc_smooth.T)

    sign = pca.components_.mean()  # may need to flip
    if sign < 0:
        scores = scores * -1

    scores = scores[:, 0]

    return scores.astype(float64, copy=False)
