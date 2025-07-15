"""Smooth row values based on neighbors."""

from numba import njit
from numpy import float64, uint64, zeros_like
from numpy.typing import NDArray


@njit
def neighbor_smoothing_row(
    vals: NDArray[float64], neighbors: NDArray[uint64], weights: NDArray[float64], _lambda: float = 0.9
) -> NDArray[float64]:
    """Output is (neighborhood average) * _lambda + self * (1-_lambda).

    Parameters
    ----------
    vals : NDArray[float64]
        Expression matrix (genes x cells)
    neighbors : NDArray[uint64]
        neighbor indices (cells x K)
    weights : NDArray[float64]
        neighbor weights (cells x K)
    _lambda : float
        Ratio controlling self vs. neighborhood

    Returns
    -------
    NDArray[float64]
        Smooth row values.

    """
    N = neighbors.shape[0]  # Cells
    K = neighbors.shape[1]  # Neighbors

    out = zeros_like(vals, dtype=float64)
    out_denom = zeros_like(vals, dtype=float64)

    for i in range(N):
        xi = vals[i]

        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            out[i] += vals[j] * wij
            out[j] += xi * wij

            out_denom[i] += wij
            out_denom[j] += wij

    # Pre-compute the constant multiplier
    one_minus_lambda = 1.0 - _lambda

    # Vectorized operations for final computation
    for i in range(N):
        out[i] = (out[i] / out_denom[i]) * _lambda + vals[i] * one_minus_lambda

    return out
