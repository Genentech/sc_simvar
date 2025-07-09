"""Calculate the mean density of joins for sub-trees underneath each node."""

from numpy import float64
from numpy.typing import NDArray


# TODO: get rid of recursion?
def calc_mean_dists(
    Z: NDArray[float64], node_index: int, out_mean_dists: NDArray[float64]
) -> tuple[float64, float64]:
    """Calculate the mean density of joins for sub-trees underneath each node."""
    N = Z.shape[0] + 1  # number of leaves

    left_child = int(Z[node_index, 0] - N)
    right_child = int(Z[node_index, 1] - N)

    if left_child < 0:
        left_average = float64(0.0)
        left_merges = float64(0.0)
    else:
        left_average, left_merges = calc_mean_dists(Z, left_child, out_mean_dists)

    if right_child < 0:
        right_average = float64(0.0)
        right_merges = float64(0.0)
    else:
        right_average, right_merges = calc_mean_dists(Z, right_child, out_mean_dists)

    this_height = Z[node_index, 2]
    this_merges = left_merges + right_merges + 1
    this_average = (left_average * left_merges + right_average * right_merges + this_height) / this_merges

    out_mean_dists[node_index] = this_average

    return this_average, this_merges
