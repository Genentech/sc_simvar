"""Assign modules to the leaves of a dendrogram."""

from typing import Any

from numpy import integer, ones, sort, unique
from numpy.typing import NDArray
from pandas import Index, Series


def prop_label(
    Z: NDArray[Any], node_index: int, label: int, labels: NDArray[integer], out_clusters: NDArray[integer]
):
    """Propagate node labels downward if they are not -1.

    Used to find the correct cluster label at the leaves
    """
    N = Z.shape[0] + 1  # number of leaves

    if label == -1:
        label = labels[node_index]

    left_child = int(Z[node_index, 0] - N)
    right_child = int(Z[node_index, 1] - N)

    if left_child < 0:
        out_clusters[left_child + N] = label
    else:
        prop_label(Z, left_child, label, labels, out_clusters)

    if right_child < 0:
        out_clusters[right_child + N] = label
    else:
        prop_label(Z, right_child, label, labels, out_clusters)


def assign_modules_core(
    Z: NDArray[Any], leaf_labels: Index, offset: int, MIN_THRESHOLD: int = 10, Z_THRESHOLD: float = 3.0
):
    clust_i = 0

    labels = ones(Z.shape[0], dtype=int) * -1
    N = Z.shape[0] + 1

    for i in range(Z.shape[0]):
        ca = int(Z[i, 0])
        cb = int(Z[i, 1])

        if ca - N < 0:  # leaf node
            n_members_a = 1
            clust_a = -1
        else:
            n_members_a = Z[ca - N, 3]
            clust_a = labels[ca - N]

        if cb - N < 0:  # leaf node
            n_members_b = 1
            clust_b = -1
        else:
            n_members_b = Z[cb - N, 3]
            clust_b = labels[cb - N]

        if n_members_a >= MIN_THRESHOLD and n_members_b >= MIN_THRESHOLD:
            # don't join them
            new_clust_assign = -1
        elif Z[i, 2] > offset - Z_THRESHOLD:
            new_clust_assign = -1
        elif n_members_a >= MIN_THRESHOLD:
            new_clust_assign = clust_a
        elif n_members_b >= MIN_THRESHOLD:
            new_clust_assign = clust_b
        elif (n_members_b + n_members_a) >= MIN_THRESHOLD:
            # A new cluster is born!
            new_clust_assign = clust_i
            clust_i += 1
        else:
            new_clust_assign = -1  # Still too small

        labels[i] = new_clust_assign

    out_clusters = ones(N, dtype=int) * -2
    prop_label(Z, Z.shape[0] - 1, labels[-1], labels, out_clusters)

    # remap out_clusters
    unique_clusters = list(sort(unique(out_clusters)))

    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    clust_map = {x: i + 1 for i, x in enumerate(unique_clusters)}
    clust_map[-1] = -1

    out_clusters = [clust_map[x] for x in out_clusters]
    out_clusters = Series(out_clusters, index=leaf_labels)

    return out_clusters
