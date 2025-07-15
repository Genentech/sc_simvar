"""Sorts linkage by 'node_values' in place."""

from numba import njit
from numpy import empty, float64, int32
from numpy.typing import NDArray


@njit
def sort_linkage(Z: NDArray[float64], node_index: int, node_values: NDArray[float64]) -> None:
    """Sorts linkage by 'node_values' in place."""
    N = Z.shape[0] + 1  # number of leaves

    # Use a pre-allocated array as stack instead of Python list
    stack = empty(Z.shape[0] * 2, dtype=int32)
    stack_size = 0
    stack[stack_size] = node_index
    stack_size += 1

    while stack_size > 0:
        stack_size -= 1
        node_index = stack[stack_size]

        if node_index < 0:
            continue

        left_child = int(Z[node_index, 0] - N)
        right_child = int(Z[node_index, 1] - N)

        swap = False

        if left_child < 0 and right_child < 0:
            swap = False
        elif left_child < 0 and right_child >= 0:
            swap = True
        elif left_child >= 0 and right_child < 0:
            swap = False
        else:
            if node_values[left_child] > node_values[right_child]:
                swap = True
            else:
                swap = False

        if swap:
            Z[node_index, 0] = right_child + N
            Z[node_index, 1] = left_child + N

        if left_child >= 0:
            stack[stack_size] = left_child
            stack_size += 1
        if right_child >= 0:
            stack[stack_size] = right_child
            stack_size += 1
