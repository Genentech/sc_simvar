"""Functions for comparing hotspot results to python results."""

from typing import Any

from numpy import isclose
from numpy.typing import NDArray
from pandas import DataFrame
from pandas.testing import assert_frame_equal


def compare_ndarrays(nd1: NDArray[Any], nd2: NDArray[Any]) -> None:
    """Compare two NDArrays.

    Parameters
    ----------
    nd1 : NDArray[Any]
        The first array.
    nd2 : NDArray[Any]
        The second array.

    Raises
    ------
    AssertionError
        If the arrays are not equal.

    """
    assert nd1.shape == nd2.shape

    try:
        assert nd1.dtype == nd2.dtype
    except AssertionError as e:
        print(nd1.dtype)
        print(nd2.dtype)
        raise e

    if nd1.dtype.kind == "f":
        identities = isclose(nd1, nd2)
    else:
        identities = nd1 == nd2

    try:
        assert identities.all()
    except AssertionError as e:
        print("Num diffs: ", sum(~identities))
        if nd1.dtype.kind in ["i", "u", "f"]:
            print((nd1[~identities] - nd2[~identities])[:10])
        else:
            print(list(zip(nd1[~identities], nd2[~identities]))[:10])
        raise e


def compare_data_frames(df1: DataFrame, df2: DataFrame) -> None:
    """Compare two DataFrames.

    Parameters
    ----------
    df1 : DataFrame
        The first DataFrame.
    df2 : DataFrame
        The second DataFrame.

    Raises
    ------
    AssertionError
        If the DataFrames are not equal.

    """
    assert_frame_equal(df1, df2, check_dtype=True, check_index_type="equiv", check_like=True)
