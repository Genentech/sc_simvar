"""Utility functions for tests."""

from datetime import timedelta
from time import time
from typing import Any, Callable, TypeVar

A = TypeVar("A", bound=Any)


def timer(_func: Callable[..., A], *args: Any, prefix: str = "", **kwargs: Any) -> A:
    """Time a function.

    Parameters
    ----------
    _func : Callable[..., A]
        The function to time.
    *args : Any
        The positional arguments to pass to the function.
    prefix : str, optional
        A prefix to add to the function name in the print statement, by default "".
    **kwargs : Any
        The keyword arguments to pass to the function.

    Returns
    -------
    A
        The return value of the function.

    """
    start = time()
    result = _func(*args, **kwargs)
    diff = timedelta(seconds=time() - start)
    print(f"Time to run {prefix}{_func.__name__}: {diff}")

    return result


def timer2(_func: Callable[..., A], *args: Any, prefix: str = "", **kwargs: Any) -> tuple[A, timedelta]:
    """Time a function.

    Parameters
    ----------
    _func : Callable[..., A]
        The function to time.
    *args : Any
        The positional arguments to pass to the function.
    prefix : str, optional
        A prefix to add to the function name in the print statement, by default "".
    **kwargs : Any
        The keyword arguments to pass to the function.

    Returns
    -------
    A
        The return value of the function.
    timedelta
        The time it took to run the function.

    """
    start = time()
    result = _func(*args, **kwargs)
    diff = timedelta(seconds=time() - start)

    print(f"Time to run {prefix}{_func.__name__}: {diff}")

    return result, diff
