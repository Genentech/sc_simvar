"""Create centered counts row."""

from typing import Literal

from numpy import float64
from numpy.typing import NDArray

from sc_simvar._lib import fit_bernoulli_model, fit_danb_model, fit_none_model, fit_normal_model


def create_centered_counts_row(
    vals_x: NDArray[float64], model: Literal["bernoulli", "danb", "normal", "none"], num_umi: NDArray[float64]
) -> NDArray[float64]:
    """Create a centered counts row."""
    match model:
        case "bernoulli":
            vals_x = (vals_x > 0).astype("double")
            mu_x, var_x, _ = fit_bernoulli_model(vals_x, num_umi)
        case "danb":
            mu_x, var_x, _ = fit_danb_model(vals_x, num_umi)
        case "normal":
            mu_x, var_x, _ = fit_normal_model(vals_x, num_umi)
        case "none":
            mu_x, var_x, _ = fit_none_model(vals_x)
        case _:
            raise ValueError(f"Invalid model: {model}")

    var_x[var_x == 0] = 1
    out_x = (vals_x - mu_x) / (var_x**0.5)
    out_x[out_x == 0] = 0

    return out_x
