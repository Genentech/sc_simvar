"""Test the different models."""

from compare import compare_ndarrays
from hotspot.bernoulli_model import fit_gene_model_linear as hs_fit_bernoulli_model
from hotspot.danb_model import fit_gene_model as hs_fit_danb_model
from hotspot.none_model import fit_gene_model as hs_fit_none_model
from hotspot.normal_model import fit_gene_model as hs_fit_normal_model
from simulate import sim_bernoulli_counts, sim_danb_counts
from utils import timer

from sc_simvar._lib import fit_bernoulli_model, fit_danb_model, fit_none_model, fit_normal_model


def test_none_model() -> None:
    """Test the none model."""
    gene_counts, umi_counts = sim_danb_counts()

    for row in gene_counts:
        # for JIT compilation
        hs_fit_none_model(row, umi_counts)
        fit_none_model(row)

        hs_means, hs_vars, hs_x2s = timer(hs_fit_none_model, row, umi_counts, prefix="hs_")
        sv_means, sv_vars, sv_x2s = timer(fit_none_model, row)

        compare_ndarrays(hs_means, sv_means)
        compare_ndarrays(hs_vars, sv_vars)
        compare_ndarrays(hs_x2s, sv_x2s)

        break


def test_normal_model() -> None:
    """Test the normal model."""
    gene_counts, umi_counts = sim_danb_counts()
    f_umi_counts = umi_counts.astype("float64")

    for row in gene_counts:
        # for JIT compilation
        hs_fit_normal_model(row, umi_counts)
        fit_normal_model(row, f_umi_counts)

        hs_means, hs_vars, hs_x2s = timer(hs_fit_normal_model, row, umi_counts, prefix="hs_")
        sv_means, sv_vars, sv_x2s = timer(fit_normal_model, row, f_umi_counts)

        compare_ndarrays(hs_means, sv_means)
        compare_ndarrays(hs_vars, sv_vars)
        compare_ndarrays(hs_x2s, sv_x2s)

        break


def test_danb_model() -> None:
    """Test the danb model."""
    gene_counts, umi_counts = sim_danb_counts()
    f_umi_counts = umi_counts.astype("float64")

    for row in gene_counts:
        # for JIT compilation
        hs_fit_danb_model(row, umi_counts)
        fit_danb_model(row, f_umi_counts)

        hs_means, hs_vars, hs_x2s = timer(hs_fit_danb_model, row, umi_counts, prefix="hs_")
        sv_means, sv_vars, sv_x2s = timer(fit_danb_model, row, f_umi_counts)

        compare_ndarrays(hs_means, sv_means)
        compare_ndarrays(hs_vars, sv_vars)
        compare_ndarrays(hs_x2s, sv_x2s)

        break


def test_bernoulli_model() -> None:
    """Test the bernoulli model."""
    gene_counts, umi_counts = sim_bernoulli_counts()
    f_umi_counts = umi_counts.astype("float64")

    for row in gene_counts:
        # for JIT compilation
        hs_fit_bernoulli_model(row, umi_counts)
        fit_bernoulli_model(row, f_umi_counts)

        hs_means, hs_vars, hs_x2s = timer(hs_fit_bernoulli_model, row, umi_counts, prefix="hs_")
        sv_means, sv_vars, sv_x2s = timer(fit_bernoulli_model, row, f_umi_counts)

        compare_ndarrays(hs_means, sv_means)
        compare_ndarrays(hs_vars, sv_vars)
        compare_ndarrays(hs_x2s, sv_x2s)

        break


if __name__ == "__main__":
    # works
    test_none_model()
    # works
    test_normal_model()
    # works
    test_danb_model()
    # # works
    test_bernoulli_model()
