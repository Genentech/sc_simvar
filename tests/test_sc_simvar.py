"""Test the whole thing."""

from math import floor

from anndata import AnnData
from compare import compare_data_frames, compare_ndarrays
from hotspot import Hotspot
from pandas import DataFrame
from simulate import sim_cell_labels, sim_danb_counts, sim_gene_labels, sim_latent
from utils import timer

from sc_simvar import SCSimVar

gene_counts, umi_counts = sim_danb_counts()
cell_labels = sim_cell_labels()
gene_labels = sim_gene_labels()
partial_gene_labels = gene_labels[: floor(len(gene_labels) / 2)]
latent = sim_latent()

ann_data = AnnData(DataFrame(gene_counts.transpose(), columns=gene_labels, index=cell_labels))
ann_data.obsm["latent"] = DataFrame(latent, index=cell_labels)
ann_data.obs["umi_counts"] = umi_counts


def test_simvar() -> None:
    """Test the whole thing."""
    hotspot = timer(
        Hotspot,
        ann_data,
        latent_obsm_key="latent",
        umi_counts_obs_key="umi_counts",
        prefix="hs_",
    )
    sc_simvar = timer(
        SCSimVar,
        ann_data,
        latent_obsm_key="latent",
        umi_counts_obs_key="umi_counts",
    )

    timer(hotspot.create_knn_graph, approx_neighbors=False, prefix="hs_")
    timer(sc_simvar.create_knn_graph, approx_neighbors=False)

    hs_results = timer(hotspot.compute_autocorrelations, prefix="hs_")
    sv_results = timer(sc_simvar.compute_autocorrelations)

    if not isinstance(hs_results, DataFrame):
        raise TypeError("The autocorrelations from Hotspot are not a DataFrame.")

    compare_data_frames(hs_results, sv_results)

    hs_lc_z = timer(hotspot.compute_local_correlations, partial_gene_labels, prefix="hs_")
    sv_ls_z = timer(sc_simvar.compute_local_correlations, partial_gene_labels)

    compare_data_frames(hs_lc_z, sv_ls_z)

    sv_results, sv_ls_z = timer(sc_simvar.compute_auto_and_local_correlations, partial_gene_labels)

    sub_hs_results = hs_results[hs_results.index.isin(sv_results.index)].reindex(sv_results.index)

    if not isinstance(sub_hs_results, DataFrame):
        raise TypeError("Subsetting the autocorrelations did not return a DatFrame.")

    compare_data_frames(sub_hs_results, sv_results)
    compare_data_frames(hs_lc_z, sv_ls_z)

    hs_modules = timer(hotspot.create_modules, min_gene_threshold=2, fdr_threshold=1, prefix="hs_")
    sv_modules = timer(sc_simvar.create_modules, min_gene_threshold=2, fdr_threshold=1)

    compare_ndarrays(hs_modules.to_numpy(), sv_modules.to_numpy())
    compare_ndarrays(hotspot.linkage, sc_simvar._linkage)  # type: ignore

    hs_module_scores = timer(hotspot.calculate_module_scores, prefix="hs_")
    sv_module_scores = timer(sc_simvar.calculate_module_scores)

    compare_data_frames(hs_module_scores, sv_module_scores)


if __name__ == "__main__":
    test_simvar()
