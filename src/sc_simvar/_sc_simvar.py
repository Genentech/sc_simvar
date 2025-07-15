"""The SimVar class."""

from typing import Any, Literal, cast

from anndata import AnnData
from anndata._core.views import ArrayView
from matplotlib.colors import Colormap
from numpy import array, asarray, float64, ndarray, ones_like, str_, uint64
from numpy.typing import NDArray
from pandas import DataFrame, Series
from tqdm import tqdm

from ._lib import (
    compute_simvar,
    compute_simvar_and_pairs,
    compute_simvar_pairs_centered_cond,
    make_weights_non_redundant,
)
from ._not_yet_rust import (
    TreeNode,
    compute_modules,
    compute_scores,
    distances_neighbors_and_weights,
    latent_neighbors_and_weights,
    local_correlation_plot,
    tree_neighbors_and_weights,
)
from ._types import HasToArray


class SCSimVar:
    """The SCSimVar class."""

    def __init__(
        self,
        ann_data: AnnData,
        layer_key: str | None = None,
        model: Literal["normal", "danb", "bernoulli", "none"] = "danb",
        *,
        latent_obsm_key: str | None = None,
        distances_obsp_key: str | None = None,
        tree: TreeNode | None = None,
        umi_counts_obs_key: str | None = None,
    ) -> None:
        """Initialize the SimVar class.

        One of `obsm_latent_key`, `obsp_distances_key` or `tree` must be
        provided.

        All matrices will be converted to `float64` for computation. Ensure
        this does not cause a loss of data/precision.

        Parameters
        ----------
        ann_data : AnnData
            The annotated data matrix, shape is cells x genes.
        layer_key : str, optional
            The layer to use for the counts data/matrix, uses `X` if `None`.
        model : Literal["normal", "danb", "bernoulli", "none"], optional
            The model to use, by default `"danb"`.
            - 'danb': Depth-Adjusted Negative Binomial
            - 'bernoulli': Models probability of detection
            - 'normal': Depth-Adjusted Normal
            - 'none': Assumes data has been pre-standardized
        latent_obsm_key : str, optional
            The key in the `AnnData.obsm` field containing the cell-cell
            similarities calculated from euclidean distances. Defaults to
            `None`.
        distances_obsp_key : str, optional
            The key in the `AnnData.obsp` field containing the cell-cell
            distances. Defaults to `None`.
        tree : TreeNode, optional
            Root `ete3.TreeNode` to calculate cell-cell distances from.
            Defaults to `None`.
        umi_counts_obs_key : str, optional
            The key in the `AnnData.obs` field containing the UMI counts.
            If omitted the sum over the genes in the counts matrix is used.
            Defaults to `None`.

        """
        self._ann_data = ann_data

        self._validate_knn_key(latent_obsm_key, distances_obsp_key, tree)

        self._model: Literal["normal", "danb", "bernoulli", "none"] = model

        self._layer_key = layer_key
        self._counts = self._validate_counts()
        self._umi_counts = self._validate_umi_counts(umi_counts_obs_key)

        self._distances = (
            None
            if distances_obsp_key is None
            else self._convert_to_float64(self._ann_data.obsp[distances_obsp_key])
        )

        self._latent = (
            None
            if latent_obsm_key is None
            else self._convert_to_float64(self._ann_data.obsm[latent_obsm_key])
        )

        self._tree = tree

        self._gene_labels: NDArray[str_] = self._ann_data.var_names.to_numpy(dtype="U25")
        self._cell_labels = self._ann_data.obs_names.to_series()

        # NOTE: these are filled by methods
        self._neighbors: NDArray[uint64] | None = None
        self._weights: NDArray[float64] | None = None
        self._results: DataFrame | None = None
        self._local_correlation_c: DataFrame | None = None
        self._local_correlation_z: DataFrame | None = None
        self._linkage: NDArray[uint64] | None = None
        self._modules: Series | None = None
        self._module_scores: DataFrame | None = None

    def _validate_knn_key(
        self, latent_obsm_key: str | None, distances_obsp_key: str | None, tree: TreeNode | None
    ) -> None:
        """Validate the knn key.

        Parameters
        ----------
        latent_obsm_key : str | None
            The key in the `AnnData.obsm` field containing the cell-cell
            similarities calculated from euclidean distances.
        distances_obsp_key : str | None
            The key in the `AnnData.obsp` field containing the cell-cell
            distances.
        tree : TreeNode | None
            Root `ete3.TreeNode` to calculate cell-cell distances from.

        """
        if (
            sum(
                [
                    latent_obsm_key is None,
                    distances_obsp_key is None,
                    tree is None,
                ]
            )
            == 3
        ):
            raise ValueError("One of `obsm_latent_key`, `obsp_distances_key` or `tree` must be provided.")

        if (
            sum(
                [
                    latent_obsm_key is not None,
                    distances_obsp_key is not None,
                    tree is not None,
                ]
            )
            > 1
        ):
            raise ValueError("Only one of `obsm_latent_key`, `obsp_distances_key` or `tree` can be provided.")

        if tree is not None:
            try:
                all_leaves: set[str] = set(x.name for x in tree if x.is_leaf())
            except AttributeError:
                raise ValueError("The tree must be an `ete3.TreeNode` object.") from None

            if len(all_leaves) != len(self._ann_data.obs_names) or len(
                all_leaves & set(self._ann_data.obs_names)
            ) != len(self._ann_data.obs_names):
                raise ValueError("The tree must contain all the cells in the AnnData object.")

    def _counts_from_ann_data(self, ann_data: AnnData) -> NDArray[float64]:
        """Get the counts matrix from an AnnData object.

        Parameters
        ----------
        ann_data : AnnData
            The AnnData object.

        Returns
        -------
        NDArray[float64]
            The counts matrix.

        """
        counts = ann_data.X if self._layer_key is None else ann_data.layers[self._layer_key]
        if counts is None:
            raise ValueError("No counts matrix found in AnnData object.")

        if not isinstance(counts, HasToArray | DataFrame | ndarray | ArrayView | Series):
            raise ValueError("Counts matrix must be a sparse matrix, DataFrame, or ndarray.")

        return self._convert_to_float64(counts).transpose()

    def _validate_counts(self) -> NDArray[float64]:
        """Validate the counts matrix."""
        counts = self._counts_from_ann_data(self._ann_data)

        if counts.shape[0] > (counts.sum(axis=1) > float64(0.0)).sum():
            raise ValueError(
                "Counts matrix contains genes with zero sums. Please "
                "remove these genes before running SimVar."
            )

        return counts

    def _validate_umi_counts(self, umi_counts_obs_key: str | None) -> NDArray[float64]:
        """Validate the UMI counts.

        Parameters
        ----------
        umi_counts_obs_key : str | None
            The key in the `AnnData.obs` field containing the UMI counts.

        Returns
        -------
        NDArray[float64]
            The UMI counts.

        """
        if umi_counts_obs_key is None:
            umi_counts: NDArray[float64] = self._counts.sum(axis=0)
        else:
            umi_counts = self._convert_to_float64(self._ann_data.obs[umi_counts_obs_key])

            if umi_counts.ndim != 1:
                raise ValueError("UMI counts must be 1D.")

            if umi_counts.shape[0] != self._counts.shape[1]:
                raise ValueError("UMI counts must be the same length as the number of cells.")

        return umi_counts

    def _convert_to_float64(
        self, matrix: HasToArray | DataFrame | NDArray[Any] | ArrayView | Series
    ) -> NDArray[float64]:
        """Convert a matrix to float64.

        Parameters
        ----------
        matrix : Union[V, ArrayView]
            The matrix to convert.

        Returns
        -------
        NDArray[float64]
            The converted matrix.

        """
        if isinstance(matrix, HasToArray):
            if not hasattr(matrix, "toarray"):
                raise ValueError("Matrix must have a `toarray` method.")

            matrix = matrix.toarray()
        elif isinstance(matrix, (DataFrame, Series)):
            matrix = cast(NDArray[Any], matrix.to_numpy())
        elif isinstance(matrix, ArrayView):
            matrix = cast(NDArray[Any], asarray(matrix))

        return matrix.astype(float64, copy=False)

    @property
    def neighbors(self) -> DataFrame | None:
        """The indices of the neighbors.

        Returns
        -------
        DataFrame | None
            The indices of the neighbors.

        """
        return DataFrame(self._neighbors, index=self._gene_labels, columns=self._cell_labels)

    @property
    def weights(self) -> DataFrame | None:
        """The weights of the neighbors.

        Returns
        -------
        DataFrame | None
            The weights of the neighbors.

        """
        return DataFrame(self._weights, index=self._gene_labels, columns=self._cell_labels)

    @property
    def results(self) -> DataFrame | None:
        """The results.

        Returns
        -------
        DataFrame | None
            The results.

        """
        return self._results

    @property
    def local_correlation_c(self) -> DataFrame | None:
        """The local correlation C.

        Returns
        -------
        NDArray[float64] | None
            The local correlation C.

        """
        return self._local_correlation_c

    @property
    def local_correlation_z(self) -> DataFrame | None:
        """The local correlation Z.

        Returns
        -------
        NDArray[float64] | None
            The local correlation Z.

        """
        return self._local_correlation_z

    @property
    def linkage(self) -> NDArray[uint64] | None:
        """The linkage.

        Returns
        -------
        NDArray[uint64] | None
            The linkage.

        """
        return self._linkage

    @property
    def modules(self) -> Series | None:
        """The modules.

        Returns
        -------
        Optional[Series[int]]
            The modules.

        """
        return self._modules

    @property
    def module_scores(self) -> DataFrame | None:
        """The module scores.

        Returns
        -------
        DataFrame | None
            The module scores.

        """
        return self._module_scores

    def create_knn_graph(
        self,
        weighted_graph: bool = False,
        n_neighbors: int = 30,
        neighborhood_factor: int = 3,
        approx_neighbors: bool = True,
    ) -> None:
        """Create a k-nearest neighbor graph.

        Parameters
        ----------
        weighted_graph : bool, optional
            Whether to create a weighted graph, by default `False`.
        n_neighbors : int, optional
            The number of nearest neighbors to use, by default `30`.
        neighborhood_factor : int, optional
            The number of neighbors to approximate the full graph with,
            by default `3`.
        approx_neighbors : bool, optional
            Whether to approximate the neighbors, by default `True`.

        """
        if self._latent is not None:
            neighbors, weights = latent_neighbors_and_weights(
                self._latent,
                n_neighbors,
                neighborhood_factor,
                approx_neighbors,
            )
        elif self._distances is not None:
            neighbors, weights = distances_neighbors_and_weights(
                self._distances,
                n_neighbors,
                neighborhood_factor,
            )
        elif self._tree is not None:
            neighbors, weights = tree_neighbors_and_weights(
                self._tree,
                n_neighbors=n_neighbors,
                cell_labels=self._cell_labels.to_list(),
            )
        else:
            # NOTE: should never happen due to checks in __init__
            raise ValueError("No latent space or distances provided.")

        if not weighted_graph:
            weights = ones_like(weights)
        weights = make_weights_non_redundant(weights, neighbors)
        self._neighbors = neighbors
        self._weights = weights

    def compute_autocorrelations(self, jobs: int = 1) -> DataFrame:
        """Compute the auto correlations.

        Parameters
        ----------
        jobs : int, optional
            Not used.

        Returns
        -------
        DataFrame
            A `DataFrame` with four columns:
            - C: Scaled autocorrelation coefficients
            - Z: Z-scores for autocorrelation coefficients
            - Pval: P-values computed from Z-scores
            - FDR: Adjusted P-values using the Benjamini-Hochberg procedure

        """
        if self._neighbors is None or self._weights is None:
            raise ValueError(
                "No neighbors or weights computed, please call the `create_knn_graph` method first."
            )

        results = compute_simvar(
            self._counts,
            self._neighbors,
            self._weights,
            self._umi_counts,
            self._gene_labels,
            self._model,
            True,
        )

        self._results = DataFrame(dict(zip(["C", "Z", "Pval", "FDR"], results[1:])), index=results[0])
        self._results.index.name = "Gene"

        return self._results

    def compute_local_correlations(
        self, genes: list[str] | NDArray[str_] | None = None, jobs: int = 1
    ) -> DataFrame:
        """Compute the local correlations.

        Parameters
        ----------
        genes : list[str] | None, optional
            The genes to compute the local correlations for, if `None` all
            genes are used, by default `None`.
        jobs : int, optional
            Not used.

        Returns
        -------
        DataFrame
            A `DataFrame` with the local correlations Z scores of dimensions
            genes x genes.

        """
        if self._neighbors is None or self._weights is None:
            raise ValueError(
                "No neighbors or weights computed, please call the `create_knn_graph` method first."
            )

        if genes is None:
            genes = self._gene_labels
        elif isinstance(genes, list):
            genes = array(genes, dtype="U25")

        print(f"Computing pair-wise local correlation on {len(genes)} features...")

        lcps, zs = compute_simvar_pairs_centered_cond(
            self._counts_from_ann_data(self._ann_data[:, genes]),
            self._neighbors,
            self._weights,
            self._umi_counts,
            self._model,
        )

        self._local_correlation_c = DataFrame(lcps, index=genes, columns=genes)
        self._local_correlation_z = DataFrame(zs, index=genes, columns=genes)

        return self._local_correlation_z

    def compute_auto_and_local_correlations(
        self, genes: list[str] | NDArray[str_] | None = None
    ) -> tuple[DataFrame, DataFrame]:
        """Compute the auto and local correlations.

        Avoids returning to the Python layer between the two computations.

        Parameters
        ----------
        genes : list[str] | None, optional
            The genes to compute the local correlations for, if `None` all
            genes are used, by default `None`.

        Returns
        -------
        DataFrame
            A `DataFrame` with four columns:
            - C: Scaled -1:1 autocorrelation coefficients
            - Z: Z-score for autocorrelation
            - Pval:  P-values computed from Z-scores
            - FDR:  Q-values using the Benjamini-Hochberg procedure
        DataFrame
            A `DataFrame` with the local correlations Z scores of dimensions
            genes x genes.

        """
        if self._neighbors is None or self._weights is None:
            raise ValueError(
                "No neighbors or weights computed, please call the `create_knn_graph` method first."
            )

        if genes is None:
            genes = self._gene_labels
        elif isinstance(genes, list):
            genes = array(genes, dtype="U25")

        print(f"Computing pair-wise local correlation on {len(genes)} features...")

        all_results = compute_simvar_and_pairs(
            self._counts,
            self._counts_from_ann_data(self._ann_data[:, genes]),
            self._neighbors,
            self._weights,
            self._umi_counts,
            self._gene_labels,
            self._model,
            True,
        )

        self._results = DataFrame(
            dict(zip(["C", "Z", "Pval", "FDR"], all_results[1:5])), index=all_results[0]
        )
        self._results.index.name = "Gene"

        self._local_correlation_c = DataFrame(all_results[5], index=genes, columns=genes)
        self._local_correlation_z = DataFrame(all_results[6], index=genes, columns=genes)

        return self._results, self._local_correlation_z

    # TODO: make rust code for this next?
    def create_modules(
        self,
        min_gene_threshold: int = 20,
        core_only: bool = True,
        fdr_threshold: float = 0.05,
    ) -> Series:
        """Group genes into modules.

        Parameters
        ----------
        min_gene_threshold : int, optional
            The minimum number of genes in a module, decrease if too many
            modules are formed, decrease if sub-structure is not being
            captured. Defaults to `20`.
        core_only : bool, optional
            If `False` genes which cannot be unambiguously assigned to a module
            are instead assigned to a noise module indicated as `-1` in the
            results. If `True` they are assigned to the likeliest module.
            Defaults to `True`.
        fdr_threshold : float, optional
            The FDR threshold to use for module assignment, defaults to `0.05`.

        Returns
        -------
        Series
            A `Series` with the module assignments for each gene, unassigned
            genes are indicated with `-1`.

        """
        if self._local_correlation_z is None:
            raise ValueError(
                "No local correlations computed, please call the `compute_local_correlations` method first."
            )

        self._modules, self._linkage = compute_modules(
            self._local_correlation_z,
            min_gene_threshold=min_gene_threshold,
            fdr_threshold=fdr_threshold,
            core_only=core_only,
        )

        return self._modules

    # TODO: move python code here for now
    # TODO: make rust code for this next
    def calculate_module_scores(self) -> DataFrame:
        """Calculate module scores.

        Returns
        -------
        module_scores : DataFrame
            A `DataFrame` of dimensions genes x modules containing the
            module scores for each gene.

        """
        if self._modules is None or self._linkage is None:
            raise ValueError("No modules or linkage computed, please call the `create_modules` method first.")

        if self._neighbors is None or self._weights is None:
            raise ValueError(
                "No neighbors or weights computed, please call the `create_knn_graph` method first."
            )

        modules_to_compute: list[int] = sorted([x for x in self._modules.unique() if x != -1])

        print(f"Computing scores for {len(modules_to_compute)} modules.")

        module_scores: dict[int, NDArray[Any]] = {}
        for module in tqdm(modules_to_compute):
            module_genes = self._modules.index[self._modules == module].to_series()  # type: ignore

            counts_dense = self._counts_from_ann_data(self._ann_data[:, module_genes])

            scores = cast(
                NDArray[Any],
                compute_scores(
                    counts_dense,
                    self._model,
                    self._umi_counts,
                    self._neighbors,
                    self._weights,
                ),
            )

            module_scores[module] = scores

        self._module_scores = DataFrame(module_scores, index=self._cell_labels)

        return self._module_scores

    # TODO: move python code here
    def plot_local_correlation(
        self,
        mod_cmap: str | Colormap = "tab10",
        vmin: int = -8,
        vmax: int = 8,
        z_cmap: str | Colormap = "RdBu_r",
        yticklabels: bool = False,
    ) -> None:
        """Plot a cluster-grid of the local correlation values.

        Parameters
        ----------
        mod_cmap : str | Colormap, optional
            Valid `matplotlib` colormap `str` or a `ColorMap` from the
            `matplotlib.colormaps` for module assignments on the left side.
        vmin : int, optional
            Min value for Z-scores color scale. Default is `-8.0`.
        vmax : int, optional
            Max value for Z-scores color scale. Default is `8.0`.
        z_cmap : str | Colormap, optional
            Valid `matplotlib` colormap `str` or a `ColorMap` from the
            `matplotlib.colormaps` for correlation Z-scores.
        yticklabels: bool, optional
            If `True` plot all gene labels on the Y-axis. Useful if using plot
            interactively and can zoom in, otherwise there are too many genes.
            Default is `False`.

        """
        if self._local_correlation_z is None:
            raise ValueError(
                "No local correlations computed, please call the `compute_local_correlations` method first."
            )

        if self._modules is None or self._linkage is None:
            raise ValueError("No modules or linkage computed, please call the `create_modules` method first.")

        local_correlation_plot(
            self._local_correlation_z,
            self._modules,
            self._linkage,
            mod_cmap=mod_cmap,
            vmin=vmin,
            vmax=vmax,
            z_cmap=z_cmap,
            yticklabels=yticklabels,
        )
