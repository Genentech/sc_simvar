"""Benchmark the full sc_simvar pipeline vs hotspot."""

from itertools import product
from json import JSONDecodeError, dump, load
from math import floor
from pathlib import Path
from time import time

from anndata import AnnData
from hotspot import Hotspot
from matplotlib.pyplot import savefig, show, style, subplots, tight_layout
from numpy import arange, isnan, sqrt
from pandas import DataFrame, Series
from scipy import stats
from simulate import sim_cell_labels, sim_danb_counts, sim_gene_labels, sim_latent
from tqdm import tqdm
from utils import timer2

from sc_simvar import SCSimVar


def _make_ann_data(n_cells: int, n_genes: int, n_dims: int) -> tuple[AnnData, list[str]]:
    """Create an AnnData object with simulated data."""
    gene_counts, umi_counts = sim_danb_counts(n_genes=n_genes, n_cells=n_cells)
    cell_labels = sim_cell_labels(n_cells)
    gene_labels = sim_gene_labels(n_genes=n_genes)
    latent = sim_latent(n_cells, n_dims)
    partial_gene_labels = gene_labels[: floor(len(gene_labels) / 2)]

    if not isinstance(partial_gene_labels, Series):
        raise TypeError("Subsettin the gene_labels did not return a Series object.")

    ann_data = AnnData(DataFrame(gene_counts.transpose(), columns=gene_labels, index=cell_labels))
    ann_data.obsm["latent"] = DataFrame(latent, index=cell_labels)
    ann_data.obs["umi_counts"] = umi_counts

    return ann_data, partial_gene_labels.to_list()


def _run_pipeline(
    cls: type[Hotspot | SCSimVar],
    ann_data: AnnData,
    weighted_graph: bool,
    approx_neighbors: bool,
    partial_gene_labels: list[str],
) -> None:
    """Run the sc_simvar pipeline on the given AnnData object."""
    obj = cls(ann_data, latent_obsm_key="latent", umi_counts_obs_key="umi_counts")

    print("Creating latent space...")
    start = time()
    obj.create_knn_graph(weighted_graph=weighted_graph, approx_neighbors=approx_neighbors)
    print(f"Created knn graph in {time() - start:.2f} seconds")

    print("Computing correlations...")
    if isinstance(obj, SCSimVar):
        start = time()
        obj.compute_auto_and_local_correlations(partial_gene_labels)
        print(f"Computed auto and local correlations in {time() - start:.2f} seconds")
    else:
        start = time()
        obj.compute_autocorrelations()
        obj.compute_local_correlations(partial_gene_labels)
        print(f"Computed auto and local correlations in {time() - start:.2f} seconds")

    print("Creating modules and calculating scores...")
    start = time()
    obj.create_modules(min_gene_threshold=2, fdr_threshold=1)
    obj.calculate_module_scores()
    print(f"Created modules and calculated scores in {time() - start:.2f} seconds")


def adhoc_benchmark() -> None:
    """Run a quick benchmark with fixed parameters."""
    n_cells = 10_000
    n_genes = 2_000
    n_dims = 150

    ann_data, partial_gene_labels = _make_ann_data(n_cells, n_genes, n_dims)

    _run_pipeline(
        SCSimVar,
        ann_data,
        weighted_graph=True,
        approx_neighbors=True,
        partial_gene_labels=partial_gene_labels,
    )

    print("Running Hotspot")
    _, hs_diff = timer2(
        _run_pipeline,
        Hotspot,
        ann_data,
        weighted_graph=True,
        approx_neighbors=True,
        partial_gene_labels=partial_gene_labels,
    )
    print(f"Hotspot completed in {hs_diff.total_seconds()} seconds")

    print("Running SCSimVar")
    _, sc_diff = timer2(
        _run_pipeline,
        SCSimVar,
        ann_data,
        weighted_graph=True,
        approx_neighbors=True,
        partial_gene_labels=partial_gene_labels,
    )
    print(f"SCSimVar completed in {sc_diff.total_seconds()} seconds")


def benchmark() -> None:
    """Benchmark the full sc_simvar pipeline vs hotspot."""
    results_file = "benchmark_results.json"

    # Load existing results if file exists
    if Path(results_file).exists():
        with open(results_file, "r") as f:
            try:
                results = load(f)
            except JSONDecodeError:
                print(f"Error reading {results_file}. Starting fresh.")
                results = {}
    else:
        results = {}

    # Get all combinations first to show progress
    combinations = list(
        product([True, False], [True, False], [5_000, 10_000, 15_000], [1_000, 2_000, 3_000], [50, 100, 150])
    )

    for weighted_graph, approx_neighbors, n_cells, n_genes, n_dims in tqdm(combinations, desc="Benchmarking"):
        # Create unique key for this combination
        key = f"{weighted_graph}_{approx_neighbors}_{n_cells}_{n_genes}_{n_dims}"

        # Skip if already computed
        if key in results:
            print(f"Skipping {key} - already computed")
            continue

        print(
            f"Running benchmark for {n_genes} genes and {n_cells} cells with {n_dims} dimensions, "
            f"weighted_graph={weighted_graph}, approx_neighbors={approx_neighbors}"
        )
        ann_data, partial_gene_labels = _make_ann_data(n_cells, n_genes, n_dims)

        print("Running SCSimVar")
        _, sc_diff = timer2(
            _run_pipeline, SCSimVar, ann_data, weighted_graph, approx_neighbors, partial_gene_labels
        )
        print(f"SCSimVar completed in {sc_diff.total_seconds()} seconds")

        print("Running Hotspot")
        _, hs_diff = timer2(
            _run_pipeline,
            Hotspot,
            ann_data,
            weighted_graph,
            approx_neighbors,
            partial_gene_labels,
        )
        print(f"Hotspot completed in {hs_diff.total_seconds()} seconds")

        # Store results
        results[key] = {
            "weighted_graph": weighted_graph,
            "approx_neighbors": approx_neighbors,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_dims": n_dims,
            "sc_simvar_time": sc_diff.total_seconds(),
            "hotspot_time": hs_diff.total_seconds(),
        }

        # Save results after each iteration
        with open(results_file, "w") as f:
            dump(results, f, indent=2)

    # Create visualization
    _create_benchmark_plots(results)
    _analyze_parameter_impact(results)


def _create_benchmark_plots(results: dict) -> None:
    """Create bar chart showing runtime ratio comparison and absolute times in separate subplots."""
    # Convert to DataFrame
    df = DataFrame.from_dict(results, orient="index")

    # Calculate ratio for all individual runs first
    df["ratio"] = df["hotspot_time"] / df["sc_simvar_time"]

    # Calculate 95% confidence interval for the ratio across all runs
    mean_ratio = df["ratio"].mean()
    std_ratio = df["ratio"].std()
    n_samples = len(df)

    # 95% confidence interval using t-distribution
    confidence_level = 0.95
    degrees_freedom = n_samples - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_error = t_value * (std_ratio / sqrt(n_samples))

    ci_lower = mean_ratio - margin_error
    ci_upper = mean_ratio + margin_error

    print(f"CI lower: {ci_lower}, ci mid: {mean_ratio}, ci upper: {ci_upper}")

    # Group by all parameters except n_dims and weighted_graph and calculate averages
    grouping_cols = ["n_genes", "n_cells", "approx_neighbors"]
    df_grouped = (
        df.groupby(grouping_cols)
        .agg({"n_dims": "mean", "weighted_graph": "mean", "hotspot_time": "mean", "sc_simvar_time": "mean"})
        .reset_index()
    )

    # Create a label for each combination
    df_grouped["combination"] = df_grouped.apply(
        lambda row: f"Cells:{row['n_cells']} | Approx:{row['approx_neighbors']}",
        axis=1,
    )

    # Calculate ratio
    df_grouped["ratio"] = df_grouped["hotspot_time"] / df_grouped["sc_simvar_time"]

    # Split by n_genes
    unique_n_genes = sorted(df_grouped["n_genes"].unique())
    n_subplots = len(unique_n_genes)

    # One Dark Pro color scheme
    style.use("dark_background")
    colors = {
        "background": "#1a1a1a",
        "text": "#e6e6e6",
        "blue": "#4a90e2",
        "green": "#5cb85c",
        "orange": "#f0ad4e",
        "red": "#d9534f",
        "purple": "#8e6fb8",
        "cyan": "#5bc0de",
        "yellow": "#f0c341",
    }

    # Create subplots - 2 rows (ratio and absolute times)
    fig, axes = subplots(2, n_subplots, figsize=(7 * n_subplots, 12), facecolor=colors["background"])
    if n_subplots == 1:
        axes = axes.reshape(-1, 1)

    for i, n_genes in enumerate(unique_n_genes):
        df_subset = df_grouped[df_grouped["n_genes"] == n_genes].copy()

        # Sort by n_cells (ascending), then by approx_neighbors (False first, True second)
        df_subset = df_subset.sort_values(["n_cells", "approx_neighbors"], ascending=[True, True])

        # Calculate bar positions
        x = arange(len(df_subset))
        width = 0.35

        # Top subplot: Ratio comparison
        _ = axes[0, i].bar(
            x, df_subset["ratio"], width, alpha=0.8, color=colors["purple"], label="Ratio (Hotspot/SCSimVar)"
        )

        axes[0, i].set_xlabel("Parameter Combinations", color=colors["text"])
        axes[0, i].set_ylabel("Runtime Ratio (Hotspot/SCSimVar)", color=colors["text"])
        axes[0, i].set_title(f"Runtime Ratio: n_genes = {n_genes}", color=colors["text"])
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels(df_subset["combination"], rotation=45, ha="right", color=colors["text"])
        axes[0, i].grid(True, alpha=0.3, color=colors["text"])
        axes[0, i].axhline(y=1, color=colors["orange"], linestyle="--", alpha=0.7)
        ci_band = axes[0, i].axhspan(ci_lower, ci_upper, alpha=0.3, color=colors["cyan"])
        axes[0, i].set_ylim(0, 4.0)
        axes[0, i].legend(
            [ci_band],
            ["95% Ratio CI All Runs"],
            loc="upper left",
            facecolor=colors["background"],
            edgecolor=colors["text"],
            labelcolor=colors["text"],
        )
        axes[0, i].tick_params(colors=colors["text"])
        axes[0, i].set_facecolor(colors["background"])

        # Bottom subplot: Absolute times
        hotspot_bars = axes[1, i].bar(
            x - width / 2,
            df_subset["hotspot_time"],
            width,
            alpha=0.8,
            color=colors["red"],
            label="Hotspot Time",
        )
        sc_bars = axes[1, i].bar(
            x + width / 2,
            df_subset["sc_simvar_time"],
            width,
            alpha=0.8,
            color=colors["blue"],
            label="SCSimVar Time",
        )

        axes[1, i].set_xlabel("Parameter Combinations", color=colors["text"])
        axes[1, i].set_ylabel("Runtime (seconds)", color=colors["text"])
        axes[1, i].set_title(f"Absolute Runtime: n_genes = {n_genes}", color=colors["text"])
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels(df_subset["combination"], rotation=45, ha="right", color=colors["text"])
        axes[1, i].grid(True, alpha=0.3, color=colors["text"])
        axes[1, i].set_ylim(0, 600)
        axes[1, i].legend(
            [hotspot_bars, sc_bars],
            ["Hotspot Time", "SCSimVar Time"],
            loc="upper left",
            facecolor=colors["background"],
            edgecolor=colors["text"],
            labelcolor=colors["text"],
        )
        axes[1, i].tick_params(colors=colors["text"])
        axes[1, i].set_facecolor(colors["background"])

    tight_layout()
    savefig("benchmark_results.png", dpi=300, bbox_inches="tight", facecolor=colors["background"])
    show()


def _analyze_parameter_impact(results: dict) -> None:
    """Analyze the impact of each parameter on SC SimVar runtime."""
    # Convert to DataFrame
    df = DataFrame.from_dict(results, orient="index")

    # Parameters to analyze
    parameters = ["n_genes", "n_cells", "n_dims", "weighted_graph", "approx_neighbors"]
    impact_scores = {}

    for param in parameters:
        if param in ["weighted_graph", "approx_neighbors"]:
            # For boolean parameters, compare means between groups
            group_true = df[df[param]]["sc_simvar_time"]
            group_false = df[~df[param]]["sc_simvar_time"]

            if len(group_true) > 0 and len(group_false) > 0:
                # Use effect size (Cohen's d) as impact measure
                pooled_std = sqrt(  # type: ignore
                    ((len(group_true) - 1) * group_true.var() + (len(group_false) - 1) * group_false.var())  # type: ignore
                    / (len(group_true) + len(group_false) - 2)  # type: ignore
                )
                cohens_d = abs(group_true.mean() - group_false.mean()) / pooled_std
                impact_scores[param] = cohens_d
            else:
                impact_scores[param] = 0
        else:
            # For numerical parameters, use correlation with runtime
            correlation = abs(df[param].corr(df["sc_simvar_time"]))
            impact_scores[param] = correlation if not isnan(correlation) else 0

    # Sort by impact (descending)
    sorted_impact = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)

    print("Parameter Impact on SC SimVar Runtime (ordered by impact):")
    print("=" * 58)
    for i, (param, score) in enumerate(sorted_impact, 1):
        print(f"{i}. {param:<20} Impact Score: {score:.4f}")


if __name__ == "__main__":
    # adhoc_benchmark()
    # benchmark()
    # print("Benchmark completed.")

    results_file = "benchmark_results.json"
    with open(results_file, "r") as f:
        results = load(f)

    _analyze_parameter_impact(results)
    _create_benchmark_plots(results)
