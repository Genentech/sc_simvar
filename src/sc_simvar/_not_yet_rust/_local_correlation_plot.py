"""Create a local correlation plot."""

from matplotlib.colors import Colormap
from matplotlib.pyplot import gcf, get_cmap, sca, text, xlabel, xticks, ylabel
from numpy import arange, uint64
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy.cluster.hierarchy import leaves_list
from seaborn import clustermap


def local_correlation_plot(
    local_correlation_z: DataFrame,
    modules: "Series[int]",
    linkage: NDArray[uint64] | None,
    *,
    mod_cmap: str | Colormap = "tab10",
    vmin: int = -8,
    vmax: int = 8,
    z_cmap: str | Colormap = "RdBu_r",
    yticklabels: bool = False,
) -> None:
    row_colors = None
    cmap = get_cmap(mod_cmap)
    _max = modules.max()

    module_colors = {i: cmap(i / _max) for i in modules.unique()}
    module_colors[-1] = "#ffffff"
    row_colors = DataFrame({"Modules": [module_colors[i] for i in modules]}, index=local_correlation_z.index)

    cm = clustermap(
        local_correlation_z,
        row_linkage=linkage,
        col_linkage=linkage,
        vmin=vmin,
        vmax=vmax,
        cmap=z_cmap,
        xticklabels=False,
        yticklabels=yticklabels,
        row_colors=row_colors,
        rasterized=True,
    )

    fig = gcf()
    sca(cm.ax_heatmap)
    ylabel("")
    xlabel("")

    cm.ax_row_dendrogram.remove()

    # Add 'module X' annotations
    mod_reordered = modules.iloc[leaves_list(linkage)]
    y = arange(modules.size)
    mod_map = {}
    for x in mod_reordered.unique():
        if x == -1:
            continue

        mod_map[x] = y[mod_reordered == x].mean()

    if cm.ax_row_colors is None:
        raise ValueError("Row colors are not present in the clustermap.")

    sca(cm.ax_row_colors)
    for mod, mod_y in mod_map.items():
        text(
            -0.5, y=mod_y, s="Module {}".format(mod), horizontalalignment="right", verticalalignment="center"
        )

    xticks([])

    # Find the colorbar 'child' and modify
    min_delta = 1e99
    min_aa = None
    for aa in fig.get_children():
        if aa.axes is None:
            continue

        bbox = aa.axes.get_position()
        delta = (0 - bbox.xmin) ** 2 + (1 - bbox.ymax) ** 2
        if delta < min_delta:
            delta = min_delta
            min_aa = aa

    if min_aa is not None and min_aa.axes is not None:
        min_aa.axes.set_ylabel("Z-Scores")
        min_aa.axes.yaxis.set_label_position("left")
