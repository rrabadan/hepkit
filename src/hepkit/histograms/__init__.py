from .building import (
    _process_weight,
    axis_from_var,
    hist1d_from_var,
    hist_nd_from_vars,
    histogram_from_axes,
    quick_hist1d,
)
from .comparisons import multi_hist1d_comparison, plot_hist1d_comparison
from .plotting import (
    _add_stats_box,
    plot_hist1d,
    plot_hist1d_filled,
    plot_hist1d_outline,
    plot_hist1d_with_errors,
)

__all__ = [
    "axis_from_var",
    "histogram_from_axes",
    "hist1d_from_var",
    "hist_nd_from_vars",
    "quick_hist1d",
    "_process_weight",
    "plot_hist1d",
    "_add_stats_box",
    "plot_hist1d_filled",
    "plot_hist1d_outline",
    "plot_hist1d_with_errors",
    "plot_hist1d_comparison",
    "multi_hist1d_comparison",
]
