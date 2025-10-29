from typing import Any

import hist
import matplotlib.pyplot as plt
import mplhep
import numpy as np
from hist import Hist

from .variables import Var, arrays_from_vars
from .plotting import get_color_palette


def axis_from_var(var: Var, **kwargs) -> hist.axis.AxesMixin:
    """
    Create a histogram axis from a Var object.

    Args:
        var: Variable object with attributes:
            - x_min, x_max: Range of the axis
            - n_bins: Number of bins
            - label: Axis label

    Returns:
        Hist axis object configured with the variable's properties

    Raises:
        AttributeError: If required attributes are missing
        ValueError: If variable configuration is invalid
    """

    if not hasattr(var, "x_min") or not hasattr(var, "x_max"):
        raise AttributeError("Variable must have 'x_min' and 'x_max' attributes")

    if var.x_min >= var.x_max:
        raise ValueError(f"Invalid variable range: x_min={var.x_min}, x_max={var.x_max}")

    if not hasattr(var, "x_discrete"):
        raise AttributeError("Variable must have 'x_discrete' attribute")

    underflow = kwargs.get("underflow", False)
    overflow = kwargs.get("overflow", False)

    if var.x_discrete:
        axis = hist.axis.Integer(
            int(var.x_min),
            int(var.x_max),
            name=var.name,
            label=var.label,
            underflow=underflow,
            overflow=overflow,
        )
        return axis

    if not hasattr(var, "n_bins"):
        raise AttributeError("Non discrete variable must have 'n_bins' attribute")

    if hasattr(var, "binning") and len(var.binning) != 3:
        if not hasattr(var, "bin_edges"):
            raise AttributeError("Variable must have 'bin_edges' attribute for variable binning")
        axis = hist.axis.Variable(
            var.bin_edges,
            name=var.name,
            label=var.label,
            underflow=underflow,
            overflow=overflow,
        )
        return axis

    axis = hist.axis.Regular(
        var.n_bins,
        var.x_min,
        var.x_max,
        name=var.name,
        label=var.label,
        underflow=underflow,
        overflow=overflow,
    )

    return axis


def histogram_from_axes(
    axes: list[hist.axis.AxesMixin],
    name: str,
    storage: hist.storage.Storage | None = None,
):
    """
    Create a histogram from a list of axes.

    Args:
        axes: List of hist.axis.AxesMixin objects to define the histogram axes
        name: Name identifier for the histogram
        storage: Optional storage type for the histogram (default: hist.storage.Weight())

    Returns:
        Empty histogram object
    """
    if storage is None:
        storage = hist.storage.Weight()

    h = Hist(*axes, storage=storage)
    h.name = name

    return h


def hist1d_from_var(
    var: Var,
    data: dict[str, Any],
    weight: str | np.ndarray | None = None,
    **kwargs,
) -> Hist:
    """
    Create and fill a 1D histogram from a Var object.

    Args:
        var: Variable object with required attributes
        arr: Dictionary-like object containing branch data
        weight: Optional weights (branch name or array)
        **kwargs: Additional configuration passed to axis creation

    Returns:
        Filled 1D histogram

    Raises:
        AttributeError: If required variable attributes are missing
        ValueError: If variable configuration is invalid
    """
    # Create axis from variable
    axis = axis_from_var(var, **kwargs)

    # Create empty histogram
    histogram = histogram_from_axes([axis], var.name)

    # Pre-filter data to only needed columns for performance
    if hasattr(data, 'columns'):  # pandas DataFrame
        needed_cols = set(var.input_branches)
        if isinstance(weight, str):
            needed_cols.add(weight)
        if needed_cols:
            data_filtered = data[list(needed_cols)]
        else:
            data_filtered = data
    else:
        # For dict-like objects, keep as-is
        data_filtered = data

    # Get data using Var's functionality
    array = var.compute_array(data_filtered)

    # Handle weights
    weight_array = _process_weight(weight, data_filtered)

    # Fill histogram
    try:
        histogram.fill(array, weight=weight_array)
    except Exception as e:
        raise RuntimeError(f"Error filling histogram for variable '{var.name}': {e}") from e

    return histogram


def hist_nd_from_vars(
    variables: list[Var],
    data: dict[str, Any],
    name: str,
    weight: str | np.ndarray | None = None,
    **kwargs,
) -> Hist:
    """
    Create and fill an N-dimensional histogram from multiple Var objects.

    Args:
        variables: List of Var objects (one per dimension)
        arr: Dictionary-like object containing branch data
        name: Name for the histogram
        weight: Optional weights
        **kwargs: Additional configuration for axes

    Returns:
        Filled N-dimensional histogram
    """
    if not variables:
        raise ValueError("At least one variable must be provided")

    # Create axes from variables
    axes = [axis_from_var(var, **kwargs) for var in variables]

    # Create empty histogram
    histogram = histogram_from_axes(axes, name)

    # Pre-filter data to only needed columns for performance
    if hasattr(data, 'columns'):  # pandas DataFrame
        needed_cols = set()
        for var in variables:
            needed_cols.update(var.input_branches)
        if isinstance(weight, str):
            needed_cols.add(weight)
        if needed_cols:
            data_filtered = data[list(needed_cols)]
        else:
            data_filtered = data
    else:
        # For dict-like objects, keep as-is
        data_filtered = data

    # Get data from all variables
    arrays = arrays_from_vars(variables, data_filtered)

    # Handle weights
    weight_array = _process_weight(weight, data_filtered)

    # Fill histogram
    try:
        histogram.fill(*arrays, weight=weight_array)
    except Exception as e:
        raise RuntimeError(f"Error filling {len(variables)}D histogram: {e}") from e

    return histogram


def _process_weight(weight: str | np.ndarray | None, arr: dict[str, Any]) -> np.ndarray | None:
    """
    Process weight parameter into a weight array.

    Args:
        weight: Weight specification (branch name or array)
        arr: Dictionary containing branch data

    Returns:
        Weight array or None
    """
    if weight is None:
        return None

    if isinstance(weight, str):
        if weight not in arr:
            raise ValueError(f"Weight branch '{weight}' not found in array")
        return arr[weight]
    else:
        return weight


def quick_hist1d(
    array: np.ndarray,
    name: str,
    label: str,
    bins: int = 50,
    range_: tuple[float, float] | None = None,
    weight: np.ndarray | None = None,
    **kwargs,
) -> Hist:
    """
    Quick histogram creation with automatic range detection.

    Args:
        arr: Dictionary-like object containing branch data
        branch: Branch name to histogram
        bins: Number of bins (default: 50)
        range_: Optional range tuple, auto-detected if None
        weight: Optional weights
        **kwargs: Additional configuration
    """

    # Auto-detect range if not provided
    if range_ is None:
        if len(array) == 0:
            raise ValueError("Cannot auto-detect range for empty data")
        range_ = (float(np.min(array)), float(np.max(array)))
        # Add small padding
        padding = (range_[1] - range_[0]) * 0.05
        range_ = (range_[0] - padding, range_[1] + padding)

    underflow = kwargs.get("underflow", False)
    overflow = kwargs.get("overflow", False)

    axis = hist.axis.Regular(
        bins,
        range_[0],
        range_[1],
        name=name,
        label=label,
        underflow=underflow,
        overflow=overflow,
    )

    histogram = histogram_from_axes([axis], name, storage=hist.storage.Weight())

    # Fill histogram
    try:
        histogram.fill(array, weight=weight)
    except Exception as e:
        raise RuntimeError(f"Error filling histogram '{name}': {e}") from e

    return histogram


def plot_hist1d(
    ax: plt.Axes,
    h: Hist,
    logy: bool = False,
    show_stats: bool = False,
    show_overflow: bool = False,
    show_underflow: bool = False,
    **kwargs,
) -> None:
    """
    Plot a histogram on the given axes with extensive customization options.

    Args:
        ax: Matplotlib axes to plot on
        h: Histogram object to plot
        logy: Use logarithmic y-scale (default: False)
        show_stats: Display statistics box (entries, mean, std) (default: False)
        show_overflow: Include overflow in flow parameter (default: False)
        show_underflow: Include underflow in flow parameter (default: False)
        **kwargs: Additional plotting options:
            # Style options
            - histtype: "step", "fill", "stepfilled" (default: "step")
            - color: Plot color (default: "black")
            - alpha: Transparency (default: 1.0)
            - linewidth/lw: Line width (default: 1.5)
            - linestyle/ls: Line style (default: "-")

            # Label and legend options
            - label: Legend label (default: histogram name or None)
            - show_label_in_legend: Include label in legend (default: True if label provided)

            # Data options
            - density: Normalize to density (default: False)
            - flow: Flow handling "hint", "show", "sum", "none" (default: auto-determined)

            # Axis options
            - ylabel: Y-axis label (default: "Candidates" or "Density")
            - xlabel: X-axis label (default: histogram label)
            - ylim_bottom: Y-axis bottom limit (default: 0)
            - ylim_top: Y-axis top limit (default: auto)
            - xlim: X-axis limits tuple (default: auto)

            # Grid and styling
            - grid: Show grid (default: False)
            - grid_alpha: Grid transparency (default: 0.3)

    Raises:
        ValueError: If histogram is empty or invalid
        TypeError: If histogram type is not supported
    """
    # Input validation
    if h is None:
        raise ValueError("Histogram cannot be None")

    if not hasattr(h, "values"):
        raise TypeError("Object must be a histogram with 'values' method")

    # Check if histogram is empty
    if h.sum() == 0:
        print(f"Warning: Histogram '{getattr(h, 'name', 'unnamed')}' is empty")

    # Extract style parameters
    histtype = kwargs.pop("histtype", "step")
    color = kwargs.pop("color", "black")
    alpha = kwargs.pop("alpha", 1.0)
    linewidth = kwargs.pop("linewidth", kwargs.pop("lw", 1.5))
    linestyle = kwargs.pop("linestyle", kwargs.pop("ls", "-"))

    # Extract label parameters
    label = kwargs.pop("label", getattr(h, "name", None))
    show_label_in_legend = kwargs.pop("show_label_in_legend", label is not None)

    # Extract data parameters
    density = kwargs.pop("density", False)

    # Determine flow parameter
    flow = kwargs.pop("flow", None)
    if flow is None:
        if show_overflow and show_underflow:
            flow = "show"
        elif show_overflow or show_underflow:
            flow = "hint"
        else:
            flow = "none"

    # Extract axis parameters
    ylabel = kwargs.pop("ylabel", "Density" if density else "Entries")
    xlabel = kwargs.pop("xlabel", getattr(h.axes[0], "label", ""))
    ylim_bottom = kwargs.pop("ylim_bottom", 0)
    ylim_top = kwargs.pop("ylim_top", None)
    xlim = kwargs.pop("xlim", None)

    # Extract grid parameters
    grid = kwargs.pop("grid", False)
    grid_alpha = kwargs.pop("grid_alpha", 0.3)

    # Plot histogram using mplhep
    try:
        mplhep.histplot(
            h,
            ax=ax,
            histtype=histtype,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            flow=flow,
            label=label if show_label_in_legend else None,
            density=density,
            **kwargs,  # Pass any remaining kwargs to mplhep.histplot
        )
    except Exception as e:
        raise RuntimeError(f"Error plotting histogram: {e}") from None

    # Configure axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set y-axis limits
    if logy:
        ax.set_yscale("log")
        # For log scale, set a small positive bottom limit
        current_ylim = ax.get_ylim()
        if ylim_bottom <= 0:
            ylim_bottom = max(0.1, current_ylim[0])

    if ylim_top is not None:
        ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
    else:
        ax.set_ylim(bottom=ylim_bottom)

    # Set x-axis limits if specified
    if xlim is not None:
        ax.set_xlim(xlim)

    # Configure grid
    if grid:
        ax.grid(True, alpha=grid_alpha)

    # Add statistics box if requested
    if show_stats:
        _add_stats_box(ax, h, density=density)


def _add_stats_box(ax: plt.Axes, h: Hist, density: bool = False) -> None:
    """Add a statistics text box to the plot."""
    try:
        entries = int(h.sum())

        # Calculate mean and std if possible
        if hasattr(h, "axes") and len(h.axes) > 0:
            # Get bin centers and values
            bin_centers = h.axes[0].centers
            bin_values = h.values()

            if entries > 0:
                # Calculate weighted mean and std
                weights = bin_values / np.sum(bin_values) if np.sum(bin_values) > 0 else bin_values
                mean = np.average(bin_centers, weights=weights)
                variance = np.average((bin_centers - mean) ** 2, weights=weights)
                std = np.sqrt(variance)

                stats_text = f"Entries: {entries}\nMean: {mean:.3f}\nStd: {std:.3f}"
            else:
                stats_text = f"Entries: {entries}\nMean: N/A\nStd: N/A"
        else:
            stats_text = f"Entries: {entries}"

        # Add text box
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    except Exception:
        # If stats calculation fails, just show entries
        try:
            entries = int(h.sum())
            ax.text(
                0.02,
                0.98,
                f"Entries: {entries}",
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        except Exception:
            pass  # If even basic stats fail, don't show anything


# Convenience functions for common plot styles
def plot_hist1d_filled(ax: plt.Axes, h: Hist, **kwargs) -> None:
    """Plot histogram with filled style."""
    kwargs.setdefault("histtype", "stepfilled")
    kwargs.setdefault("alpha", 0.7)
    plot_hist1d(ax, h, **kwargs)


def plot_hist1d_outline(ax: plt.Axes, h: Hist, **kwargs) -> None:
    """Plot histogram with outline style."""
    kwargs.setdefault("histtype", "step")
    kwargs.setdefault("linewidth", 2)
    plot_hist1d(ax, h, **kwargs)


def plot_hist1d_with_errors(ax: plt.Axes, h: Hist, **kwargs) -> None:
    """Plot histogram with error bars."""
    kwargs.setdefault("yerr", True)  # This depends on mplhep support
    plot_hist1d(ax, h, **kwargs)


def plot_hist1d_comparison(
    hists: list[Hist],
    legends: list[Hist],
    ax: plt.Axes,
    histtypes: list[str] | None = None,
    colors: list[str] | None = None,
    normalize: bool = True,
    auto_ylim: bool = True,
    ylim_margin: float = 0.05,
    **kwargs,
):
    """
    Plot multiple histograms on the same axes for comparison.

    Args:
        hists: List of histogram objects to plot
        legends: List of legend labels for each histogram
        ax: Matplotlib axes to plot on
        histtypes: List of histogram types for each plot (default: "step" for all)
        colors: List of colors for each histogram (default: automatic color palette)
        normalize: Whether to normalize histograms to density (default: True)
        auto_ylim: Automatically set y-axis limits (default: True)
        ylim_margin: Margin factor for y-axis top limit (default: 0.05)
        **kwargs: Additional arguments passed to plot_hist1d

    Raises:
        ValueError: If input lists have different lengths or are empty
        TypeError: If histogram objects are invalid
    """
    # Set defaults for optional parameters
    if histtypes is None:
        histtypes = ["step"] * len(hists)
    if colors is None:
        colors = get_color_palette(n_colors=len(hists))
    
    list_names = ["hists", "legends", "histtypes", "colors"]
    list_lengths = [len(hists), len(legends), len(histtypes), len(colors)]

    if not all(length == len(hists) for length in list_lengths[1:]):
        lengths = list_lengths
        raise ValueError(
            f"Inputs must have the same length. Got: {dict(zip(list_names, lengths, strict=False))}"
        )

    # Validate histogram objects
    for i, h in enumerate(hists):
        if not hasattr(h, "values") or not hasattr(h, "sum"):
            raise TypeError(f"hists[{i}] is not a valid histogram object")

    max_value = 0
    plotted_hists = []

    for h, legend, histtype, color in zip(hists, legends, histtypes, colors, strict=True):
        try:
            plot_hist1d(
                ax, h, label=legend, histtype=histtype, color=color, density=normalize, **kwargs
            )

            if auto_ylim:
                if normalize and hasattr(h, "density"):
                    current_max = np.max(h.density())
                else:
                    current_max = np.max(h.values())
                max_value = max(max_value, current_max)

            plotted_hists.append(h)

        except Exception as e:
            print(f"Warning: Failed to plot histogram '{legend}': {e}")
            continue

        if not plotted_hists:
            raise ValueError("No valid histograms were plotted. Check input data.")

        if auto_ylim and max_value > 0:
            # Set y-axis limits based on maximum value
            ax.set_ylim(bottom=0, top=max_value * (1 + ylim_margin))
        # max_density = max(max_density, max(h.density()))
        # ax.set_ylim(bottom=0, top=max_density * 1.05)
        # ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)

        if any(legends):
            ax.legend(loc="best", fontsize="small", frameon=False)


def multi_hist1d_comparison(
    hists: list[dict[str, Hist]],
    legends: list[str],
    histtypes: list[str],
    colors: list[str],
    figsize_per_plot: tuple[float, float] = (3, 2.5),
    max_cols: int = 3,
    normalize: bool = True,
    shared_legend: bool = True,
    legend_location: str = "upper right",
    tight_layout: bool = True,
    subplot_titles: bool = True,
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create multiple subplot comparisons of 1D histograms.

    Args:
        hists: List of dictionaries, each containing histograms with same keys
        legends: List of legend labels for each histogram set
        histtypes: List of histogram types for each set
        colors: List of colors for each histogram set
        figsize_per_plot: Size of each subplot (width, height)
        max_cols: Maximum columns before wrapping to new row
        normalize: Whether to normalize histograms to density (default: True)
        shared_legend: Use single legend for entire figure (default: True)
        legend_location: Location for shared legend (default: "upper right")
        tight_layout: Apply tight layout (default: True)
        subplot_titles: Show titles on subplots (default: True)
        **kwargs: Additional arguments passed to plot_hist1d_comparison

    Returns:
        Tuple of (figure, list of axes)

    Raises:
        ValueError: If input validation fails
        KeyError: If histogram dictionaries have inconsistent keys
    """
    # Input validation
    if not hists:
        raise ValueError("hists list cannot be empty")

    # input_lists = [hists, legends, histtypes, colors]
    list_names = ["hists", "legends", "histtypes", "colors"]

    if not all(len(lst) == len(hists) for lst in [legends, histtypes, colors]):
        lengths = [len(hists), len(legends), len(histtypes), len(colors)]
        raise ValueError(
            "All input lists must have the same length."
            + f" Got: {dict(zip(list_names, lengths, strict=False))}"
        )

    # Validate that all histogram dictionaries have the same keys
    reference_keys = set(hists[0].keys())
    for i, hist_dict in enumerate(hists[1:], 1):
        if set(hist_dict.keys()) != reference_keys:
            raise KeyError(f"Histogram dictionary {i} has different keys than the first one")

    # Get histogram keys and determine layout
    hist_keys = list(hists[0].keys())
    num_plots = len(hist_keys)

    if num_plots == 0:
        raise ValueError("No histograms found in dictionary")

    # Calculate optimal subplot layout
    if num_plots <= max_cols:
        num_rows, num_cols = 1, num_plots
    else:
        num_cols = min(max_cols, int(np.ceil(np.sqrt(num_plots))))
        num_rows = int(np.ceil(num_plots / num_cols))

    # Calculate figure size
    fig_width = num_cols * figsize_per_plot[0]
    fig_height = num_rows * figsize_per_plot[1]
    figsize = (fig_width, fig_height)

    # Create figure and axes
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=figsize, squeeze=False  # Always return 2D array
    )

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Plot histograms
    plotted_successfully = []

    for i, key in enumerate(hist_keys):
        ax = axes_flat[i]

        try:
            # Extract histograms for this key
            key_hists = [hist_dict[key] for hist_dict in hists]

            # Plot comparison on this subplot
            plot_hist1d_comparison(
                key_hists, legends, ax, histtypes, colors, normalize=normalize, **kwargs
            )

            # Set subplot title
            if subplot_titles:
                # Clean up the key for display
                title = key.replace("_", " ").title()
                ax.set_title(title, fontsize=10, pad=10)

            # Remove individual legends if using shared legend
            if shared_legend and ax.get_legend():
                ax.get_legend().remove()

            plotted_successfully.append(key)

        except Exception as e:
            print(f"Warning: Failed to plot histogram '{key}': {e}")
            # Hide the failed subplot
            ax.set_visible(False)
            continue

    # Hide unused subplots
    for i in range(len(hist_keys), len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Add shared legend
    if shared_legend and plotted_successfully:
        # Get handles and labels from the first successful plot
        first_plot_idx = hist_keys.index(plotted_successfully[0])
        first_ax = axes_flat[first_plot_idx]

        # Temporarily add legend to get handles
        temp_legend = first_ax.legend()
        handles, labels = first_ax.get_legend_handles_labels()
        if temp_legend:
            temp_legend.remove()

        if handles:
            # Position mapping for shared legend
            legend_positions = {
                "upper right": (0.98, 0.98),
                "upper left": (0.02, 0.98),
                "lower right": (0.98, 0.02),
                "lower left": (0.02, 0.02),
                "center right": (0.98, 0.5),
            }

            pos = legend_positions.get(legend_location, (0.98, 0.98))
            # ha = "right" if "right" in legend_location else "left"
            # va = (
            #     "top"
            #     if "upper" in legend_location
            #     else ("bottom" if "lower" in legend_location else "center")
            # )

            fig.legend(
                handles,
                labels,
                loc="center",
                bbox_to_anchor=pos,
                bbox_transform=fig.transFigure,
                fontsize="small",
                frameon=True,
                fancybox=True,
                shadow=True,
                # ha=ha,
                # va=va,
            )

    # Apply layout adjustments
    if tight_layout:
        plt.tight_layout()

        # Adjust layout to make room for shared legend if needed
        if shared_legend and plotted_successfully:
            if "right" in legend_location:
                plt.subplots_adjust(right=0.85)
            elif "left" in legend_location:
                plt.subplots_adjust(left=0.15)

    if not plotted_successfully:
        raise RuntimeError("Failed to plot any histograms successfully")

    return fig, axes_flat[: len(hist_keys)]
