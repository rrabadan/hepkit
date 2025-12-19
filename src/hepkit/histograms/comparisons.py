import matplotlib.pyplot as plt
import numpy as np
from hist import Hist

from ..plotting import get_color_palette
from .plotting import plot_hist1d


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
        nrows=num_rows,
        ncols=num_cols,
        figsize=figsize,
        squeeze=False,  # Always return 2D array
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
