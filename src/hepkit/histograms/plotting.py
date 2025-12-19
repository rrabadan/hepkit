import matplotlib.pyplot as plt
import mplhep
import numpy as np
from hist import Hist


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
