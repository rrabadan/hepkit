from typing import Any

import hist
import numpy as np
from hist import Hist

from ..variables import Var, arrays_from_vars


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
    data: Any,
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
    if hasattr(data, "columns"):  # pandas DataFrame
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
    data: Any,
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
    if hasattr(data, "columns"):  # pandas DataFrame
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


def _process_weight(weight: str | np.ndarray | None, arr: Any) -> np.ndarray | None:
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
