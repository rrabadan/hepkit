import hist
import numpy as np

from hepkit.histograms import (
    _process_weight,
    axis_from_var,
    histogram_from_axes,
    quick_hist1d,
)
from hepkit.variables import Var


def mockvar():
    """Mock Var-like object for testing axis creation."""
    var = Var(
        name="pT",
        description="Leading photon transverse momentum",
        label="pT",
        x_title="pT",
        binning=(50, 0, 500),
        unit="GeV/c",
        input_branches=["lp_pt"],
    )
    return var


def test_axis_from_var_regular():
    """Test creating a regular axis from a variable."""
    var = mockvar()
    axis = axis_from_var(var)
    assert isinstance(axis, hist.axis.Regular)
    assert axis.size == 50
    assert axis.edges[0] == 0
    assert axis.edges[-1] == 500
    assert axis.label == "pT"


def test_histogram_from_axes():
    """Test basic histogram shell creation."""
    axis = hist.axis.Regular(10, 0, 10, name="x")
    h = histogram_from_axes([axis], "test_hist")
    assert isinstance(h, hist.Hist)
    assert h.name == "test_hist"
    assert len(h.axes) == 1


def test_process_weight_string():
    """Test weight processing from a branch name."""
    data = {"weights": np.array([0.5, 1.5])}
    w = _process_weight("weights", data)
    np.testing.assert_array_equal(w, np.array([0.5, 1.5]))


def test_process_weight_array():
    """Test weight processing from an existing array."""
    weights = np.array([1.0, 2.0])
    w = _process_weight(weights, {})
    np.testing.assert_array_equal(w, weights)


def test_quick_hist1d():
    """Test the quick plot utility."""
    data = np.array([1, 2, 2, 3, 3, 3])
    h = quick_hist1d(data, name="h", label="L", bins=3, range_=(0, 4))
    axes = h.axes
    assert h.sum().value == 6
    assert h.name == "h"
    assert len(axes) == 1
    assert isinstance(axes[0], hist.axis.Regular)
    assert axes[0].size == 3
    assert axes[0].edges[0] == 0
    assert axes[0].edges[-1] == 4
    assert axes[0].label == "L"
    assert axes[0].name == "h"
