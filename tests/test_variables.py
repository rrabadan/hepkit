import numpy as np
import pandas as pd
import pytest

from hepkit.variables import Var, arrays_from_vars, get_all_var_input_branches


def test_var_initialization():
    """Test basic Var object creation."""
    var = Var(
        name="pt",
        description="Transverse momentum",
        label=r"$p_{T}$ [GeV]",
        input_branches=["lep_pt"],
    )
    assert var.name == "pt"
    assert var.description == "Transverse momentum"
    assert var.label == r"$p_{T}$ [GeV]"
    assert var.input_branches == ["lep_pt"]
    assert not var.is_computed


def test_var_simple_factory():
    """Test the Var.simple class method."""
    var = Var.simple("mu_eta")
    assert var.name == "mu_eta"
    assert "Mu Eta" in var.label
    assert var.description == "mu_eta"


def test_var_computation():
    """Test variable computation from input branches."""
    # Expression that sums two branches
    var = Var(name="sum_pt", input_branches=["pt1", "pt2"], expression=lambda p1, p2: p1 + p2)

    data = {"pt1": np.array([10.0, 20.0]), "pt2": np.array([5.0, 15.0])}

    result = var.compute_array(data)
    expected = np.array([15.0, 35.0])
    np.testing.assert_array_equal(result, expected)
    assert var.is_computed


def test_var_compute_single_branch():
    """Test computation where expression just transforms one branch."""
    var = Var(name="abs_eta", input_branches="eta", expression=lambda x: np.abs(x))
    data = {"eta": np.array([-1.5, 2.0, -0.5])}
    result = var.compute_array(data)
    expected = np.array([1.5, 2.0, 0.5])
    np.testing.assert_array_equal(result, expected)


def test_var_with_pandas():
    """Test that Var works with pandas DataFrames."""
    var = Var(name="x", input_branches="a")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    result = var.compute_array(df)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_missing_branches_raises():
    """Test that missing branches cause a ValueError."""
    var = Var(name="x", input_branches=["a", "b"])
    data = {"a": np.array([1, 2])}  # 'b' is missing

    with pytest.raises(ValueError, match="Missing branches"):
        var.compute_array(data)


def test_arrays_from_vars():
    """Test the bulk extraction function."""
    v1 = Var(name="x", input_branches="a")
    v2 = Var(name="y", input_branches="b", expression=lambda b: b * 2)

    data = {"a": np.array([1, 2]), "b": np.array([10, 20])}
    results = arrays_from_vars([v1, v2], data)

    assert len(results) == 2
    np.testing.assert_array_equal(results[0], np.array([1, 2]))
    np.testing.assert_array_equal(results[1], np.array([20, 40]))


def test_get_all_input_branches():
    """Test extraction of unique branch names."""
    v1 = Var(name="v1", input_branches=["a", "b"])
    v2 = Var(name="v2", input_branches=["b", "c"])

    branches = get_all_var_input_branches([v1, v2])
    assert sorted(branches) == ["a", "b", "c"]
