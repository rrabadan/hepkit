import numpy as np
import pandas as pd
import pytest

from hepkit.classification.preprocessing import VarInputTransformer, prepare_training_data
from hepkit.variables import Var


class TestVarInputTransformer:
    """Test suite for VarInputTransformer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "branch1": [1.0, 2.0, 3.0, 4.0],  # Float to test dtype preservation
                "branch2": [10, 20, 30, 40],
                "branch3": [100, 200, 300, 400],
            }
        )

    @pytest.fixture
    def simple_vars(self):
        """Create simple Var objects for testing."""
        return [
            Var(name="feature1", input_branches=["branch1"], expression="branch1"),  # Identity
            Var(
                name="feature2", input_branches=["branch2"], expression=lambda x: x * 2
            ),  # Computed
        ]

    def test_transform_identity(self, sample_data, simple_vars):
        """Test transformation with identity expression."""
        transformer = VarInputTransformer(vars=simple_vars)
        result = transformer.fit_transform(sample_data)

        expected = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0],
                "feature2": [20, 40, 60, 80],  # branch2 * 2
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_transform_with_nan(self, simple_vars):
        """Test that NaN values are dropped."""
        data_with_nan = pd.DataFrame(
            {
                "branch1": [1, np.nan, 3, 4],
                "branch2": [10, 20, np.nan, 40],
            }
        )
        transformer = VarInputTransformer(vars=simple_vars)
        result = transformer.fit_transform(data_with_nan)

        # Should drop rows with NaN in any feature
        expected = pd.DataFrame(
            {
                "feature1": [1.0, 4.0],
                "feature2": [20.0, 80.0],
            }
        )
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_categorical_variables_as_int(self, sample_data):
        """Test categorical variable handling as int."""
        vars_with_cat = [
            Var(name="feature1", input_branches=["branch1"], expression="branch1"),
            Var(name="cat_feature", input_branches=["branch2"], expression="branch2"),
        ]
        transformer = VarInputTransformer(
            vars=vars_with_cat, cat_vars=["cat_feature"], cat_as_int=True
        )
        result = transformer.fit_transform(sample_data)

        assert result["cat_feature"].dtype == int
        assert result["feature1"].dtype != int  # Should remain float or original

    def test_categorical_variables_as_category(self, sample_data):
        """Test categorical variable handling as category."""
        vars_with_cat = [
            Var(name="feature1", input_branches=["branch1"], expression="branch1"),
            Var(name="cat_feature", input_branches=["branch2"], expression="branch2"),
        ]
        transformer = VarInputTransformer(
            vars=vars_with_cat, cat_vars=["cat_feature"], cat_as_int=False
        )
        result = transformer.fit_transform(sample_data)

        assert result["cat_feature"].dtype.name == "category"

    def test_get_feature_names_out(self, simple_vars):
        """Test get_feature_names_out method."""
        transformer = VarInputTransformer(vars=simple_vars)
        names = transformer.get_feature_names_out()
        assert names == ["feature1", "feature2"]

    def test_missing_categorical_column(self, sample_data, simple_vars):
        """Test that missing categorical columns are ignored."""
        transformer = VarInputTransformer(vars=simple_vars, cat_vars=["nonexistent"])
        result = transformer.fit_transform(sample_data)
        # Should work without error, ignoring nonexistent cat var
        assert "feature1" in result.columns


class TestPrepareTrainingData:
    """Test suite for prepare_training_data function."""

    @pytest.fixture
    def sig_data(self):
        """Sample signal data."""
        return pd.DataFrame(
            {
                "branch1": [1, 2, 3],
                "branch2": [10, 20, 30],
                "runNumber": [100, 100, 101],
                "eventNumber": [1000, 1001, 1000],
                "candNumber": [1, 1, 2],
            }
        )

    @pytest.fixture
    def bkg_data(self):
        """Sample background data."""
        return pd.DataFrame(
            {
                "branch1": [4, 5, 6],
                "branch2": [40, 50, 60],
                "runNumber": [102, 102, 103],
                "eventNumber": [2000, 2001, 2000],
                "candNumber": [1, 2, 1],
            }
        )

    @pytest.fixture
    def input_vars(self):
        """Input variables for both sig and bkg."""
        return [
            Var(name="feature1", input_branches=["branch1"], expression="branch1"),
            Var(name="feature2", input_branches=["branch2"], expression=lambda x: x / 10),
        ]

    def test_basic_preparation(self, sig_data, bkg_data, input_vars):
        """Test basic data preparation without weights."""
        combined = prepare_training_data(
            sig_df=sig_data,
            bkg_df=bkg_data,
            sig_inputvars=input_vars,
            bkg_inputvars=input_vars,
        )

        # Check combined data
        assert len(combined) == 6
        assert "feature1" in combined.columns
        assert "feature2" in combined.columns
        assert "label" in combined.columns
        assert "runNumber" not in combined.columns  # IDs not included by default

        # Check labels
        assert all(combined.loc[:2, "label"] == 1)  # Signal
        assert all(combined.loc[3:, "label"] == 0)  # Background

    def test_with_weights(self, sig_data, bkg_data, input_vars):
        """Test preparation with weights."""
        sig_weights = np.array([1.0, 2.0, 3.0])
        bkg_weights = np.array([0.5, 1.5, 2.5])

        combined = prepare_training_data(
            sig_df=sig_data,
            bkg_df=bkg_data,
            sig_inputvars=input_vars,
            bkg_inputvars=input_vars,
            sig_weights=sig_weights,
            bkg_weights=bkg_weights,
        )

        assert "weights" in combined.columns
        pd.testing.assert_series_equal(
            combined["weights"],
            pd.Series([1.0, 2.0, 3.0, 0.5, 1.5, 2.5], name="weights"),
            check_names=False,
        )

    def test_custom_id_columns(self, sig_data, bkg_data, input_vars):
        """Test with custom ID columns."""
        custom_ids = ["runNumber", "eventNumber"]
        combined = prepare_training_data(
            sig_df=sig_data,
            bkg_df=bkg_data,
            sig_inputvars=input_vars,
            bkg_inputvars=input_vars,
            id_columns=custom_ids,
        )

        assert list(combined[custom_ids].columns) == custom_ids
        assert "candNumber" not in combined.columns  # Not in IDs, so not included

    def test_missing_id_columns(self, sig_data, bkg_data, input_vars):
        """Test error when ID columns are missing."""
        sig_data_missing = sig_data.drop(columns=["runNumber"])
        with pytest.raises(KeyError, match="ID columns missing from signal data"):
            prepare_training_data(
                sig_df=sig_data_missing,
                bkg_df=bkg_data,
                sig_inputvars=input_vars,
                bkg_inputvars=input_vars,
                id_columns=[
                    "runNumber",
                    "eventNumber",
                    "candNumber",
                ],  # Specify to trigger validation
            )

    def test_with_categorical_vars(self, sig_data, bkg_data, input_vars):
        """Test with categorical variables."""
        combined = prepare_training_data(
            sig_df=sig_data,
            bkg_df=bkg_data,
            sig_inputvars=input_vars,
            bkg_inputvars=input_vars,
            cat_vars=["feature1"],
            cat_as_int=True,
        )

        assert combined["feature1"].dtype == int

    def test_nan_handling(self, input_vars):
        """Test NaN handling in combined data."""
        sig_data = pd.DataFrame(
            {
                "branch1": [1, np.nan, 3],
                "branch2": [10, 20, 30],
                "runNumber": [100, 100, 101],
                "eventNumber": [1000, 1001, 1000],
                "candNumber": [1, 1, 2],
            }
        )
        bkg_data = pd.DataFrame(
            {
                "branch1": [4, 5, 6],
                "branch2": [40, 50, 60],
                "runNumber": [102, 102, 103],
                "eventNumber": [2000, 2001, 2000],
                "candNumber": [1, 2, 1],
            }
        )

        combined = prepare_training_data(
            sig_df=sig_data,
            bkg_df=bkg_data,
            sig_inputvars=input_vars,
            bkg_inputvars=input_vars,
        )

        # Should have 5 rows (6 - 1 NaN)
        assert len(combined) == 5
