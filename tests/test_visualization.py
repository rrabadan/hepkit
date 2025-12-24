import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hepkit.classification.visualization import plot_feature_importance, plot_roc_auc


class TestPlotRocAuc:
    """Test plot_roc_auc function with different styles and inputs."""

    @pytest.fixture
    def sample_data(self):
        """Create sample binary classification data."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)
        return y_true, y_scores

    @pytest.fixture
    def sample_multi_data(self):
        """Create sample data for multiple ROC curves."""
        np.random.seed(42)
        y_true_1 = np.random.randint(0, 2, 50)
        y_scores_1 = np.random.rand(50)
        y_true_2 = np.random.randint(0, 2, 50)
        y_scores_2 = np.random.rand(50)
        return [y_true_1, y_true_2], [y_scores_1, y_scores_2]

    def test_standard_style_single_curve(self, sample_data):
        """Test standard ROC style with single curve."""
        y_true, y_scores = sample_data
        fig = plot_roc_auc(y_true, y_scores, style="standard")
        assert fig is not None
        assert hasattr(fig, "axes")

    def test_efficiency_style_single_curve(self, sample_data):
        """Test efficiency style with single curve."""
        y_true, y_scores = sample_data
        fig = plot_roc_auc(y_true, y_scores, style="efficiency")
        assert fig is not None

    def test_rejection_style_single_curve(self, sample_data):
        """Test rejection style with single curve."""
        y_true, y_scores = sample_data
        fig = plot_roc_auc(y_true, y_scores, style="rejection")
        assert fig is not None

    def test_multiple_curves_standard(self, sample_multi_data):
        """Test multiple curves with standard style."""
        y_true_list, y_scores_list = sample_multi_data
        fig = plot_roc_auc(y_true_list, y_scores_list, style="standard")
        assert fig is not None

    def test_custom_labels(self, sample_data):
        """Test with custom labels."""
        y_true, y_scores = sample_data
        fig = plot_roc_auc(y_true, y_scores, labels="Custom Model")
        assert fig is not None

    def test_fig_size_parameter(self, sample_data):
        """Test custom figure size."""
        y_true, y_scores = sample_data
        fig = plot_roc_auc(y_true, y_scores, fig_size=(10, 8))
        assert fig is not None
        # Note: matplotlib figure size testing would require more complex setup

    def test_invalid_style_raises_error(self, sample_data):
        """Test that invalid style raises ValueError."""
        y_true, y_scores = sample_data
        with pytest.raises(ValueError, match="Unknown style"):
            plot_roc_auc(y_true, y_scores, style="invalid")


class TestPlotFeatureImportance:
    """Test plot_feature_importance function with different model types."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        return X, y

    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]

    def test_random_forest_model(self, sample_data, feature_names):
        """Test with RandomForest model."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        fig = plot_feature_importance(model, feature_names=feature_names)
        assert fig is not None

    def test_linear_model(self, sample_data, feature_names):
        """Test with LogisticRegression model."""
        X, y = sample_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        fig = plot_feature_importance(model, feature_names=feature_names)
        assert fig is not None

    def test_direct_importances(self, feature_names):
        """Test with direct importance array."""
        importances = np.array([0.1, 0.3, 0.2, 0.05, 0.35])
        fig = plot_feature_importance(None, feature_names=feature_names, importances=importances)
        assert fig is not None

    def test_top_n_parameter(self, sample_data, feature_names):
        """Test top_n parameter."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        fig = plot_feature_importance(model, feature_names=feature_names, top_n=3)
        assert fig is not None

    def test_auto_feature_names(self, sample_data):
        """Test automatic feature name generation."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        fig = plot_feature_importance(model)  # No feature_names provided
        assert fig is not None

    def test_fig_size_parameter(self, sample_data, feature_names):
        """Test custom figure size."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        fig = plot_feature_importance(model, feature_names=feature_names, fig_size=(12, 8))
        assert fig is not None

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        feature_names = ["a", "b", "c"]
        importances = np.array([0.1, 0.2])  # Different length

        with pytest.raises(ValueError, match="Length of importances must match feature_names"):
            plot_feature_importance(None, feature_names=feature_names, importances=importances)

    def test_unsupported_model_raises_error(self):
        """Test that unsupported model raises ValueError."""

        class UnsupportedModel:
            pass

        model = UnsupportedModel()
        with pytest.raises(
            ValueError, match="Model does not have feature_importances_ or coef_ attribute"
        ):
            plot_feature_importance(model)
