# hepkit

Basic tools for High Energy Physics (HEP) analysis.

## Features

- **Variable Management**: Extend order.Variable with additional features for analysis and plotting.
- **Histogramming**: Interface built on top of `hist` for generating 1D and n-dimensional histograms directly from variable definitions.
- **HEP Plotting**: Histogram plotting utilities (multi-variate comparisons, signal vs. background).
- **Machine Learning**: Deterministic event-based splitting, data preprocessing for BDTs, and performance evaluation metrics.
- **Figure Management**: Context managers for automated figure creation, styling, and saving in multiple formats.

## Core Modules

### 1. Variables (`hepkit.variables`)
Define your variables once and compute them from different data sources (Dicts, Pandas, etc.).

```python
from hepkit.variables import Var

# Define a computed variable
pt_sum = Var(
    name="lp_pt_sum",
    label=r"$\sum p_{T}$ [GeV]",
    input_branches=["lep1_pt", "lep2_pt"],
    expression=lambda p1, p2: p1 + p2,
    binning=(50, 0, 500)
)

# Compute from a DataFrame
data = {"lep1_pt": [10, 20], "lep2_pt": [5, 15]}
array = pt_sum.compute_array(data) # [15, 35]
```

### 2. Histograms (`hepkit.histograms`)
Create and fill histograms directly from `Var` objects.

```python
from hepkit.histograms import hist1d_from_var

# Create and fill a 1D histogram
h = hist1d_from_var(pt_sum, data)

# Quick comparison of multiple histograms
from hepkit.histograms import multi_hist1d_comparison
fig, axes = multi_hist1d_comparison([sig_hists, bkg_hists], legends=["Signal", "Background"])
```

### 3. Plotting & Styles (`hepkit.plotting`)
Apply experiment-specific styles and manage figure saving.

```python
from hepkit.plotting import figure_context, set_cms_style

set_cms_style()

with figure_context("my_plot", prefix="analysis_v1_", format="pdf") as (fig, ax):
    # Standard matplotlib/mplhep plotting here
    ax.set_title("Analysis Results")
```

### 4. Classification & ML (`hepkit.classification`)

#### Data Preprocessing and Splitting
Ensure deterministic and reproducible results by splitting datasets based on unique event identifiers.

```python
from hepkit.classification.preprocessing import split_train_test_by_unique_id, prepare_training_data

# Deterministic split based on runNumber and eventNumber
train, test = split_train_test_by_unique_id(df, df_ids, test_ratio=0.3)

# Prepare signal and background with weights and transformations
X, y, w = prepare_training_data(sig_df, bkg_df, sig_vars, bkg_vars)
```

#### Evaluation and Visualization

```python
from hepkit.classification.evaluation import calculate_hep_metrics
from hepkit.classification.visualization import plot_train_test_response, plot_correlations

# Calculate S/sqrt(B) and efficiency
metrics = calculate_hep_metrics(y_true, y_pred, y_proba)

# Visualize classifier separation and check for overtraining
plot_train_test_response(clf, X_train, y_train, X_test, y_test)

# Plot feature correlations
plot_correlations(df, columns=["pt", "eta", "phi"])
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/
```
