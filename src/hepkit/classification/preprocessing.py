import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin

from ..variables import Var

# Ensure sklearn outputs DataFrames where applicable
set_config(transform_output="pandas")


def _apply_transforms(df: pd.DataFrame, inputvars: list[Var]) -> pd.DataFrame:
    """
    Helper function to apply transformations to input variables.

    Parameters:
    - df: DataFrame containing the raw data
    - inputvars: List of variable objects with name, branch, and expression attributes

    Returns:
    - DataFrame with transformed features
    """
    features = [var.name for var in inputvars]
    transforms = [var.expression for var in inputvars]
    branches = [var.input_branches for var in inputvars]

    transformed_data = {}
    for feature, branch_list, transform in zip(features, branches, transforms, strict=False):
        # Apply transform using the specified branches
        if isinstance(transform, str):
            # If transform is a string, apply identity function (return first branch)
            transformed_data[feature] = df[branch_list[0]]
        else:
            transformed_data[feature] = transform(*(df[b] for b in branch_list))

    return pd.DataFrame(transformed_data)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from a DataFrame."""

    def __init__(self, column_names: list[str]):
        """
        Initialize the ColumnSelector.

        Parameters:
        - column_names: List of column names to select
        """
        self.column_names = column_names

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnSelector":  # noqa: N803
        """Fit the transformer (no-op for column selection)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Select the specified columns from the DataFrame."""
        missing_cols = [col for col in self.column_names if col not in X.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
        return X[self.column_names].copy()


class VarInputTransformer(BaseEstimator, TransformerMixin):
    """
    A sklearn-compatible transformer for preparing input data using Var definitions.

    This transformer applies data transformations using provided Var definitions
    and handles categorical variables.

    Parameters
    ----------
    vars : list[Var]
        List of Var-like objects (with name, branch, expression attributes)
    cat_vars : list[str] | None, default None
        Categorical variable names to cast (as int or category)
    cat_as_int : bool, default True
        If True, cast categorical vars as int; else as "category"
    """

    def __init__(
        self,
        vars: list[Var],
        cat_vars: list[str] | None = None,
        cat_as_int: bool = True,
    ):
        self.vars = list(vars)
        self.cat_vars = cat_vars
        self.cat_as_int = cat_as_int

    def fit(self, X, y=None):
        """Fit method (no-op, required by sklearn)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input dataframe using the Var definitions.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with raw features

        Returns
        -------
        pd.DataFrame
            Transformed dataframe with features only
        """
        # Apply transformations using hepkit utility
        result_df = _apply_transforms(X, self.vars)

        # Handle categorical variables
        if self.cat_vars is not None:
            catdtype = int if self.cat_as_int else "category"
            for col in self.cat_vars:
                if col in result_df.columns:
                    result_df[col] = result_df[col].astype(catdtype)

        # Clean up any rows with NaN
        result_df.dropna(inplace=True)
        # Note: Keep original index for alignment in prepare_training_data

        return result_df

    def get_feature_names_out(self, input_features=None):
        """Return the feature names of the transformed output (sklearn convention)."""
        # Return names of vars that were transformed
        return [v.name for v in self.vars]


def prepare_training_data(
    sig_df: pd.DataFrame,
    bkg_df: pd.DataFrame,
    sig_inputvars: list[Var],
    bkg_inputvars: list[Var],
    sig_weights: np.ndarray | None = None,
    bkg_weights: np.ndarray | None = None,
    id_columns: list[str] | None = None,
    cat_vars: list[str] | None = None,
    cat_as_int: bool = True,
) -> pd.DataFrame:
    """
    Prepare the data for training by transforming the input variables and adding necessary columns.

    Args:
        sig_df: DataFrame containing the signal data.
        bkg_df: DataFrame containing the background data.
        sig_inputvars: List of input variables for the signal data.
        bkg_inputvars: List of input variables for the background data.
        sig_weights: Array-like object containing the weights for the signal data.
        bkg_weights: Array-like object containing the weights for the background data.
        id_columns: List of ID column names. If None, ID columns are not included in the output.
        cat_vars: List of categorical variable names.
        cat_as_int: If True, cast categorical vars as int; else as "category".

    Returns:
        DataFrame with features, labels, weights (if provided), and ID columns (if id_columns is not None).
    """
    if id_columns is not None:
        # Validate ID columns exist
        missing_sig_cols = [col for col in id_columns if col not in sig_df.columns]
        missing_bkg_cols = [col for col in id_columns if col not in bkg_df.columns]
        if missing_sig_cols:
            raise KeyError(f"ID columns missing from signal data: {missing_sig_cols}")
        if missing_bkg_cols:
            raise KeyError(f"ID columns missing from background data: {missing_bkg_cols}")

    # Use ModelInputTransformer for transformations (no ID handling here)
    sig_transformer = VarInputTransformer(
        vars=sig_inputvars, cat_vars=cat_vars, cat_as_int=cat_as_int
    )
    bkg_transformer = VarInputTransformer(
        vars=bkg_inputvars, cat_vars=cat_vars, cat_as_int=cat_as_int
    )

    sig_transformed = sig_transformer.fit_transform(sig_df)
    bkg_transformed = bkg_transformer.fit_transform(bkg_df)

    # Add ID columns if provided
    if id_columns is not None:
        for col in id_columns:
            sig_transformed[col] = sig_df.loc[sig_transformed.index, col].values
            bkg_transformed[col] = bkg_df.loc[bkg_transformed.index, col].values

    # Add weights if provided
    if sig_weights is not None:
        sig_transformed["weights"] = sig_weights[sig_transformed.index]
    if bkg_weights is not None:
        bkg_transformed["weights"] = bkg_weights[bkg_transformed.index]

    # Add labels
    sig_transformed["label"] = 1
    bkg_transformed["label"] = 0

    # Combine datasets
    combined_data = pd.concat([sig_transformed, bkg_transformed], ignore_index=True)

    # Remove rows with NaN values (additional safety)
    combined_data.dropna(inplace=True)
    combined_data.reset_index(drop=True, inplace=True)

    return combined_data
