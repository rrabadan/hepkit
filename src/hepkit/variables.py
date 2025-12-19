from collections.abc import Callable
from typing import Any

import numpy as np
import order as od


class Var(od.Variable):
    """Extended Variable class with name, description, label and input branches information.

    Args:
        name: Variable name
        description: Optional description of the variable
        label: Optional Human-readable label for plotting
        input_branches: Optional Single branch name or list of input branch names
        expression: Optional Function to transform the input data
        **kwargs: Additional arguments passed to od.Variable
    """

    def __init__(
        self,
        name: str,
        description: str | None = None,
        label: str | None = None,
        input_branches: str | list[str] | None = None,
        expression: Callable[[Any], Any] | None = None,
        *args,
        **kwargs,
    ):
        # Remove branch and expression from kwargs to avoid conflicts
        kwargs.pop("input_branches", None)
        kwargs.pop("description", None)
        kwargs.pop("label", None)
        kwargs.pop("expression", None)

        super().__init__(name, *args, **kwargs)

        self._description = description or name
        self._label = label or name

        # Handle input branches assignment
        if input_branches is None:
            self._input_branches = []  # Default to empty list if no branches provided
        elif isinstance(input_branches, str):
            self._input_branches = [input_branches]  # Single branch as list
        else:
            self._input_branches = input_branches

        # Handle expression assignment
        self._expression = expression  # Leave as None if not provided

    @property
    def description(self) -> str:
        """Get the description of the variable."""
        return self._description

    @property
    def label(self) -> str:
        """Get the human-readable label for the variable."""
        return self._label

    @property
    def input_branches(self) -> list[str]:
        """Get list of input branch names needed for this variable."""
        if not self.has_input_branches:
            raise ValueError(f"Variable '{self.name}' has no input branches defined.")
        return self._input_branches.copy()  # Return copy to prevent external modification

    @input_branches.setter
    def input_branches(self, branches: list[str] | str) -> None:
        """Set the input branches for this variable."""
        if not branches:
            raise ValueError("At least one branch name must be provided.")
        if isinstance(branches, str):
            self._input_branches = [branches]
        elif isinstance(branches, list):
            self._input_branches = list(set(branches))
        else:
            raise TypeError("input_branches must be string or list")

    @property
    def has_input_branches(self) -> bool:
        """Check if this variable has any input branches defined."""
        return bool(self._input_branches)

    @property
    def is_computed(self) -> bool:
        """Check if this variable requires computation."""
        return not isinstance(self.expression, str) or self._expression is not None

    def validate_branches(self, available_branches: list[str]) -> bool:
        """Check if all required branches are available."""
        return all(branch in available_branches for branch in self.input_branches)

    def get_branches(self) -> list[str]:
        """Alias for get_input_branches() for consistency."""
        return self.input_branches

    def __repr__(self) -> str:
        """String representation of the variable."""
        # branches_str = ", ".join(self.branch)
        return (
            f"Var(name='{self.name}', description='{self.description}'"
            + f", x_min='{self.x_min}', x_max='{self.x_max}', is_computed='{self.is_computed}')"
        )

    @classmethod
    def simple(cls, name: str, description: str | None = None, label: str | None = None) -> "Var":
        """Create a simple variable with minimal configuration."""
        return cls(
            name=name,
            description=description or name,
            label=label or name.replace("_", " ").title(),
        )

    def compute_from_branches(self, data: Any) -> np.ndarray:
        """
        Compute this variable's array from branches using its expression.

        Args:
            data: Dictionary-like object containing branch data

        Returns:
            Computed numpy array

        Raises:
            AttributeError: If variable doesn't have an expression
            ValueError: If required branches are missing
            RuntimeError: If transformation fails
        """

        # Check if we have an expression
        if self.expression is None:
            raise AttributeError(f"Variable '{self.name}' does not have an 'expression' attribute")

        try:
            branch_data = [data[b] for b in self.input_branches]
            if len(self.input_branches) == 1:
                result = self.expression(branch_data[0])
            else:
                result = self.expression(*branch_data)
            return result
        except Exception as e:
            raise RuntimeError(f"Error computing variable '{self.name}': {e}") from None

    def compute_array(self, data: Any) -> np.ndarray:
        """
        Compute this variable's data array, with or without transformation.

        Args:
            data: Dictionary-like object containing branch data

        Returns:
            Data array for the variable

        Raises:
            ValueError: If required branches are missing or no branches defined
            RuntimeError: If transformation fails
        """
        # Check if variable has input branches
        if not self.has_input_branches:
            raise ValueError(f"Variable '{self.name}' has no input branches defined")

        # Validate branches exist
        self.validate_branches_exist(data)

        if self.is_computed:
            # Use transformation
            return self.compute_from_branches(data)

        return data[self.input_branches[0]]

    def validate_branches_exist(self, data: Any) -> None:
        """
        Validate that all required branches exist in the data.

        Args:
            data: Dictionary-like object containing branch data

        Raises:
            ValueError: If required branches are missing
        """
        if not self.has_input_branches:
            return  # Nothing to validate

        # Handle both dict-like objects and pandas DataFrames
        if hasattr(data, "columns"):
            # For pandas DataFrame, check columns
            missing_branches = [b for b in self.input_branches if b not in data.columns]
        else:
            # For dict-like objects
            missing_branches = [b for b in self.input_branches if b not in data]
        if missing_branches:
            raise ValueError(f"Missing branches for variable '{self.name}': {missing_branches}")


def compute_single_var_array(var: Var, data: Any) -> np.ndarray:
    """
    Compute a single variable's data array (convenience function).

    Args:
        var: Variable object
        data: Dictionary-like object containing branch data

    Returns:
        Data array for the variable
    """
    return var.compute_array(data)


def arrays_from_vars(
    variables: list[Var],
    data: Any,
) -> list[np.ndarray]:
    """Extract and transform data arrays from variables."""
    if not variables:
        return []

    # Pre-filter data to only needed columns for performance
    if hasattr(data, "columns"):  # pandas DataFrame
        needed_cols = set()
        for var in variables:
            needed_cols.update(var.input_branches)
        if needed_cols:
            data_filtered = data[list(needed_cols)]
        else:
            data_filtered = data
    else:
        # For dict-like objects, keep as-is
        data_filtered = data

    data_arrays = []
    for var in variables:
        data_array = var.compute_array(data_filtered)
        data_arrays.append(data_array)

    return data_arrays


def get_all_var_input_branches(variables: list[Var]) -> list[str]:
    """Get all unique branch names needed for a list of variables.

    Args:
        variables: List of Var instances

    Returns:
        List of unique branch names
    """
    branches = set()
    for var in variables:
        branches.update(var.input_branches)

    # Filter out None values if they somehow get through
    return [branch for branch in branches if branch is not None]
