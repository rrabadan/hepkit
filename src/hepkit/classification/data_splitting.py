import hashlib

import numpy as np
import pandas as pd


def _validate_split_inputs(
    data: pd.DataFrame, test_ratio: float, id_columns: list[str] | str
) -> list[str]:
    """
    Common validation logic for train/test splitting functions.

    Returns:
        List of validated id_columns
    """
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError("test_ratio must be between 0.0 and 1.0")

    if isinstance(id_columns, str):
        id_columns = [id_columns]

    missing_cols = [col for col in id_columns if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    return id_columns


def _compute_test_mask(hashes: np.ndarray, test_ratio: float) -> np.ndarray:
    """
    Compute boolean mask for test set based on hash values.

    Returns:
        Boolean array indicating which rows belong to test set
    """
    return hashes < test_ratio * 2**64


def _apply_split(data: pd.DataFrame, test_mask: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the train/test split based on boolean mask.

    Returns:
        Tuple of (train_set, test_set)
    """
    train_set = data.loc[~test_mask].copy().reset_index(drop=True)
    test_set = data.loc[test_mask].copy().reset_index(drop=True)
    return train_set, test_set


def split_train_test_by_id_hash(
    data: pd.DataFrame,
    test_ratio: float,
    id_columns: list[str] | str,
    separator: str = "_",
    return_ids: bool = False,
) -> (
    tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """Split using optimized hash functions.

    Parameters:
    - data: DataFrame with features and ID columns
    - test_ratio: Fraction for test set
    - id_columns: ID column names
    - separator: Separator for multi-column IDs
    - return_ids: If True, return train/test IDs as well

    Returns:
    - (train, test) or (train, test, train_ids, test_ids)
    """
    id_columns = _validate_split_inputs(data, test_ratio, id_columns)

    # Separate IDs and features
    id_df = data[id_columns]
    feature_df = data.drop(columns=id_columns)

    # For single numeric column, use direct numeric hash
    if len(id_columns) == 1 and pd.api.types.is_numeric_dtype(id_df[id_columns[0]]):
        values = id_df[id_columns[0]].astype(np.uint64).values
        hashes = values * np.uint64(2654435761)  # Knuth's multiplicative hash
    else:
        # String-based approach
        if len(id_columns) == 1:
            identifiers = id_df[id_columns[0]].astype(str)
        else:
            identifiers = id_df[id_columns].apply(
                lambda row: separator.join(row.astype(str)), axis=1
            )

        def fast_hash(s):
            return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

        hashes = identifiers.apply(fast_hash)

    test_mask = _compute_test_mask(hashes, test_ratio)
    train_features, test_features = _apply_split(feature_df, test_mask)
    if return_ids:
        train_ids, test_ids = _apply_split(id_df, test_mask)
        return train_features, test_features, train_ids, test_ids
    else:
        return train_features, test_features


def split_train_test_by_id_bitshift(
    data: pd.DataFrame,
    test_ratio: float,
    id_columns: list[str] | str = ["runNumber", "eventNumber", "candNumber"],  # noqa: B006
    return_ids: bool = False,
) -> (
    tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """Split dataset into train/test sets using bit-shifting hash of numeric ID combinations.
    This function creates a deterministic train/test split by combining multiple numeric
    ID columns into a single 64-bit integer using bit-shifting operations, then applying
    a hash function for randomization. The bit-shifting approach is efficient for up to
    3 ID columns with specific range constraints.
    The bit-shifting layout for different column counts:
    - 1 column: Direct use of the column value
    - 2 columns: [run_col (20 bits) | event_col (44 bits)]
    - 3 columns: [run_col (20 bits) | event_col (34 bits) | cand_col (10 bits)]
    For more than 3 columns, automatically falls back to prime multiplication method.
    Parameters
    ----------
    data : pd.DataFrame
        The dataset to be split, containing both features and ID columns.
    test_ratio : float
        Fraction of data to allocate to test set. Must be between 0 and 1.
    id_columns : list[str] | str, default ["runNumber", "eventNumber", "candNumber"]
        Column name(s) in `data` to use for creating unique identifiers.
        Can be a single string or list of strings.
    return_ids : bool, default False
        If True, also return the train and test ID DataFrames.
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame] or Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        If return_ids is False: (train_data, test_data) where both are subsets of the
        original `data` DataFrame with ID columns removed.
        If return_ids is True: (train_data, test_data, train_ids, test_ids) where
        train_ids and test_ids are the corresponding ID subsets.
    Raises
    ------
    ValueError
        If ID column values exceed the bit range limits:
        - 1st column (run): >= 2^20 (1,048,576)
        - 2nd column (event): >= 2^44 for 2-col mode or >= 2^34 for 3-col mode
        - 3rd column (candidate): >= 2^10 (1,024)
    Notes
    -----
    - The split is deterministic: same inputs always produce the same split
    - Uses Knuth's multiplicative hash with constant 2654435761 for randomization
    - Bit-shifting provides better performance than string concatenation methods
    - Range validation ensures no bit overlap in the combined ID
    - For datasets with ID values exceeding range limits, consider using
      `split_train_test_by_id_prime_mult` instead
    Examples
    --------
    >>> data = pd.DataFrame({'run': [1, 1, 2, 2], 'event': [100, 101, 100, 101], 'feature1': [1, 2, 3, 4], 'target': [0, 1, 0, 1]})
    >>> train, test = split_train_test_by_id_bitshift(data, 0.5, ['run', 'event'])
    >>> train, test, train_ids, test_ids = split_train_test_by_id_bitshift(data, 0.5, ['run', 'event'], return_ids=True)
    """
    id_columns = _validate_split_inputs(data, test_ratio, id_columns)

    # Separate IDs and features
    id_df = data[id_columns]
    feature_df = data.drop(columns=id_columns)

    if len(id_columns) == 1:
        unique_ids = id_df[id_columns[0]].astype(np.uint64)

    elif len(id_columns) == 2:
        run_col, event_col = id_columns[0], id_columns[1]

        # Validate ranges
        max_run = id_df[run_col].max()
        max_event = id_df[event_col].max()

        if max_run >= 2**20:
            raise ValueError(
                f"{id_columns[0]} number too large for 2-column mode: {max_run} >= 2^20"
            )
        if max_event >= 2**44:
            raise ValueError(
                f"{id_columns[1]} number too large for 2-column mode: {max_event} >= 2^44"
            )

        unique_ids = (id_df[run_col].astype(np.uint64).values << 44) | id_df[event_col].astype(
            np.uint64
        ).values

    elif len(id_columns) == 3:
        run_col, event_col, cand_col = id_columns[0], id_columns[1], id_columns[2]

        # Validate ranges
        max_run = id_df[run_col].max()
        max_event = id_df[event_col].max()
        max_cand = id_df[cand_col].max()

        if max_run >= 2**20:
            raise ValueError(
                f"{id_columns[0]} number too large for 3-column mode: {max_run} >= 2^20"
            )
        if max_event >= 2**34:
            raise ValueError(
                f"{id_columns[1]} number too large for 3-column mode: {max_event} >= 2^34"
            )
        if max_cand >= 2**10:
            raise ValueError(
                f"{id_columns[2]} number too large for 3-column mode: {max_cand} >= 2^10"
            )

        unique_ids = (
            (id_df[run_col].astype(np.uint64).values << 44)
            | (id_df[event_col].astype(np.uint64).values << 10)
            | id_df[cand_col].astype(np.uint64).values
        )

    else:
        # Fall back to prime method for >3 columns
        return split_train_test_by_id_prime_mult(data, test_ratio, id_columns, return_ids)

    # Apply hash and split
    hashes = unique_ids * np.uint64(2654435761)
    test_mask = _compute_test_mask(hashes, test_ratio)
    train_features, test_features = _apply_split(feature_df, test_mask)
    if return_ids:
        train_ids, test_ids = _apply_split(id_df, test_mask)
        return train_features, test_features, train_ids, test_ids
    else:
        return train_features, test_features


def split_train_test_by_id_prime_mult(
    data: pd.DataFrame, test_ratio: float, id_columns: list[str] | str, return_ids: bool = False
) -> (
    tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """Split dataset into train and test sets using prime multiplication for ID combinations.
    This function creates deterministic train/test splits by combining multiple ID columns
    using prime number multiplication to create unique composite IDs, then hashing these
    IDs to determine the split. This ensures that all rows with the same combination of
    ID values are kept together in either train or test set.
    Args:
        data (pd.DataFrame): The dataset to split, containing both features and ID columns.
        test_ratio (float): Fraction of unique ID combinations to assign to test set.
            Must be between 0 and 1.
        id_columns (Union[List[str], str]): Column name(s) in data to use for
            creating composite IDs. Can be a single column name or list of column names.
            Maximum of 5 columns supported.
        return_ids (bool): If True, also return train and test ID DataFrames.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame] or Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        If return_ids is False: (train_data, test_data) with ID columns removed.
        If return_ids is True: (train_data, test_data, train_ids, test_ids).
    Raises:
        ValueError: If more than 5 ID columns are provided (exceeds available primes).
    Note:
        The function uses predetermined large prime numbers [982451653, 982451629,
        982451581, 982451567, 982451563] to create unique composite IDs. The split
        is deterministic - the same input will always produce the same train/test split.
    Example:
        >>> data = pd.DataFrame({'run': [1, 1, 2, 2], 'event': [100, 101, 100, 101], 'feature1': [1, 2, 3, 4], 'target': [0, 1, 0, 1]})
        >>> train, test = split_train_test_by_id_prime_mult(data, 0.2, ['event', 'run'])
        >>> train, test, train_ids, test_ids = split_train_test_by_id_prime_mult(data, 0.2, ['event', 'run'], return_ids=True)
    """
    id_columns = _validate_split_inputs(data, test_ratio, id_columns)

    # Separate IDs and features
    id_df = data[id_columns]
    feature_df = data.drop(columns=id_columns)

    primes = [982451653, 982451629, 982451581, 982451567, 982451563]

    if len(id_columns) > len(primes):
        raise ValueError(
            f"Too many ID columns ({len(id_columns)}), maximum supported is {len(primes)}"
        )

    # Create unique IDs using prime multiplication
    unique_ids = np.zeros(len(id_df), dtype=np.uint64)
    for i, col in enumerate(id_columns):
        unique_ids += id_df[col].astype(np.uint64).values * primes[i]

    # Hash and split
    hashes = unique_ids * np.uint64(2654435761)
    test_mask = _compute_test_mask(hashes, test_ratio)
    train_features, test_features = _apply_split(feature_df, test_mask)
    if return_ids:
        train_ids, test_ids = _apply_split(id_df, test_mask)
        return train_features, test_features, train_ids, test_ids
    else:
        return train_features, test_features


# Simplified convenience functions
def split_train_test_by_run_event(
    data: pd.DataFrame,
    test_ratio: float,
    run_col: str = "runNumber",
    event_col: str = "eventNumber",
    return_ids: bool = False,
) -> (
    tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """Convenience function for runNumber + eventNumber splitting."""
    return split_train_test_by_id_bitshift(data, test_ratio, [run_col, event_col], return_ids)


def split_train_test_by_run_event_cand(
    data: pd.DataFrame,
    test_ratio: float,
    run_col: str = "runNumber",
    event_col: str = "eventNumber",
    cand_col: str = "candNumber",
    return_ids: bool = False,
) -> (
    tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """Convenience function for runNumber + eventNumber + candNumber splitting."""
    return split_train_test_by_id_bitshift(
        data, test_ratio, [run_col, event_col, cand_col], return_ids
    )


def split_train_test_by_unique_id(
    data: pd.DataFrame,
    test_ratio: float,
    id_columns: list[str] | str = ["runNumber", "eventNumber", "candNumber"],  # noqa: B006
    method: str = "auto",
    return_ids: bool = False,
) -> (
    tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """
    Unified interface for train/test splitting with automatic method selection.

    Parameters:
    - data: DataFrame containing the dataset to split (features and ID columns)
    - test_ratio: Float between 0.0 and 1.0, proportion of data for test set
    - id_columns: List of column names or single column name to use for ID-based splitting
    - method: "auto" (default) - automatically chooses best method
             "bitshift" - force bit-shifting approach
             "hash" - force hash-based approach
             "prime" - force prime multiplication approach
    - return_ids: If True, also return train and test ID DataFrames

    Returns:
    - (train_set, test_set) or (train_set, test_set, train_ids, test_ids) with ID columns removed from features
    """
    if method == "auto":
        # Try bitshift first (fastest), fall back if it fails
        try:
            if len(id_columns) <= 3 and all(
                pd.api.types.is_numeric_dtype(data[col]) for col in id_columns
            ):
                return split_train_test_by_id_bitshift(data, test_ratio, id_columns, return_ids)
        except (ValueError, OverflowError):
            pass

        # Fall back to hash method
        return split_train_test_by_id_hash(data, test_ratio, id_columns, return_ids=return_ids)

    elif method == "bitshift":
        return split_train_test_by_id_bitshift(data, test_ratio, id_columns, return_ids)
    elif method == "hash":
        return split_train_test_by_id_hash(data, test_ratio, id_columns, return_ids=return_ids)
    elif method == "prime":
        return split_train_test_by_id_prime_mult(data, test_ratio, id_columns, return_ids)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: auto, bitshift, hash, prime")
