import numpy as np
import pandas as pd
import pytest

from hepkit.classification.data_splitting import (
    split_train_test_by_id_bitshift,
)


@pytest.fixture
def sample_data_ids():
    """Fixture for test data with unique run/event/cand IDs."""
    np.random.seed(42)
    n_samples = 1000

    # Create unique IDs: run 900000-900999, event 1000000-1000999, cand 0-99
    run_ids = np.arange(900000, 900999)
    event_ids = np.arange(9000000000, 9000010000)
    cand_ids = np.arange(0, 100)

    # Repeat to make 1000 samples
    run_ids = np.tile(run_ids[:100], 10)
    event_ids = np.tile(event_ids[:100], 10)
    cand_ids = np.tile(cand_ids, 10)

    data_df = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "label": np.random.randint(0, 2, n_samples),
            "runNumber": run_ids,
            "eventNumber": event_ids,
            "candNumber": cand_ids,
        }
    )

    return data_df


def test_split_train_test_by_id_bitshift(sample_data_ids):
    """Test bitshift splitter with large run/event IDs."""
    data_df = sample_data_ids
    test_ratio = 0.3

    train, test, train_ids, test_ids = split_train_test_by_id_bitshift(
        data_df, test_ratio, ["runNumber", "eventNumber", "candNumber"], return_ids=True
    )

    # Check train and test do not contain ID columns
    for col in ["runNumber", "eventNumber", "candNumber"]:
        assert col not in train.columns
        assert col not in test.columns

    # Check proportions (approximate due to hashing)
    assert 0.25 <= len(test) / len(data_df) <= 0.35

    # Check no overlap in IDs
    train_ids_set = set(
        zip(train_ids["runNumber"], train_ids["eventNumber"], train_ids["candNumber"], strict=True)
    )
    test_ids_set = set(
        zip(test_ids["runNumber"], test_ids["eventNumber"], test_ids["candNumber"], strict=True)
    )
    assert train_ids_set.isdisjoint(test_ids_set)

    # Check determinism
    train2, test2 = split_train_test_by_id_bitshift(
        data_df, test_ratio, ["runNumber", "eventNumber", "candNumber"]
    )
    pd.testing.assert_frame_equal(train, train2)
    pd.testing.assert_frame_equal(test, test2)
