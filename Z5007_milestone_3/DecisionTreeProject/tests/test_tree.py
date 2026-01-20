import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import traceback

from src.tree import DecisionTreeRegressor


def load_small_dataset():
    """Load a small subset of the dataset for testing."""
    df = pd.read_csv("data/dsm_dataset.csv")
    df = df.dropna().drop_duplicates()

    # Preprocessing to match main.py
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df = df.drop_duplicates()

    X = df.drop(columns=["DSM"]).values
    y = df["DSM"].values

    return X[:100], y[:100]   # small subset for fast tests


def test_tree_trains_without_error():
    print("Running test_tree_trains_without_error...", end=" ")
    X, y = load_small_dataset()
    tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
    tree.fit(X, y)
    assert tree.root is not None
    print("PASSED")


def test_prediction_shape():
    print("Running test_prediction_shape...", end=" ")
    X, y = load_small_dataset()
    tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
    tree.fit(X, y)

    preds = tree.predict(X)
    assert len(preds) == len(X)
    print("PASSED")


def test_predictions_no_nan():
    print("Running test_predictions_no_nan...", end=" ")
    X, y = load_small_dataset()
    tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
    tree.fit(X, y)

    preds = tree.predict(X)
    assert not np.isnan(preds).any()
    print("PASSED")


def test_tree_structure_counts():
    print("Running test_tree_structure_counts...", end=" ")
    X, y = load_small_dataset()
    tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
    tree.fit(X, y)

    nodes = tree.count_nodes()
    leaves = tree.count_leaves()

    assert nodes > 0
    assert leaves > 0
    assert leaves <= nodes
    print("PASSED")


if __name__ == "__main__":
    try:
        test_tree_trains_without_error()
        test_prediction_shape()
        test_predictions_no_nan()
        test_tree_structure_counts()
        print("\nAll tests passed successfully!")
    except Exception:
        print("\nFAILED")
        traceback.print_exc()
