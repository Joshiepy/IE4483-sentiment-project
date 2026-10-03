"""Step 1 of the project brief: reading in the data.

The course gives us two files in data/:
  train.json -> list of {"reviews": "...", "sentiments": 0 or 1}
  test.json  -> list of {"reviews": "..."}   (no labels - we predict these)

Sentiment labels: 0 = negative, 1 = positive.
"""

import json
from pathlib import Path

import pandas as pd

# Folder layout: this file is in src/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def load_train():
    """Return a DataFrame with columns: reviews, sentiments."""
    return _read_json(DATA_DIR / "train.json")


def load_test():
    """Return a DataFrame with one column: reviews."""
    return _read_json(DATA_DIR / "test.json")


if __name__ == "__main__":
    # Quick sanity check: python src/load_data.py
    train = load_train()
    test = load_test()
    print(f"train: {len(train)} rows, columns = {list(train.columns)}")
    print(f"test : {len(test)} rows, columns = {list(test.columns)}")
    print("\nlabel balance (0 = negative, 1 = positive):")
    print(train["sentiments"].value_counts())
    print("\nfirst training review:")
    print(train.iloc[0]["reviews"])
