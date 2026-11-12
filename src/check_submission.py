"""Check that a submission CSV has the format the course asks for.

Run it with:  python src/check_submission.py results/submission.csv

Prints PASS or FAIL for each check and exits with code 1 if any check fails.
"""

import sys
from pathlib import Path

import pandas as pd

from load_data import DATA_DIR

EXPECTED_ROWS = 1851


def main():
    if len(sys.argv) != 2:
        print("usage: python src/check_submission.py <path to csv>")
        sys.exit(2)
    path = Path(sys.argv[1])
    sub = pd.read_csv(path)
    test = pd.read_json(DATA_DIR / "test.json")
    example = pd.read_csv(DATA_DIR / "submission_example.csv")

    preds = sub["predictions"] if "predictions" in sub.columns else pd.Series(dtype=object)
    checks = [
        ("exactly two columns, named reviews and predictions (in that order)",
         list(sub.columns) == ["reviews", "predictions"]),
        (f"{EXPECTED_ROWS} rows", len(sub) == EXPECTED_ROWS),
        ("reviews identical to data/test.json, row by row, same order",
         len(sub) == len(test)
         and "reviews" in sub.columns
         and (sub["reviews"].astype(str).values == test["reviews"].astype(str).values).all()),
        ("every prediction is the integer 0 or 1, no blanks",
         len(preds) > 0
         and preds.notna().all()
         and pd.api.types.is_integer_dtype(preds)
         and set(preds.unique()) <= {0, 1}),
        ("same columns, row count and review order as data/submission_example.csv",
         list(sub.columns) == list(example.columns)
         and len(sub) == len(example)
         and (sub["reviews"].astype(str).values == example["reviews"].astype(str).values).all()),
    ]

    print(f"Checking {path}")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if len(preds):
        n_pos, n_neg = int((preds == 1).sum()), int((preds == 0).sum())
        print(f"  predictions: {n_pos} positive (1), {n_neg} negative (0), "
              f"{100 * n_pos / len(preds):.1f}% positive")

    failed = sum(not ok for _, ok in checks)
    print("ALL CHECKS PASSED" if not failed else f"{failed} CHECK(S) FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
