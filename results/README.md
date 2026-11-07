# results/

Everything here is **produced by code** — never edit these files by hand.
Re-run the script instead.

| File | Produced by |
|------|-------------|
| `submission.csv` | `python src/run_baseline.py` |
| `submission_distilbert.csv` | `notebooks/02_distilbert.ipynb` (Ryan) |

Both files have the same shape: columns `reviews`, `predictions` (0 = negative,
1 = positive), 1851 rows, in the same order as `data/test.json`.

| File | Model | Validation | Positive rate |
|---|---|---|---|
| `submission.csv` | TF-IDF + Logistic Regression | 0.9095 acc | 1512 / 1851 (81.7%) |
| `submission_distilbert.csv` | DistilBERT, 4 epochs, `max_length=256` | 0.9487 acc, 0.8951 macro F1 | 1598 / 1851 (86.3%) |

**Which one goes to NTULearn is still open.** DistilBERT is the stronger model on the
validation split, but only one `submission.csv` can be submitted - agree as a group,
then copy the chosen file over `submission.csv` in the final commit.
