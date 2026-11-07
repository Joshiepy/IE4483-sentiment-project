# results/

Everything here is **produced by code** — never edit these files by hand.
Re-run the script instead.

| File | Produced by |
|------|-------------|
| `submission.csv` | `python src/run_baseline.py` |
| `submission_distilbert.csv` | `notebooks/03_distilbert.ipynb` (Ryan) |
| `figures/fig01` - `fig05` | `notebooks/01_explore_data.ipynb` |
| `figures/fig06_architecture.png` | `python src/make_architecture_figure.py` |

## Figures

`figures/` holds the 200 dpi PNGs for the report:

| Figure | Shows | Report question |
|---|---|---|
| `fig01_class_balance.png` | 85/15 positive-negative split | (a), (d) |
| `fig02_review_length.png` | length distribution + the `max_length=256` choice | (a), (b) |
| `fig03_length_by_class.png` | length is not a label leak | (a) |
| `fig04_top_words.png` | frequent words barely differ between classes | (a) |
| `fig05_distinctive_words.png` | log-odds ratio - the terms that do discriminate | (a), (e) |
| `fig06_architecture.png` | the TF-IDF + LogReg pipeline, dimensions at each stage | (b) |

## The two submission files

Both have the same shape: columns `reviews`, `predictions` (0 = negative,
1 = positive), 1851 rows, in the same order as `data/test.json`.

| File | Model | Validation | Positive rate |
|---|---|---|---|
| `submission.csv` | TF-IDF + Logistic Regression | 0.9095 acc, 0.83 macro F1 | 1512 / 1851 (81.7%) |
| `submission_distilbert.csv` | DistilBERT, 4 epochs, `max_length=256` | 0.9487 acc, 0.8951 macro F1 | 1598 / 1851 (86.3%) |

**Which one goes to NTULearn is still open.** DistilBERT is the stronger model on the
validation split, but only one `submission.csv` can be submitted - agree as a group,
then copy the chosen file over `submission.csv` in the final commit.

One caveat before using that comparison in the report: DistilBERT was validated on a
**15%** split and everything else on **20%**, so the two columns are not strictly
comparable. See `notebooks/README.md`.
