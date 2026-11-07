# results/

Everything here is **produced by code** — never edit these files by hand.
Re-run the script instead.

| File | Produced by |
|------|-------------|
| `submission.csv` | `python src/run_baseline.py` |
| `submission_distilbert.csv` | `notebooks/03_distilbert.ipynb` (Ryan) |
| `figures/fig01` - `fig05` | `notebooks/01_explore_data.ipynb` |
| `figures/fig06_architecture.png` | `python src/make_architecture_figure.py` |
| `model_comparison.csv` / `.json` | `python src/compare_models.py` |
| `figures/fig07` - `fig08` | `python src/make_comparison_figures.py` |
| `distilbert_metrics.json` | `notebooks/04_distilbert_20split.ipynb` (not run yet) |

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
| `fig07_confusion_matrices.png` | where each model's errors fall, normalised per true class | (e), (f) |
| `fig08_model_comparison.png` | per-class F1 and macro F1 against the majority-class floor | (c), (f) |

## The model comparison

`model_comparison.csv` is the table for questions (c) and (f); the `.json` beside
it adds confusion matrices, full hyperparameter configs and the resource numbers.
Both come from refitting every TF-IDF model against **one** split and **one**
vocabulary, so the rows differ only by classifier:

| Model | Accuracy | Macro F1 | Negative F1 | Train time | Parameters |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9095 | 0.8334 | 0.7208 | 0.05 s | 20,001 |
| **TF-IDF + Linear SVM** | **0.9325** | **0.8579** | **0.7549** | 0.03 s | 20,001 |
| TF-IDF + SVD(200) + MLP(128,64) | 0.9095 | 0.7995 | 0.6510 | 7.7 s | 4,034,049 |
| DistilBERT *(15% split)* | 0.9487 | 0.8951 | not recorded | GPU | ~66M |

Two things to carry into the report:

1. **Linear SVM is the best strictly comparable model**, and it trains in 0.03 s.
   DistilBERT scores higher but on a different split - run
   `notebooks/04_distilbert_20split.ipynb` to make that row honest.
2. **The weak spot is the negative class, not accuracy.** Negative F1 runs
   0.65-0.75 against 0.95-0.96 positive, and predicting "positive" for every
   review already scores 0.853 accuracy. Macro F1 is the number to quote.

The MLP figures here differ slightly from the ones printed in
`notebooks/02_svm_mlp.ipynb` (0.7995 vs 0.8078 macro F1). Adam plus early
stopping is sensitive to the scikit-learn version; the numbers in this table are
the ones to cite, because they are the ones produced on the shared split.

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
