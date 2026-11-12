# results/

Everything here is **produced by code** — never edit these files by hand.
Re-run the script instead.

| File | Produced by |
|------|-------------|
| `submission.csv` | **the file to submit** - final DistilBERT, `notebooks/05_distilbert_tuning.ipynb` |
| `submission_baseline.csv` | `python src/run_baseline.py` - logistic regression baseline (not submitted) |
| `submission_distilbert_20split.csv` | superseded DistilBERT run, `notebooks/04_distilbert_20split.ipynb` |
| `cleaned_train.csv`, `cleaned_val.csv` | `notebooks/02_svm_mlp.ipynb` section 2b - raw review, cleaned review and label for the training and validation halves; the exact text SVM and MLP are trained on |
| `part_b/` | `notebooks/02_svm_mlp.ipynb` - SVM/MLP metrics and config |
| `figures/fig01` - `fig05` | `notebooks/01_explore_data.ipynb` |
| `figures/fig06_architecture.png` | `python src/make_architecture_figure.py` |
| `model_comparison.csv` / `.json` | `python src/compare_models.py` |
| `hyperparameter_search.csv` | `python src/tune_hyperparameters.py` |
| `figures/fig07` - `fig08` | `python src/make_comparison_figures.py` |
| `distilbert_metrics.json` | final DistilBERT run, from `results/distilbert_tuning/` |
| `distilbert_tuning/` | `notebooks/05_distilbert_tuning.ipynb` - tuning table and per-epoch history |
| `distilbert_val_predictions.csv`, `distilbert_test_probabilities.csv` | `notebooks/05_distilbert_tuning.ipynb` |
| `svm_val_predictions.csv` | `python src/svm_val_predictions.py` |
| `figures/fig09_distilbert_architecture.png` | `python src/make_distilbert_architecture.py` |

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
| `fig09_distilbert_architecture.png` | DistilBERT layers and output shapes | (b) |

## The model comparison

`model_comparison.csv` is the table for questions (c) and (f); the `.json` beside
it adds confusion matrices, full hyperparameter configs and the resource numbers.
Both come from refitting every TF-IDF model against **one** split and **one**
vocabulary, so the rows differ only by classifier:

| Model | Accuracy | Macro F1 | Negative F1 | Train time | Parameters |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9095 | 0.8334 | 0.7208 | 0.17 s | 20,001 |
| TF-IDF + Linear SVM | 0.9325 | 0.8579 | 0.7549 | 0.10 s | 20,001 |
| TF-IDF + SVD(200) + MLP(128,64) | 0.9095 | 0.7995 | 0.6510 | 7.8 s | 4,034,049 |
| **DistilBERT** (lr 2e-5, seed 42) | **0.9507** | **0.9001** | **0.8290** | 227.8 s (L4 GPU) | 66,955,010 |

CPU timings vary between runs. DistilBERT over seeds 42, 1, 2: 0.8986 ± 0.0089 macro F1.

Two things to carry into the report:

1. **DistilBERT is the best model and is the one submitted.** The Linear SVM is the
   best TF-IDF model and trains in about 0.1 s on a CPU.
2. **The weak spot is the negative class, not accuracy.** Negative F1 runs
   0.65-0.83 against 0.95-0.97 positive, and predicting "positive" for every
   review already scores 0.853 accuracy. Macro F1 is the number to quote.

**Library-version note.** The report quotes the SVM and MLP numbers saved in
`notebooks/02_svm_mlp.ipynb` (SVM 0.9325 / 0.8585, MLP 0.9115 / 0.8078), from the
original run (Python 3.12.7, Anaconda). The table above, and a fresh re-run of that same
notebook with scikit-learn 1.9.0, give SVM 0.8579 and MLP 0.9095 / 0.7995. Code and
settings are identical; the difference comes from the scikit-learn version.

## The submission files

All have the same shape: columns `reviews`, `predictions` (0 = negative,
1 = positive), 1851 rows, in the same order as `data/test.json`.

| File | Model | Validation | Positive rate |
|---|---|---|---|
| **`submission.csv`** | **DistilBERT, lr 2e-5, seed 42 (final)** | 0.9507 acc, 0.9001 macro F1 | 1613 / 1851 (87.1%) |
| `submission_distilbert_20split.csv` | DistilBERT, same settings, earlier T4 run | 0.9446 acc, 0.8893 macro F1 | 1591 / 1851 (86.0%) |
| `submission_baseline.csv` | TF-IDF + Logistic Regression | 0.9095 acc, 0.8334 macro F1 | 1512 / 1851 (81.7%) |

The 15%-split DistilBERT predictions (0.9487 / 0.8951, 1598 positive) are in
`archive/submission_distilbert_15split.csv` and are not used.

**`submission.csv` is the file that goes to NTULearn** - the final DistilBERT
(`notebooks/05_distilbert_tuning.ipynb`, run `baseline`). No script in `src/` writes to
it. Check it with `python src/check_submission.py results/submission.csv`.
