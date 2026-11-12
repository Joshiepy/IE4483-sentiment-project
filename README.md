# IE4483 Mini Project — Sentiment of Product Reviews (Option 1)

Classify product reviews as **positive (1)** or **negative (0)**.

- **Course:** IE4483 Artificial Intelligence and Data Mining
- **Deadline:** Friday 13 Nov 2026, 11:59pm (submit to NTULearn)
- **Deliverables:** one group report (PDF) + `submission.csv`
- **Group:** 3 students — the report must list every member and who did what

---

## Folder structure

```
IE4483-project/
├── data/          Data given by the course (do not edit these)
├── notebooks/     01 explore, 02 SVM + MLP, 04/05 DistilBERT (Colab GPU)
├── src/           Python scripts: data loading, baseline, checks, figures
├── results/       Everything the code produces - submission.csv is the final answer
├── archive/       Superseded early runs, kept for the record (not used in the report)
├── report/        The project brief + report outline
├── .devcontainer/ Config so GitHub Codespaces sets itself up automatically
└── requirements.txt
```

One rule to keep it simple: **`data/` is read-only, `results/` is write-only.**
Nothing in `results/` is hand-edited — if it's wrong, fix the code and re-run.

---

## For the marker: how to reproduce everything

**The submitted file is `results/submission.csv`** - final DistilBERT predictions,
1,851 rows, 1,613 positive / 238 negative.

### 1. Setup (CPU parts, any machine)

```bash
# from the repo root
python -m venv .venv
.venv\Scripts\activate         # Windows (PowerShell / cmd)
# source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
```

### 2. Run, in this order

| Step | Command / notebook | What it does | Output |
|---|---|---|---|
| 1 | `python src/load_data.py` | Data sanity check: sizes, label balance, a sample | printed |
| 2 | `python src/run_baseline.py` | TF-IDF + Logistic Regression baseline | `results/submission_baseline.csv` |
| 3 | `notebooks/02_svm_mlp.ipynb` (Jupyter, run top to bottom) | **Shows the cleaned data**, then trains Linear SVM and MLP on it | `results/cleaned_train.csv`, `results/cleaned_val.csv`, `results/part_b/` |
| 4 | `notebooks/05_distilbert_tuning.ipynb` on **Google Colab with a GPU** | Final DistilBERT: tuning runs + the submitted predictions | `submission.csv` + tuning results (downloaded as a zip) |
| 5 | `python src/check_submission.py results/submission.csv` | Checks the submission format against `data/test.json` and `data/submission_example.csv` | PASS/FAIL per check |
| 6 | `python src/error_analysis.py` | Right/wrong examples used in report Section 6 | printed |

**Running the DistilBERT notebook on Colab:** open
[colab.research.google.com](https://colab.research.google.com) -> `File` -> `Upload notebook` ->
`notebooks/05_distilbert_tuning.ipynb`; `Runtime` -> `Change runtime type` -> **GPU**; upload
`data/train.json` and `data/test.json` to the Colab Files panel; run all cells. The final model
is the run named `baseline` (lr 2e-5, seed 42); step 5 of the notebook writes its `submission.csv`.
About 4 minutes per run on an L4; on CPU it would take hours.

### 3. Which file produces which result

All models use the same stratified 80/20 split of `train.json` (`random_state=42`):
5,920 training and 1,481 validation reviews (217 negative, 1,264 positive).
TF-IDF settings (all three classical models): `lowercase=True, stop_words="english",
ngram_range=(1, 2), min_df=2, max_features=20000`, fitted on the training split only.

| Model | Produced by | Val. accuracy | Macro F1 | Confusion matrix [[TN, FP], [FN, TP]] |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | `src/run_baseline.py` | 0.9095 | 0.8334 | [[173, 44], [90, 1174]] |
| TF-IDF + Linear SVM (C=1) | `notebooks/02_svm_mlp.ipynb` | 0.9325 | 0.8585* | [[154, 63], [37, 1227]] |
| TF-IDF + SVD(200) + MLP(128, 64) | `notebooks/02_svm_mlp.ipynb` | 0.9115* | 0.8078* | [[131, 86], [45, 1219]] |
| **DistilBERT (final, submitted)** | `notebooks/05_distilbert_tuning.ipynb` | **0.9507** | **0.9001** | [[177, 40], [33, 1231]] |

DistilBERT: `distilbert-base-uncased`, `max_length=256`, batch 16, 4 epochs, AdamW lr 2e-5,
weight decay 0.01, 10% linear warm-up then linear decay, gradient clipping 1.0, best
checkpoint by validation macro F1 (epoch 2). NVIDIA L4, 227.8 s. Results in
`results/distilbert_tuning/tuning_results.csv` (row `baseline`).

\* **Library-version note.** The SVM and MLP numbers marked * are the outputs saved in
`notebooks/02_svm_mlp.ipynb` from the run quoted in the report (Python 3.12.7, Anaconda).
Re-running the same notebook with a newer scikit-learn (tested: 1.9.0, Python 3.13) gives
SVM 0.9325 / 0.8579 and MLP 0.9095 / 0.7995, the same as `src/compare_models.py`. The code
and settings are identical; the small difference comes from the library version. The
logistic regression and DistilBERT numbers are not affected.

### 4. Reproducibility

**Every random seed is 42**: the train/validation split, every scikit-learn model
(`random_state=42`), and for DistilBERT Python, NumPy and PyTorch seeds plus deterministic
cuDNN. GPU training can still vary slightly between GPU types (a T4 run of the same notebook
scored 0.8893 macro F1, see `notebooks/04_distilbert_20split.ipynb`), which is why the report
also gives a three-seed mean.

### Other scripts (figures and tables in the report)

```bash
python src/compare_models.py             # all models on one split -> results/model_comparison.*
python src/tune_hyperparameters.py       # LogReg + SVM one-at-a-time sweeps -> results/hyperparameter_search.csv
python src/make_comparison_figures.py    # fig07 confusion matrices, fig08 comparison
python src/make_architecture_figure.py   # fig06
python src/make_distilbert_architecture.py  # fig09
python src/svm_val_predictions.py        # SVM per-review output for the error analysis
```

---

## The dataset

| File | Rows | Columns |
|------|------|---------|
| `data/train.json` | 7,401 | `reviews` (text), `sentiments` (0 or 1) |
| `data/test.json`  | 1,851 | `reviews` (text) — **no labels, we predict these** |

**The labels are imbalanced:** 6,319 positive vs 1,082 negative (~85% positive).
That matters a lot:

- A model that blindly guesses "positive" every time already scores ~85% accuracy,
  so **accuracy alone is a misleading metric** — report precision/recall/F1 too.
- The baseline uses `class_weight="balanced"` to compensate. Mention this in the report.

`data/submission_example.csv` is the course's example file. It shows the required
format: two columns, `reviews` and `predictions`. Our output matches it.

---

## Where to run this

| | GPU? | Good for |
|---|---|---|
| **Your laptop** | no | everything except DistilBERT (all CPU parts take seconds) |
| **GitHub Codespaces** (`Code` -> `Codespaces`) | no | same as laptop, nothing to install |
| **Google Colab** | **yes** | DistilBERT: 30-60 min per epoch on CPU vs about 1 min on a GPU |

Colab deletes everything when the session closes, so download the results zip the
notebook produces before closing the tab.

---

## Working as a group of three

```bash
git pull                       # always do this BEFORE you start working
# ... make your changes ...
git add -A
git commit -m "what you did"
git push
```

- **Split by file, not by line.** Each notebook has one owner (see `notebooks/README.md`);
  two people editing the same notebook produces unreadable JSON merge conflicts.
- **Never commit a trained model.** `.gitignore` blocks `*.pt`, `*.pth`, `*.bin`,
  `*.joblib` and `models/`.
