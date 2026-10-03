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
├── data/        Data given by the course (do not edit these)
├── notebooks/   Jupyter notebooks for exploring and experimenting
├── src/         Python code that actually runs the pipeline
├── results/     Things the code produces (submission.csv, plots)
├── report/      The project brief + the report we write
└── requirements.txt
```

One rule to keep it simple: **`data/` is read-only, `results/` is write-only.**
Nothing in `results/` is hand-edited — if it's wrong, fix the code and re-run.

---

## Setup

```bash
# from the project folder
python -m venv .venv
.venv\Scripts\activate         # Windows (PowerShell / cmd)
# source .venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
```

## How to run

```bash
python src/load_data.py       # sanity check: prints dataset sizes and a sample
python src/run_baseline.py    # trains the baseline, writes results/submission.csv
python src/error_analysis.py  # prints right/wrong examples for report question (e)
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

## Current baseline result

TF-IDF (unigrams + bigrams, 20k features) + Logistic Regression:

| Metric | Score |
|--------|-------|
| Validation accuracy | **0.9095** |
| Negative F1 | 0.72 |
| Positive F1 | 0.95 |

The brief *encourages a neural model* (RNN or a pretrained model like BERT), so this
baseline is the number to beat — and a useful comparison point for report question (f).

---

## Suggested plan

1. ~~Get data loading + a working baseline~~ ✅ done
2. Explore the data in a notebook (review lengths, common words, class balance)
3. Build the main model — RNN+linear layer, or fine-tune a small BERT
4. Tune hyperparameters (learning rate, feature dim) — **write down what you tried**,
   question (c) asks you to justify these choices
5. Run `error_analysis.py` and pick 1–2 cases for question (e)
6. Write the report using `report/report-outline.md`
7. Submit the report + `results/submission.csv`

## Reproducibility

Every script fixes `RANDOM_SEED = 42`. The brief explicitly asks for this in
question (b) — keep it that way, or average over multiple runs instead.
