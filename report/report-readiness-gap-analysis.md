# Report readiness & gap analysis — IE4483 Mini Project (Option 1)

**Date of review:** 7 Nov 2026
**Deadline:** Friday 13 Nov 2026, 11:59pm (NTULearn)
**Scope of this review:** committed code and saved notebook outputs in this repo only.

> **Caveat.** There is no report draft in this repo — `report/` contains only the brief
> PDF and `report-outline.md`. Everything below is judged from code and from outputs
> actually saved inside the notebooks. If a group member has results in a document
> outside the repo, those are not reflected here.

---

## 1. Headline assessment

| Model | Status | Report-ready? |
|---|---|---|
| **TF-IDF + Logistic Regression** (`src/run_baseline.py`) | Run; produced the only `submission.csv` in the repo | Baseline only — it is the number to beat, not the answer |
| **Linear SVM + MLP** (`SVM and MLP.ipynb`) | Run, with outputs saved | **~2/3 ready.** Strong on (a), (b); nothing for (c), (d), (e) |
| **DistilBERT** (`notebooks/colab_bert_training.ipynb`) | **Never executed** — every cell has empty outputs | **Contributes nothing yet** |

Two experiments are missing for *every* model:

1. **No parameter sweep anywhere in the repo.** A grep for `GridSearchCV`, `cross_val`,
   `KFold`, and loops over `C` / `lr` / `alpha` returns zero hits across all `.py` and
   `.ipynb` files. Every model is a single hardcoded configuration.
2. **No error analysis on either of the two stronger models.** `src/error_analysis.py`
   works, but it analyses the Logistic Regression baseline, and its output was never
   saved into the repo.

---

## 2. Readiness by report question

| Q | Marks | SVM / MLP | DistilBERT |
|---|---|---|---|
| **(a)** Feature format & preprocessing | 10% | Ready — TF-IDF settings all explicit, vectoriser fit on train split only | Described in markdown (tokeniser, `max_length=256`, rationale for keeping stop words) but never run |
| **(b)** Model description + architecture figure | 20% | Partial — config, layer dims, timings, MLP loss curve, and a drafted (b.1) write-up exist. **No architecture figure.** | Partial — markdown covers 6 layers / 66M params / dims / loss / optimiser / seed. **No figure, no run.** |
| **(c)** Parameter choice & justification | 10% | **Missing.** Single `C=1.0`, single `(128,64)` MLP. No "value tried → score" table. | **Missing.** `2e-5` is justified by citing the BERT paper, but no measured comparison. |
| **(d)** Test-set predictions | 20% | **Missing.** The notebook loads `test_df` in cell 1, prints its shape, and never predicts it. | **Missing.** Code to write `submission.csv` exists but was never run. |
| **(e)** Error analysis | 10% | **Missing.** No sample-level inspection at all. | **Missing.** Code exists (cell 16, most-confident mistakes), never run. |
| **(f)** Effect of feature format | 10% | **Cannot be filled honestly yet** — see gap 4. | Same. |
| **(g)** Hotel reviews, no rating scores | 10% | Discussion only — no experiments needed. Outline already lists the angles. | — |
| **(h)** Noisy rating scores, three approaches | 10% | Discussion only — outline already names three. | — |

**Questions at risk, by weight:** (c) 10% + (d) 20% + (e) 10% + (f) 10% = **50% of the
report currently has no supporting evidence in the repo.**

---

## 3. The five structural gaps

### Gap 1 — DistilBERT has never been run

The brief states: *"You are encouraged to use neural networks for this task... for
example... pretrained language model such as BERT."*

The current best model is **Linear SVM (0.9325 validation accuracy)**, which is not a
neural network. The MLP is a neural network but performs worse (0.9115). As things stand
the group would submit a non-neural model as its headline result against a brief that
points explicitly at BERT.

**Fix:** run `notebooks/colab_bert_training.ipynb` end to end on a Colab T4 (~2 min/epoch).
A single successful run closes (b), (d), and (e) for the strongest model, and supplies
the second row that (f) needs.

### Gap 2 — No parameter sweep, which maps directly onto question (c)

Step 4 of the brief: *"You will need to try different model parameters to obtain good
classification results, e.g., feature dimension, weights, initialization, and learning
rate."*

Minimum viable sweep:

| Model | Sweep | Why it answers (c) |
|---|---|---|
| Linear SVM | `C` over `[0.01, 0.1, 1, 10]` | Regularisation strength — the classic SVM knob |
| TF-IDF | `max_features` over `[5000, 20000, 50000]` | This is the "feature dimension" the brief names |
| MLP | hidden sizes `(64,)` vs `(128,64)` vs `(256,128)` | Model capacity |
| DistilBERT | learning rate `[1e-5, 2e-5, 5e-5]` × 2–3 epochs | The brief names learning rate explicitly |

Record **every value tried and its validation macro F1** in one table. That table is
answer (c). Report macro F1, not accuracy — the data is ~85% positive.

### Gap 3 — No error analysis on the models that would actually be submitted

`src/error_analysis.py` prints confident mistakes, borderline mistakes, confident correct
cases, and the strongest positive/negative n-grams — but only for Logistic Regression,
and nothing it prints is saved into the repo.

**Fix:** run the same analysis against the SVM, and (after Gap 1) against DistilBERT's
saved validation predictions. Then pick 1–2 concrete cases, quote the review text, and
explain the likely cause (negation, sarcasm, mixed sentiment, rare vocabulary).

### Gap 4 — Question (f) has only one feature format measured

Logistic Regression, SVM, and MLP are all built on TF-IDF. That is three *classifiers*
on one *feature format* — not a feature-format comparison. The outline is explicit that
the accuracy column must be filled with *"your own measured numbers, not guesses."*

| Feature format | Resource cost | Measured accuracy | Status |
|---|---|---|---|
| TF-IDF (bag of words) | Very light, CPU only, ~0.1 s to train the SVM | 0.9325 (SVM) | **Have it** |
| Pretrained word embeddings (GloVe / Word2Vec) | Moderate; needs pretrained vectors | — | **Not measured** |
| Pretrained LM (DistilBERT) | Heavy; GPU strongly preferred | — | **Not measured** |

Running DistilBERT fills the third row. An averaged-GloVe-vector + Logistic Regression
run would fill the second cheaply and make (f) a genuine three-way comparison.

### Gap 5 — No architecture figure for either model

The outline flags this as explicitly required, and it sits inside the 20% band of
question (b). Nothing in the repo produces a figure. Needed: one block diagram per
submitted model showing input → layers → output with dimensions annotated.

### Housekeeping — uncommitted results

`SVM and MLP.ipynb` writes `results/part_b/` (`svm_mlp_validation_metrics.csv`,
`model_config.json`, `validation_models.joblib`), but that folder **is not committed**.
Only `results/README.md` and `results/submission.csv` are tracked. Those metrics exist
only on one laptop.

---

## 4. One trap to avoid

Question (e) asks for *"correctly and incorrectly classified samples **in the test set**."*
But `test.json` has no labels, so correctness on the test set is unknowable.

The right move — and what the outline already assumes — is to do the error analysis on the
**held-out validation split**, and to **say so explicitly in the report**: *"since
`test.json` is unlabelled, we analyse the held-out 20% validation split."* Groups lose
credibility when they report test-set accuracy on unlabelled data.

---

## 5. Action plan, highest value first

| # | Action | Closes | Effort |
|---|---|---|---|
| 1 | Run `colab_bert_training.ipynb` end to end on a T4 GPU; save all printed metrics | (b), (d), (e) for the strongest model; one row of (f) | ~30 min |
| 2 | Add a parameter sweep for SVM (`C`), TF-IDF (`max_features`) and DistilBERT (learning rate); tabulate every value against validation macro F1 | **(c) entirely** | ~1 hr |
| 3 | Add test-set prediction to `SVM and MLP.ipynb` and write a submission from the best model | **(d)** | ~15 min |
| 4 | Run error analysis against SVM and DistilBERT; pick 1–2 quotable cases | **(e)** | ~30 min |
| 5 | Draw architecture figures for both submitted models | Completes **(b)** | ~45 min |
| 6 | Add an averaged-GloVe + LogReg run | Third row of **(f)** | ~30 min |
| 7 | Commit `results/part_b/` so the metrics are not laptop-only | Reproducibility | ~2 min |
| 8 | Write (g) and (h) — discussion only, no experiments needed | 20% | ~1 hr |

Items 1–4 are the ones that move marks. Items 5–8 are cheap and should not be left to the
final night.

---

## 6. Measured results so far

Shared protocol across all models: stratified 80/20 split, `random_state=42`, TF-IDF with
`lowercase=True`, `stop_words='english'`, `ngram_range=(1,2)`, `min_df=2`,
`max_features=20000`. Validation set = 1,481 reviews (217 negative, 1,264 positive).

| Model | Val accuracy | Macro F1 | Negative F1 | Train time |
|---|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.9095 | — | 0.72 | ~5 s |
| **TF-IDF + Linear SVM** | **0.9325** | **0.8585** | **0.7561** | 0.11 s |
| TF-IDF + SVD(200) + MLP (128,64) | 0.9115 | 0.8078 | 0.6667 | 4.60 s |
| DistilBERT fine-tuned | not run | not run | not run | — |

Dataset: 7,401 training reviews (6,319 positive / 1,082 negative, ~85% positive) and
1,851 unlabelled test reviews. Because of the imbalance, a trivial always-positive
classifier scores ~85% accuracy — **report macro F1 and per-class precision/recall
alongside accuracy throughout.**

## 7. Pre-submission checklist

- [ ] All 3 group members and individual contributions on the cover page
- [ ] Architecture figure included for each submitted model — question (b)
- [ ] Random seed / reproducibility stated — question (b)
- [ ] Parameter sweep table included — question (c)
- [ ] `submission.csv` regenerated from the **final** model, not the baseline — question (d)
- [ ] Error analysis states that the validation split was used, not the test set — question (e)
- [ ] Question (f) accuracy column filled with measured numbers from at least two feature formats
- [ ] Exactly three approaches named in (h)
- [ ] `submission.csv` has columns `reviews` and `predictions`, 1,851 rows
- [ ] All references and sources cited
- [ ] Submitted to NTULearn before Fri 13 Nov 2026, 11:59pm
