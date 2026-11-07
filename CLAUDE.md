# CLAUDE.md

Context for Claude Code. This file is read automatically at the start of every
session, so it should hold things that stay true - not a log of past chats.

## What this project is

NTU **IE4483 Artificial Intelligence and Data Mining** mini project, **Option 1:
sentiment classification of product reviews**. Binary: 0 = negative, 1 = positive.

- **Deadline:** Friday 13 Nov 2026, 11:59pm, submitted to NTULearn
- **Deliverables:** one group report (PDF) + `submission.csv`
- **Group of three.** The report must state each member's individual contribution.
- The full brief is `report/IE4483-Project1-brief.pdf`; the eight graded questions
  (a) to (h) are broken down in `report/report-outline.md`.

**The data.** `data/train.json` holds 7,401 labelled reviews (6,319 positive,
1,082 negative) and `data/test.json` holds 1,851 unlabelled ones. The fields are
`reviews` (raw text) and `sentiments` (0 or 1). They are Amazon-style product
reviews, heavily about shoes and clothing; the median review is 42 words.

**The method.** One model is finished: TF-IDF (lowercased, English stop words,
1-2 grams, `min_df=2`, 20,000 features) feeding a logistic regression with
`class_weight="balanced"`, all in `src/run_baseline.py`. A DistilBERT model is
planned but **not trained yet**, so every number in the report today comes from
this baseline.

**Measured results** - seed 42, 1,481-review validation split, re-verified 30 Oct 2026:

| Metric | Value |
|---|---|
| Validation accuracy | 0.9095 |
| Negative precision / recall / F1 | 0.66 / 0.80 / 0.72 |
| Positive precision / recall / F1 | 0.96 / 0.93 / 0.95 |
| Macro F1 | 0.83 |
| Confusion matrix | TN 173, FP 44, FN 90, TP 1174 |

The feature matrix is 5,920 x 20,000 and the 20,000 cap is actually reached:
6,120 unigrams + 13,880 bigrams, 0.15% non-zero, about 29 active terms per
review, 2.1 MB sparse against 947 MB if it were dense. Those are the resource
numbers question (f) needs for the TF-IDF side of the comparison.

## What the report must include

The brief asks eight questions worth 100% between them. `report/report-outline.md`
has the full breakdown; this is the short version plus the requirements that are
easiest to lose marks on.

| Q | Section | Marks | Status |
|---|---|---|---|
| (a) | Feature format + preprocessing | 10% | figures placed, prose to write |
| (b) | Model description + architecture figure + code | 20% | figure placed, prose to write |
| (c) | Hyperparameters + justification | 10% | needs the experiment table |
| (d) | Test predictions + `submission.csv` | 20% | numbers measured, prose to write |
| (e) | Error analysis, 1-2 cases | 10% | run `src/error_analysis.py` |
| (f) | Feature format vs resources and accuracy | 10% | needs a second model |
| (g) | Hotel reviews, no rating scores | 10% | **written** |
| (h) | Hotel reviews, noisy rating scores | 10% | **written** |

Hard requirements - these are explicit in the brief and are where marks get lost:

- **The cover page lists all three members and what each one did individually.**
- **(b) needs an architecture figure**, not a prose description of the model.
- **(b) must state reproducibility** - seed 42, fixed in every script.
- **(h) must name exactly three approaches**, clearly numbered.
- **Never report accuracy alone.** At 85% positive, always guessing "positive"
  scores 85%. Give macro F1 and per-class precision and recall, and say why.
- **Use our own measured numbers.** Never fill a table from a blog post.
- **Cite every source**, APA 7th, alphabetical.
- `submission.csv` has columns `reviews` and `predictions`, 1,851 rows.

## The report document

The report itself is a Word file and lives outside this repo.

- **Original - never edit it:** the Teams-synced copy shared with the group.
  Editing it would collide with whatever a teammate has open.
- **Working copy - the only file to edit:**
  `C:\Users\joshu\OneDrive - Nanyang Technological University\IE4483\IE4483 Project Report Team 67.docx`
  This is Joshua's own copy of the Teams document.
- **Back it up before any scripted edit**, and close it in Word first - Word holds
  a lock and OneDrive syncs in the background.
- Sections 8 and 9, questions (g) and (h), are **already written** and fully
  referenced. Do not rewrite them; they are the strongest part of the report.

## Figures

- Every chart is a **PNG in `results/figures/`** at 200 dpi - that directory, not
  a bare `figures/`, because `results/` is where generated output already lives.
- Figures are **produced by code, never drawn by hand**: the EDA ones come from
  `notebooks/01_explore_data.ipynb`, the architecture one from
  `src/make_architecture_figure.py`. If a figure is wrong, fix the script and
  re-run it rather than editing the image or the Word copy.
- The palette is fixed so every figure looks like it came from one place:
  negative `#2a78d6`, positive `#eb6834`, paper `#fcfcfb`.
- In the Word file each figure is sized to 6.27 in (A4 width minus the 1 in
  margins), centred, and followed by a `Caption`-styled line built on a
  `SEQ Figure` field so Word renumbers them automatically.

## Branching

- `main` - stable. Do not commit here directly.
- `joshua-dev` - Joshua's working branch. Default to this one.

Always check the current branch before committing. Merge to `main` via a pull
request when something is finished, not by pushing to it.

## Layout

```
data/        given by the course - READ ONLY, never edit or regenerate
notebooks/   exploration + colab_bert_training.ipynb (needs a GPU)
src/         the pipeline scripts
results/     generated output only - never hand-edit, re-run the script instead
report/      the brief and the report outline
```

## Commands

```bash
python src/load_data.py       # sanity check - prints dataset sizes
python src/run_baseline.py    # TF-IDF + LogReg, writes results/submission.csv
python src/error_analysis.py  # right/wrong examples for report question (e)
python src/make_architecture_figure.py   # fig06, the architecture figure for (b)
```

## Things that matter when working on this

**The labels are imbalanced:** 6,319 positive vs 1,082 negative in training (~85%
positive). Always guessing "positive" scores ~85% accuracy while being useless, so
**never report accuracy alone** - include per-class precision, recall and F1.

**`test.json` has no labels.** Test accuracy cannot be measured locally. That is why
the code holds out 20% of `train.json` as a validation split, stratified.

**Current baseline to beat:** TF-IDF (1-2 grams, 20k features) + Logistic Regression
= **0.9095 validation accuracy**. Any new model should be compared against this.

**Phase 1 findings** (from `notebooks/01_explore_data.ipynb`, 3 Oct 2026):

- **Use `max_length=256` for BERT.** 98.8% of reviews fit whole by word count (~97.7% after
  subword expansion). Median review is 42 words; 512 buys ~2% more coverage at ~4x the
  attention cost.
- **`train.json` contains 216 duplicate rows** (151 distinct texts, no conflicting labels),
  so copies straddle the 80/20 split and validation accuracy is mildly optimistic. We are
  **not** de-duplicating - that would move the baseline everyone is comparing against.
  Mention it in the report instead.
- **87 reviews appear in both `train.json` and `test.json`** (~4.7% of the test set). A
  property of the given data; `data/` is read-only. Worth one honest sentence in (d).
- **Stop-word removal invents bigrams.** `lower problems` ranks as the 2nd strongest
  negative term but nobody wrote it - it is "lower back problems" with the stop word
  `back` deleted, fusing two non-adjacent words. The reviews are about back pain, not bad
  products. Do not quote it in the report as evidence of anything but this. Worth a
  sentence in (a) on why stop-word removal and n-grams interact badly.
- **Bigrams help, but not via negation.** Of 155 negation bigrams in the vocabulary only 11
  are frequent enough to score and only 80 lean negative. `doesn work` occurs 7 times in
  the whole corpus. Bigrams pay off on fixed phrases (`waste money`, `highly recommend`).
  This is the data-driven argument for BERT - use it in (f).

**Reproducibility is graded** (question b). Every script fixes `RANDOM_SEED = 42`.
Keep it that way, or average over multiple runs instead.

**BERT needs a GPU.** Fine-tuning DistilBERT is 30-60 min per epoch on CPU versus
about 2 min on a free Colab T4. Use `notebooks/colab_bert_training.ipynb` in Colab.
GitHub Codespaces has no GPU on the free plan.

**Submission format:** two columns, `reviews` and `predictions`, matching
`data/submission_example.csv`. 1,851 rows. Write CSVs with `lineterminator="\n"`
so output stays identical across the group's different machines.

## Conventions

- Keep scripts readable over clever - this is coursework that gets marked, and
  teammates need to follow it.
- Never commit trained model weights. `.gitignore` blocks `*.pt`, `*.pth`, `models/`.
- Clear notebook outputs before committing, or merges become unreadable JSON diffs.
- Record hyperparameters you tried and their scores as you go - question (c) asks
  you to justify these choices, and reconstructing them later is miserable.

## Academic integrity

This is assessed work under the NTU Honour Code. Claude may help write, debug and
explain code, but the group must understand and be able to defend everything
submitted. All sources and references must be cited in the report.

## Where to start

Read `report/project-plan.md` - it breaks the eight questions into phases, flags
which ones need experiments, and lists what separates an A from a B.
