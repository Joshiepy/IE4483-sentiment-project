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
