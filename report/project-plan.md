# Plan: how to get an A on this project

Written 3 Oct 2026. Deadline **Fri 13 Nov 2026, 11:59pm** - about six weeks.

## Where the marks actually are

| Q | Topic | Marks | Needs experiments? |
|---|-------|-------|--------------------|
| (a) | Feature format + preprocessing | 10% | no - describes what you did |
| (b) | Model description + architecture figure + code | **20%** | yes |
| (c) | Hyperparameter choices + justification | 10% | yes - needs a results table |
| (d) | Test predictions + submission.csv | **20%** | yes |
| (e) | Error analysis, 1-2 cases | 10% | yes |
| (f) | How feature format affects resources + accuracy | 10% | yes - needs 2+ models compared |
| (g) | Hotel reviews with NO labels | 10% | **no - pure discussion** |
| (h) | Hotel reviews with NOISY labels, 3 approaches | 10% | **no - pure discussion** |

**The single most important observation: (g) and (h) are 20% of the grade and require
no code at all.** Most groups run out of time and write them badly the night before.
Write them early and well - it is the cheapest 20% on the paper.

**Second observation: (f) forces you to compare at least two feature formats.** You
already have TF-IDF. Once you also run BERT, (f) becomes nearly free because you will
have both sets of numbers. This is why doing BERT pays for itself twice.

---

## Phase 0 - already done

- TF-IDF + Logistic Regression baseline: **0.9095 validation accuracy**
- `submission.csv` generated in the correct format
- `src/error_analysis.py` ready to produce material for (e)

This alone would pass. It will not get an A, because the brief explicitly encourages
neural models and (f) needs a comparison.

## Phase 1 - explore the data (half a day)

Make `notebooks/01_explore_data.ipynb`:

- Review length distribution - justifies the `max_length=256` choice for BERT
- Most common words per class
- Class balance chart (85/15 split) - **put this figure in the report**, it motivates
  every later decision about metrics and class weighting

Feeds question (a).

## Phase 2 - train the strong model (one session, 1-2 hours)

Run `notebooks/colab_bert_training.ipynb` in Colab with a **T4 GPU enabled**.

Record before closing the tab - Colab deletes everything:

- validation accuracy, macro F1, per-class precision/recall
- the confusion matrix
- the error-analysis examples it prints
- download `submission.csv`

Feeds (b), (d), (e), (f).

## Phase 3 - hyperparameter experiments (one session)

Question (c) wants justification, not a number. Run the Colab notebook three times
changing one thing each time, and fill in a table like this:

| Learning rate | Epochs | Max length | Val accuracy | Macro F1 |
|---|---|---|---|---|
| 1e-5 | 3 | 256 | | |
| 2e-5 | 3 | 256 | | |
| 5e-5 | 3 | 256 | | |

Then write *why* the winner won. The expected story: 5e-5 is too aggressive and
damages the pretrained weights, 1e-5 underfits in three epochs, 2e-5 is the value
the BERT paper recommends for exactly this reason.

**Record these in CLAUDE.md as you go** so the table is not reconstructed from memory.

## Phase 4 - error analysis (half a day)

Pick **one correct and one incorrect** case. Quote the actual review text. Explain the
mechanism, not just the outcome. The strongest cases are usually:

- **Negation** - "not what I expected" where bag-of-words sees only positive words
- **Mixed sentiment** - praises shipping, criticises the product
- **Sarcasm** - positive words, negative meaning
- **Rare vocabulary** - words the TF-IDF vocabulary dropped via `min_df=2`

Contrast the two models here if you can: a case TF-IDF gets wrong and BERT gets right
is the single most persuasive paragraph you can write, because it demonstrates
*why* contextual embeddings matter rather than asserting it.

## Phase 5 - the discussion questions (one day, no code)

**(g) Hotel reviews, no rating scores.** No labels means no supervised training. Cover:
what breaks, then concrete fixes - zero-shot classification with a pretrained sentiment
model, lexicon-based scoring (VADER), transfer learning from this product-review model,
semi-supervised self-training, hand-labelling a small evaluation set. Discuss **domain
shift** explicitly: "comfortable" means something different about a bed than about shoes,
and hotel reviews mention staff, location and cleanliness, which this vocabulary has
never seen.

**(h) Hotel reviews, noisy labels.** First explain the damage: the model memorises wrong
labels, and validation becomes untrustworthy because the validation labels are noisy too.
Then name **exactly three** approaches - the question says three, so count them:

1. Robust loss functions that down-weight probable mislabels
2. Label cleaning / confident learning - find suspect labels and drop or relabel them
3. Regularisation, early stopping, or ensembling so the model fits the signal not the noise

Justify each. Do not just list them.

## Phase 6 - write and assemble (one week, do not compress this)

Follow `report/report-outline.md` section by section.

---

## What separates an A from a B here

1. **Draw the architecture figure.** Question (b) is 20% and explicitly asks for a
   figure. Groups lose marks every year by describing the model in prose instead.
2. **Never report accuracy alone.** At 85% positive, a model that always guesses
   "positive" scores 85%. Report macro F1 and per-class precision/recall, and say in
   the report *why* you are doing so. This single sentence signals you understood the
   data rather than just ran a library.
3. **Show two models, not one.** The comparison is what makes (f) answerable with real
   numbers instead of textbook claims.
4. **State reproducibility explicitly.** Seed 42, fixed in every script. It is named
   in question (b) and costs one sentence.
5. **Answer exactly what is asked.** (h) says three approaches - give three, clearly
   numbered. Markers tick boxes.
6. **Use your own measured numbers everywhere.** Never fill the (f) table with values
   from a blog post.
7. **Cite everything** - BERT, GloVe, the Blitzer dataset paper, any library or
   tutorial. The brief asks for this and the NTU Honour Code applies.
8. **Cover page: all three members and who did what.** Explicitly required.

## Suggested split for three people

Rotate so nobody only writes prose, but roughly:

- **Person 1 - modelling:** Phases 2 and 3, writes (b) and (c)
- **Person 2 - data and analysis:** Phases 1 and 4, writes (a), (e) and (f)
- **Person 3 - discussion and assembly:** Phase 5, writes (g) and (h), owns the final
  document, formatting, citations and the contribution table

Everyone commits under their own GitHub account - it is a timestamped record of
individual contribution, which the report has to state anyway.

## Suggested timeline

| Week of | Goal |
|---|---|
| 6 Oct | Phase 1 explore; Person 3 drafts (g) and (h) |
| 13 Oct | Phase 2 BERT trained, numbers recorded |
| 20 Oct | Phase 3 hyperparameter table complete |
| 27 Oct | Phase 4 error analysis; architecture figure drawn |
| 3 Nov | Full draft assembled, all sections written |
| 10 Nov | Proofread, check the submission checklist, **submit early** |

Submitting on 11 Nov instead of 13 Nov costs nothing and protects you from NTULearn
being slow, a laptop dying, or a groupmate falling ill.

## Before you submit

Run through the checklist at the bottom of `report/report-outline.md`.
