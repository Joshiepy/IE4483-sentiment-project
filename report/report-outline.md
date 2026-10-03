# Report outline — IE4483 Mini Project (Option 1)

The brief asks eight questions, (a) to (h), worth 100% in total. Write the report
in this order so nothing gets missed. Marks in brackets.

**Cover page must state:** all 3 group members, and the specific contribution of each.

---

## (a) Feature format and preprocessing — 10%
- Which feature format did you choose (TF-IDF / word embeddings / pretrained LM) and **why**
- Exact preprocessing steps: lowercasing, stop-word removal, tokenisation, n-gram range,
  vocabulary size, how you handled rare words
- Mention that the vectoriser is fit on the **training split only**, never on validation/test

## (b) Model description — 20% (biggest single section)
- A **figure of the model architecture** — this is explicitly required, don't skip it
- Input and output dimensions at each layer
- Loss function (e.g. binary cross-entropy) and why
- Training strategy: optimiser, batch size, epochs, early stopping
- **Include the code and instructions to run it** (point to this repo's README)
- **Reproducibility:** state the fixed random seed (42) / averaging over runs / cross-validation

## (c) Hyperparameters and settings — 10%
- Learning rate, feature dimension, regularisation, etc.
- *How* you chose them — a small table of "value tried → validation score" is ideal
- Justify the reasoning, not just the final number

## (d) Test set predictions — 20%
- Apply the final classifier to `test.json`
- Report validation accuracy **and** precision/recall/F1
  (the data is ~85% positive, so accuracy alone overstates performance — say this)
- Attach `results/submission.csv`

## (e) Error analysis — 10%
- Pick **1–2 concrete cases**: one classified correctly, one incorrectly
- Quote the review text and explain *why* the model likely got it right/wrong
  (negation, sarcasm, mixed sentiment, rare vocabulary…)
- Run `python src/error_analysis.py` — it prints confident mistakes and the
  strongest positive/negative features for you
- Tie it back to strengths and weaknesses of your chosen model

## (f) Effect of feature format choice — 10%
Compare at least two options on **resource consumption** and **accuracy**:

| Feature format | Memory / compute | Typical accuracy | Notes |
|---|---|---|---|
| TF-IDF (bag of words) | very light, CPU only | baseline ~0.91 | ignores word order and context |
| Word embeddings (Word2Vec / GloVe) | moderate, needs pretrained vectors | usually higher | captures similarity, still context-free |
| Pretrained LM (BERT) | heavy, GPU strongly preferred | usually highest | contextual, handles negation/sarcasm better |

Fill the accuracy column with **your own measured numbers**, not guesses.

## (g) Transfer to hotel reviews with NO rating scores — 10%
- No labels means no supervised training → describe what breaks
- Options to discuss: zero-shot / pretrained sentiment models, lexicon-based scoring,
  transfer learning from this product-review model, semi-supervised or self-training,
  manually labelling a small set, domain adaptation (vocabulary differs: "comfortable bed"
  vs "comfortable shoes")

## (h) Hotel reviews with NOISY rating scores — 10%
- Explain how label noise hurts training (the model memorises wrong labels, overfits)
- **Name exactly three approaches** — the question asks for three. For example:
  1. Robust loss functions that down-weight likely-mislabelled examples
  2. Label cleaning / confident learning — find and drop suspicious labels
  3. Regularisation, early stopping, or ensembling to avoid fitting the noise

---

## Before submitting — checklist
- [ ] All 3 group members and individual contributions on the cover page
- [ ] Model architecture **figure** included (question b)
- [ ] Random seed / reproducibility stated (question b)
- [ ] Exactly three approaches named in (h)
- [ ] `submission.csv` attached, with columns `reviews` and `predictions`
- [ ] All references and sources cited
- [ ] Submitted to NTULearn before Fri 13 Nov 2026, 11:59pm
