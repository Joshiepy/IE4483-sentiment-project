"""Linear SVM predictions on the validation split, for report question (e).

Questions (e) and (f) compare DistilBERT (the submitted model) against the
Linear SVM (the best TF-IDF model). DistilBERT's per-review output comes from
notebooks/04_distilbert_20split.ipynb as results/distilbert_val_predictions.csv.
This script produces the matching SVM file on the identical split, so the two
models can be read side by side on the same reviews.

    python src/svm_val_predictions.py

Writes results/svm_val_predictions.csv. If the DistilBERT file is present it
also prints where the two models disagree and how each handles the cases
quoted in the report.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from compare_models import RANDOM_SEED, TEST_SIZE, TFIDF_KWARGS, build_models
from load_data import RESULTS_DIR, load_train

LABEL = {0: "negative", 1: "positive"}

# The reviews quoted in the report's error analysis (Section 6). Each snippet
# matches exactly one validation review.
REPORT_CASES = {
    "Case 1 (likely wrong label)": "Dilvered quickly",
    "Case 1b (likely wrong label)": "Saucony Jazz",
    "Case 2 (product vs service)": "It took them weeks",
    "Case 3 (misspelling)": "a little dissapointed",
}


def top_terms(text, vectorizer, svm, n=3):
    """The terms in one review that push the SVM hardest each way."""
    row = vectorizer.transform([text])
    vocab = vectorizer.get_feature_names_out()
    contrib = {
        vocab[j]: row[0, j] * svm.coef_[0, j] for j in row.nonzero()[1]
    }
    ranked = sorted(contrib.items(), key=lambda kv: kv[1])
    neg = ", ".join(f"{t} ({c:+.2f})" for t, c in ranked[:n] if c < 0)
    pos = ", ".join(f"{t} ({c:+.2f})" for t, c in ranked[::-1][:n] if c > 0)
    return neg or "-", pos or "-"


def main():
    train_df = load_train()
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        train_df["reviews"].fillna(""),
        train_df["sentiments"],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_df["sentiments"],
    )

    vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    svm = next(m for m in build_models() if m["short"] == "Linear SVM")["estimator"]
    svm.fit(X_train, y_train)

    val = pd.DataFrame({
        "reviews": X_val_text.tolist(),
        "true": y_val.tolist(),
        "predicted": svm.predict(X_val).tolist(),
        # Signed distance from the boundary: > 0 means positive. Not a probability.
        "decision_score": svm.decision_function(X_val).round(4).tolist(),
    })
    val["correct"] = val["true"] == val["predicted"]
    out = RESULTS_DIR / "svm_val_predictions.csv"
    val.to_csv(out, index=False, lineterminator="\n")
    print(f"SVM macro F1 {f1_score(val['true'], val['predicted'], average='macro'):.4f} "
          f"| {(~val['correct']).sum()} wrong of {len(val)} | wrote {out}")

    print("\n=== Report cases, Linear SVM ===")
    for name, snippet in REPORT_CASES.items():
        for _, r in val[val["reviews"].str.contains(snippet, regex=False)].iterrows():
            neg, pos = top_terms(r["reviews"], vectorizer, svm)
            print(f"{name}: true {LABEL[r['true']]}, SVM predicts "
                  f"{LABEL[r['predicted']]} (score {r['decision_score']:+.2f})")
            print(f"    strongest negative terms: {neg}")
            print(f"    strongest positive terms: {pos}")

    bert_path = RESULTS_DIR / "distilbert_val_predictions.csv"
    if not bert_path.exists():
        print(f"\n{bert_path.name} not found - re-run notebooks/04_distilbert_20split.ipynb "
              "on Colab and copy it into results/ for the side-by-side comparison.")
        return

    bert = pd.read_csv(bert_path)
    if bert["reviews"].tolist() != val["reviews"].tolist():
        raise SystemExit("DistilBERT file is not on the same validation split - stopping.")
    both = val.assign(bert_predicted=bert["predicted"], bert_prob_positive=bert["prob_positive"])
    both["bert_correct"] = both["bert_predicted"] == both["true"]

    print("\n=== DistilBERT vs Linear SVM on the same 1,481 reviews ===")
    print(pd.crosstab(both["correct"].map({True: "SVM right", False: "SVM wrong"}),
                      both["bert_correct"].map({True: "BERT right", False: "BERT wrong"})))

    def show(df, title, n=5):
        print(f"\n--- {title} ({len(df)} reviews) ---")
        for _, r in df.head(n).iterrows():
            print(f"true {LABEL[r['true']]} | SVM {LABEL[r['predicted']]} "
                  f"| DistilBERT {LABEL[r['bert_predicted']]} (P(pos) {r['bert_prob_positive']:.2f})")
            print("   ", r["reviews"][:300].replace("\n", " "))

    show(both[both["bert_correct"] & ~both["correct"]], "DistilBERT right, SVM wrong - strengths")
    show(both[~both["bert_correct"] & both["correct"]], "SVM right, DistilBERT wrong - weaknesses")
    show(both[~both["bert_correct"] & ~both["correct"]], "Both wrong - possible label noise")

    print("\n=== Report cases, both models ===")
    for name, snippet in REPORT_CASES.items():
        for _, r in both[both["reviews"].str.contains(snippet, regex=False)].iterrows():
            print(f"{name}: true {LABEL[r['true']]} | SVM {LABEL[r['predicted']]} "
                  f"| DistilBERT {LABEL[r['bert_predicted']]} (P(pos) {r['bert_prob_positive']:.2f})")


if __name__ == "__main__":
    main()
