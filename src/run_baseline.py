"""Baseline end-to-end pipeline: TF-IDF features + Logistic Regression.

Run it with:  python src/run_baseline.py

What it does (matching steps 1-5 of the project brief):
  1. load train.json / test.json
  2. split train into a train part and a validation part
  3. turn raw text into TF-IDF features
  4. fit a Logistic Regression classifier and score it on validation
  5. predict the test set and write results/submission_baseline.csv
     (NOT submission.csv - that is the final DistilBERT file)

This is a *baseline*. The brief encourages a neural model (RNN or BERT),
so treat this as the number your better model has to beat.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from load_data import RESULTS_DIR, load_test, load_train

# Fixing the seed keeps results reproducible - the brief asks for this in part (b).
RANDOM_SEED = 42


def main():
    # --- 1. Load ------------------------------------------------------------
    train_df = load_train()
    test_df = load_test()

    # --- 2. Hold out a validation split (we have no labels for test.json) ---
    # stratify keeps the same positive/negative ratio in both splits, which
    # matters here because the dataset is imbalanced (~85% positive).
    X_train, X_val, y_train, y_val = train_test_split(
        train_df["reviews"],
        train_df["sentiments"],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=train_df["sentiments"],
    )

    # --- 3. Text -> features ------------------------------------------------
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams, e.g. "waste money". Note: stop_words
                              # removes "not" first, so "not good" is NOT a feature.
        min_df=2,             # ignore words appearing in only one review
        max_features=20000,
    )
    X_train_vec = vectorizer.fit_transform(X_train)   # fit on TRAIN ONLY
    X_val_vec = vectorizer.transform(X_val)

    # --- 4. Train + validate ------------------------------------------------
    # class_weight="balanced" counteracts the positive-heavy label imbalance.
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train_vec, y_train)

    val_pred = model.predict(X_val_vec)
    print(f"Validation accuracy: {accuracy_score(y_val, val_pred):.4f}\n")
    print(classification_report(y_val, val_pred, target_names=["negative", "positive"]))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_val, val_pred))

    # --- 5. Predict the test set and write the submission -------------------
    # Refit on ALL labelled data now that the settings are chosen.
    full_vec = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=2, max_features=20000,
    )
    X_full = full_vec.fit_transform(train_df["reviews"])
    final_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
    )
    final_model.fit(X_full, train_df["sentiments"])

    test_pred = final_model.predict(full_vec.transform(test_df["reviews"]))

    submission = test_df.copy()
    submission["predictions"] = test_pred
    out_path = RESULTS_DIR / "submission_baseline.csv"
    # The explicit line terminator keeps this file byte-identical on Windows,
    # macOS and Linux. Without it, re-running on a different OS shows up as a
    # fake 1800-line diff for your groupmates.
    submission.to_csv(out_path, index=False, lineterminator="\n")
    print(f"\nWrote {len(submission)} predictions to {out_path}")


if __name__ == "__main__":
    main()
