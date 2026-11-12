"""The hyperparameter experiments behind report question (c).

Run it with:  python src/tune_hyperparameters.py   (about a minute)

Question (c) asks which hyperparameters were tried and why the final values
were chosen. Answering it needs evidence, not an assertion, so this script
varies one setting at a time around the chosen configuration and scores each
variant on the same stratified 20% validation split at seed 42.

One-factor-at-a-time rather than a full grid, deliberately: a grid over these
six settings is several hundred fits whose winner would be chosen on a
1,481-review split and would mostly be measuring noise. Varying one setting
shows the *direction and size* of each effect, which is what the report has to
justify.

Macro F1 is the selection metric throughout, never accuracy: the data is 85.4%
positive, so predicting "positive" everywhere already scores 0.853 accuracy.

Writes results/hyperparameter_search.csv
"""

import csv
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from load_data import RESULTS_DIR, load_train

RANDOM_SEED = 42
TEST_SIZE = 0.2

# The configuration every variant is measured against.
BASE_TFIDF = dict(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000,
)
BASE_CLF = dict(C=1.0, class_weight="balanced")

# One entry per experiment: which knob, and the values to try.
TFIDF_SWEEPS = [
    ("max_features", [5000, 10000, 20000, 50000, None]),
    ("ngram_range", [(1, 1), (1, 2), (1, 3)]),
    ("min_df", [1, 2, 3, 5]),
    ("stop_words", ["english", None]),
    ("sublinear_tf", [False, True]),
]
CLF_SWEEPS = [
    ("C", [0.01, 0.1, 1.0, 10.0, 100.0]),
    ("class_weight", ["balanced", None]),
]


def make_classifier(name, **overrides):
    params = dict(BASE_CLF)
    params.update(overrides)
    if name == "LogReg":
        return LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, **params
        )
    return LinearSVC(max_iter=5000, random_state=RANDOM_SEED, **params)


def evaluate(vec_kwargs, clf_name, clf_kwargs, split):
    X_train_text, X_val_text, y_train, y_val = split
    vectorizer = TfidfVectorizer(**vec_kwargs)
    started = time.perf_counter()
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    model = make_classifier(clf_name, **clf_kwargs)
    model.fit(X_train, y_train)
    seconds = time.perf_counter() - started
    pred = model.predict(X_val)
    return {
        "features": X_train.shape[1],
        "accuracy": float(accuracy_score(y_val, pred)),
        "macro_f1": float(f1_score(y_val, pred, average="macro")),
        "negative_f1": float(f1_score(y_val, pred, pos_label=0)),
        "seconds": round(seconds, 2),
    }


def main():
    train_df = load_train()
    split = train_test_split(
        train_df["reviews"].fillna(""),
        train_df["sentiments"],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_df["sentiments"],
    )

    rows = []

    for clf_name in ("LogReg", "Linear SVM"):
        baseline = evaluate(dict(BASE_TFIDF), clf_name, {}, split)
        print(f"\n### {clf_name} - chosen configuration: "
              f"macro F1 {baseline['macro_f1']:.4f}, "
              f"{baseline['features']:,} features")

        for knob, values in TFIDF_SWEEPS:
            print(f"\n  {knob}")
            for value in values:
                vec_kwargs = dict(BASE_TFIDF)
                vec_kwargs[knob] = value
                result = evaluate(vec_kwargs, clf_name, {}, split)
                chosen = BASE_TFIDF.get(knob, False) == value
                delta = result["macro_f1"] - baseline["macro_f1"]
                rows.append({
                    "model": clf_name, "knob": knob, "value": str(value),
                    "chosen": chosen, **result,
                    "macro_f1_delta": round(delta, 4),
                })
                print(f"    {str(value):<12} macro F1 {result['macro_f1']:.4f} "
                      f"({delta:+.4f})  neg F1 {result['negative_f1']:.4f}  "
                      f"{result['features']:>6,} feat  {result['seconds']:>5.2f}s"
                      f"{'   <- chosen' if chosen else ''}")

        for knob, values in CLF_SWEEPS:
            print(f"\n  {knob}")
            for value in values:
                result = evaluate(dict(BASE_TFIDF), clf_name, {knob: value}, split)
                chosen = BASE_CLF.get(knob) == value
                delta = result["macro_f1"] - baseline["macro_f1"]
                rows.append({
                    "model": clf_name, "knob": knob, "value": str(value),
                    "chosen": chosen, **result,
                    "macro_f1_delta": round(delta, 4),
                })
                print(f"    {str(value):<12} macro F1 {result['macro_f1']:.4f} "
                      f"({delta:+.4f})  neg F1 {result['negative_f1']:.4f}  "
                      f"{result['features']:>6,} feat  {result['seconds']:>5.2f}s"
                      f"{'   <- chosen' if chosen else ''}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "hyperparameter_search.csv"
    cols = ["model", "knob", "value", "chosen", "features",
            "accuracy", "macro_f1", "macro_f1_delta", "negative_f1", "seconds"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}  ({len(rows)} experiments)")


if __name__ == "__main__":
    main()
