"""Every model in the report, scored on one identical validation split.

Run it with:  python src/compare_models.py

Report questions (c) and (f) both need a table rather than a paragraph:

  (c) hyperparameters and why they were chosen
  (f) how the feature format trades resources against accuracy

The three TF-IDF models live in three different places - the baseline in
run_baseline.py, the SVM and the MLP in notebooks/02_svm_mlp.ipynb - so their
printed numbers came from three separate runs. They happen to agree because
everyone used seed 42 and the same 80/20 split, but "they happen to agree" is
not something to put in a report. This script refits all three against *one*
split and *one* TF-IDF vocabulary, so any difference between the rows is the
classifier and nothing else.

DistilBERT cannot be refit here - it needs a GPU and the transformers package.
Its row is read out of the committed outputs of notebooks/03_distilbert.ipynb
and is clearly marked, because it was validated on a 15% split rather than 20%
and so is NOT strictly comparable. notebooks/04_distilbert_20split.ipynb closes
that gap; if it has been run and has written results/distilbert_metrics.json,
this script prefers that file.

Writes:
  results/model_comparison.csv   - the tidy table for the report
  results/model_comparison.json  - the same thing plus confusion matrices,
                                   full configs and the resource numbers
"""

import csv
import json
import re
import time

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from load_data import PROJECT_ROOT, RESULTS_DIR, load_train

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Exactly the settings in run_baseline.py and notebooks/02_svm_mlp.ipynb, so
# the rows below differ only in the classifier that sits on top.
TFIDF_KWARGS = dict(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000,
)


def build_models():
    """The classifiers to compare, with the config to print for question (c)."""
    return [
        {
            "name": "TF-IDF + Logistic Regression",
            "short": "LogReg",
            "owner": "Joshua",
            "source": "src/run_baseline.py",
            "estimator": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
            ),
            "config": {
                "C": 1.0,
                "penalty": "l2",
                "class_weight": "balanced",
                "max_iter": 1000,
            },
            "why": "Linear, convex, and gives a calibrated probability per review, "
                   "so the 0.50 threshold is meaningful and the coefficients are "
                   "directly readable as evidence for question (e).",
        },
        {
            "name": "TF-IDF + Linear SVM",
            "short": "Linear SVM",
            "owner": "Jayce",
            "source": "notebooks/02_svm_mlp.ipynb",
            "estimator": LinearSVC(
                C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000
            ),
            "config": {
                "C": 1.0,
                "loss": "squared_hinge",
                "class_weight": "balanced",
                "max_iter": 5000,
            },
            "why": "A max-margin boundary suits 20,000 sparse features with only "
                   "5,920 training rows: it depends on the support vectors rather "
                   "than on every feature, so it is hard to overfit here.",
        },
        {
            "name": "TF-IDF + SVD(200) + MLP(128, 64)",
            "short": "MLP",
            "owner": "Jayce",
            "source": "notebooks/02_svm_mlp.ipynb",
            "estimator": make_pipeline(
                TruncatedSVD(n_components=200, random_state=RANDOM_SEED),
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    batch_size=64,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=10,
                    max_iter=150,
                    random_state=RANDOM_SEED,
                ),
            ),
            "config": {
                "svd_components": 200,
                "hidden_layers": [128, 64],
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.001,
                "learning_rate_init": 0.001,
                "batch_size": 64,
                "early_stopping": True,
                "max_iter": 150,
            },
            "why": "A non-linear control for the two linear models. An MLP cannot "
                   "take 20,000 sparse inputs cheaply, so TruncatedSVD compresses "
                   "them to 200 dense components first - which is itself the "
                   "feature-format trade-off question (f) asks about.",
        },
    ]


def count_parameters(name, fitted, n_features):
    """Learned parameters per model - the 'resources' half of question (f)."""
    if "Logistic" in name or "Linear SVM" in name:
        return n_features + 1                      # one weight per feature + bias
    if "MLP" in name:
        svd = fitted.named_steps["truncatedsvd"]
        mlp = fitted.named_steps["mlpclassifier"]
        svd_params = int(svd.components_.size)     # the 200 x 20,000 projection
        mlp_params = sum(int(w.size) for w in mlp.coefs_) + sum(
            int(b.size) for b in mlp.intercepts_
        )
        return svd_params + mlp_params
    return None


def distilbert_row():
    """Read DistilBERT's numbers out of committed output, never hardcoded.

    Prefers results/distilbert_metrics.json, which
    notebooks/04_distilbert_20split.ipynb writes after a matched 20% run. Falls
    back to scraping the final epoch line out of the committed outputs of
    notebooks/03_distilbert.ipynb.
    """
    matched = RESULTS_DIR / "distilbert_metrics.json"
    if matched.exists():
        with open(matched, encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["comparable"] = abs(metrics.get("test_size", 0) - TEST_SIZE) < 1e-9
        metrics.setdefault("source", "results/distilbert_metrics.json")
        return metrics

    nb_path = PROJECT_ROOT / "notebooks" / "03_distilbert.ipynb"
    if not nb_path.exists():
        return None
    with open(nb_path, encoding="utf-8") as f:
        notebook = json.load(f)

    # "Epoch 4/4 | Train Loss: 0.0253 | Val Acc: 0.9487 | Val Macro F1: 0.8951"
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+).*?Val Acc:\s*([0-9.]+).*?Val Macro F1:\s*([0-9.]+)"
    )
    best = None
    for cell in notebook.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") != "stream":
                continue
            text = "".join(out.get("text", ""))
            for match in pattern.finditer(text):
                epoch, total, acc, macro_f1 = match.groups()
                row = {
                    "epoch": int(epoch),
                    "epochs": int(total),
                    "accuracy": float(acc),
                    "macro_f1": float(macro_f1),
                }
                if best is None or row["epoch"] > best["epoch"]:
                    best = row
    if best is None:
        return None

    best.update({
        "name": "DistilBERT (4 epochs, max_length=256)",
        "short": "DistilBERT",
        "owner": "Ryan",
        "source": "notebooks/03_distilbert.ipynb (read from committed outputs)",
        "test_size": 0.15,
        "comparable": False,
        "config": {
            "model": "distilbert-base-uncased",
            "max_length": 256,
            "batch_size": 16,
            "epochs": 4,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup": "10% of total steps",
        },
        "why": "Contextual subword embeddings instead of a bag of words, so "
               "negation and word order survive. The cost is a GPU and roughly "
               "66M parameters, against 20,001 for the linear models.",
        "note": "Validated on a 15% split, not 20% - NOT directly comparable to "
                "the rows above. Run notebooks/04_distilbert_20split.ipynb to fix.",
        "parameters": None,
    })
    return best


def main():
    train_df = load_train()

    # --- one split, shared by every model -----------------------------------
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        train_df["reviews"].fillna(""),
        train_df["sentiments"],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_df["sentiments"],
    )

    # --- one TF-IDF vocabulary, fit on the training half only ---------------
    vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
    started = time.perf_counter()
    X_train = vectorizer.fit_transform(X_train_text)
    vectorize_seconds = time.perf_counter() - started
    X_val = vectorizer.transform(X_val_text)

    n_features = X_train.shape[1]
    vocab = vectorizer.get_feature_names_out()
    n_bigrams = int(sum(1 for term in vocab if " " in term))
    # MB here is 10^6 bytes, not 2^20, so these match the numbers already quoted
    # in CLAUDE.md and in the report (2.1 MB sparse against 947 MB dense).
    dense_mb = X_train.shape[0] * X_train.shape[1] * 8 / 1e6
    sparse_mb = (
        X_train.data.nbytes + X_train.indices.nbytes + X_train.indptr.nbytes
    ) / 1e6

    features = {
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "unigrams": int(len(vocab) - n_bigrams),
        "bigrams": n_bigrams,
        "cap_reached": bool(len(vocab) == TFIDF_KWARGS["max_features"]),
        "density_pct": round(100 * X_train.nnz / (X_train.shape[0] * X_train.shape[1]), 4),
        "active_terms_per_review": round(X_train.nnz / X_train.shape[0], 1),
        "sparse_mb": round(sparse_mb, 2),
        "dense_mb_if_dense": round(dense_mb, 1),
        "vectorize_seconds": round(vectorize_seconds, 3),
    }

    print(
        f"Shared split: {X_train.shape[0]} train / {X_val.shape[0]} validation, "
        f"seed {RANDOM_SEED}, stratified"
    )
    print(
        f"Shared TF-IDF: {features['train_shape']}, "
        f"{features['unigrams']} unigrams + {features['bigrams']} bigrams, "
        f"{features['density_pct']}% non-zero, {features['sparse_mb']} MB sparse "
        f"vs {features['dense_mb_if_dense']} MB dense\n"
    )

    rows = []
    for spec in build_models():
        started = time.perf_counter()
        spec["estimator"].fit(X_train, y_train)
        seconds = time.perf_counter() - started
        pred = spec["estimator"].predict(X_val)

        report = classification_report(
            y_val, pred, target_names=["negative", "positive"],
            output_dict=True, zero_division=0,
        )
        matrix = confusion_matrix(y_val, pred, labels=[0, 1])

        row = {
            "name": spec["name"],
            "short": spec["short"],
            "owner": spec["owner"],
            "source": spec["source"],
            "test_size": TEST_SIZE,
            "comparable": True,
            "accuracy": float(accuracy_score(y_val, pred)),
            "macro_f1": float(f1_score(y_val, pred, average="macro")),
            "negative_precision": report["negative"]["precision"],
            "negative_recall": report["negative"]["recall"],
            "negative_f1": report["negative"]["f1-score"],
            "positive_precision": report["positive"]["precision"],
            "positive_recall": report["positive"]["recall"],
            "positive_f1": report["positive"]["f1-score"],
            "confusion_matrix": {
                "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
                "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
            },
            "train_seconds": round(seconds, 3),
            "parameters": count_parameters(spec["name"], spec["estimator"], n_features),
            "config": spec["config"],
            "why": spec["why"],
        }
        rows.append(row)

        print(
            f"{spec['short']:<12} acc {row['accuracy']:.4f}  "
            f"macro F1 {row['macro_f1']:.4f}  "
            f"neg F1 {row['negative_f1']:.4f}  "
            f"{row['train_seconds']:>7.2f}s  "
            f"{row['parameters']:,} params"
        )

    bert = distilbert_row()
    if bert is not None:
        rows.append(bert)
        flag = "" if bert.get("comparable") else "   <- different split, see note"
        print(
            f"{bert['short']:<12} acc {bert['accuracy']:.4f}  "
            f"macro F1 {bert['macro_f1']:.4f}{flag}"
        )
    else:
        print("DistilBERT   no committed metrics found - row omitted")

    comparable = [r for r in rows if r.get("comparable")]
    best = max(comparable, key=lambda r: r["macro_f1"])
    print(
        f"\nBest strictly-comparable model: {best['name']} "
        f"({best['macro_f1']:.4f} macro F1)"
    )

    # --- write the report tables --------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": RANDOM_SEED,
        "split": f"stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}",
        "tfidf": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in TFIDF_KWARGS.items()
        },
        "features": features,
        "models": rows,
        "best_comparable": best["name"],
    }
    json_path = RESULTS_DIR / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    csv_cols = [
        "name", "owner", "accuracy", "macro_f1",
        "negative_precision", "negative_recall", "negative_f1",
        "positive_f1", "train_seconds", "parameters", "test_size", "comparable",
    ]
    csv_path = RESULTS_DIR / "model_comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=csv_cols, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in csv_cols})

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
