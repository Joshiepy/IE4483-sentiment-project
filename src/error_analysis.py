"""Helper for report question (e): look at what the model got right and wrong.

Run it with:  python src/error_analysis.py

It retrains the baseline on the training split, then prints a few validation
reviews the model got wrong (and a few it got right) so you have concrete
cases to discuss in the report.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from load_data import load_train

RANDOM_SEED = 42
N_EXAMPLES = 3          # how many cases of each kind to print
MAX_CHARS = 300         # truncate long reviews so the output stays readable

LABEL_NAMES = {0: "negative", 1: "positive"}


def show(title, rows):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    for _, row in rows.iterrows():
        text = row["review"][:MAX_CHARS].replace("\n", " ")
        print(f"\ntrue = {LABEL_NAMES[row['true']]} | predicted = {LABEL_NAMES[row['pred']]} "
              f"| confidence = {row['confidence']:.2f}")
        print(f"  {text}{'...' if len(row['review']) > MAX_CHARS else ''}")


def main():
    train_df = load_train()
    X_train, X_val, y_train, y_val = train_test_split(
        train_df["reviews"], train_df["sentiments"],
        test_size=0.2, random_state=RANDOM_SEED, stratify=train_df["sentiments"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=2, max_features=20000,
    )
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
    )
    model.fit(vectorizer.fit_transform(X_train), y_train)

    X_val_vec = vectorizer.transform(X_val)
    preds = model.predict(X_val_vec)
    probs = model.predict_proba(X_val_vec).max(axis=1)

    results = X_val.to_frame(name="review")
    results["true"] = y_val
    results["pred"] = preds
    results["confidence"] = probs

    wrong = results[results["true"] != results["pred"]]
    right = results[results["true"] == results["pred"]]

    # The most confident mistakes are usually the most interesting to discuss.
    show("WRONG - model was confident but wrong (good weakness examples)",
         wrong.sort_values("confidence", ascending=False).head(N_EXAMPLES))
    show("WRONG - model was unsure (borderline / ambiguous reviews)",
         wrong.sort_values("confidence").head(N_EXAMPLES))
    show("CORRECT - confident and right (good strength examples)",
         right.sort_values("confidence", ascending=False).head(N_EXAMPLES))

    # Which words push the model towards each class? Useful for part (e) and (f).
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    top_pos = sorted(zip(weights, feature_names), reverse=True)[:15]
    top_neg = sorted(zip(weights, feature_names))[:15]
    print(f"\n{'=' * 70}\nStrongest POSITIVE words/bigrams\n{'=' * 70}")
    print(", ".join(name for _, name in top_pos))
    print(f"\n{'=' * 70}\nStrongest NEGATIVE words/bigrams\n{'=' * 70}")
    print(", ".join(name for _, name in top_neg))


if __name__ == "__main__":
    main()
