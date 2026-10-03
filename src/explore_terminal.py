"""Phase 1 exploration, drawn as text charts in the terminal.

Run it with:  python src/explore_terminal.py

Same numbers as notebooks/01_explore_data.ipynb, but the charts are drawn with
block characters instead of being saved as PNGs, so they show up over SSH, in a
plain terminal, and in Claude Code - no Jupyter, no image viewer.

The notebook is still the thing to read: it carries the written explanation of
what each number means for the report, and it produces the 200 dpi figures that
actually go into the PDF. This is for a quick look without leaving the terminal.
"""

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_data import load_test, load_train  # noqa: E402

try:
    import plotext as plt
except ImportError:
    sys.exit(
        "This script needs plotext for the terminal charts:\n"
        "    python -m pip install plotext\n"
        "(The notebook does not need it - only this script does.)"
    )

RANDOM_SEED = 42

# The charts are drawn with Unicode block characters. Windows consoles still
# default to cp1252, which cannot encode them, so the first chart would die with
# a UnicodeEncodeError. Harmless no-op on macOS and Linux.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Same two colours as the notebook figures, as close as 256-colour terminals get.
NEG, POS = "blue", "orange"

# Fit the actual terminal instead of assuming a width: anything wider than the
# window wraps and the charts turn to mush. Clamped because the labels need
# roughly 60 columns to stay readable, and past ~110 the bars just look stretched.
# get_terminal_size falls back to 80 when the output is piped or redirected.
WIDTH = max(60, min(shutil.get_terminal_size(fallback=(80, 24)).columns - 2, 110))


def heading(text):
    print(f"\n{text}")
    print("-" * len(text))


def chart(title, draw, bars=None):
    """Reset plotext, run `draw`, and print one chart. plotext keeps global
    state, so every chart has to clear the previous figure first.

    `bars` is the number of horizontal bars, if any. plotext silently drops
    row labels when the chart is shorter than the data, so the height is sized
    from the bar count rather than left at a fixed value."""
    plt.clear_figure()
    plt.theme("clear")
    plt.plotsize(WIDTH, 18 if bars is None else bars * 2 + 4)
    plt.title(title)
    draw()
    plt.show()


def main():
    train_df = load_train()
    test_df = load_test()

    heading("1. Dataset")
    print(f"train : {len(train_df):,} labelled reviews")
    print(f"test  : {len(test_df):,} unlabelled reviews")

    dup_rows = int(train_df.duplicated(subset=["reviews"]).sum())
    overlap = len(set(train_df["reviews"]) & set(test_df["reviews"]))
    print(f"duplicate rows in train       : {dup_rows}")
    print(f"reviews in both train and test: {overlap} "
          f"({overlap / len(test_df):.1%} of the test set)")

    # --- class balance ----------------------------------------------------
    heading("2. Class balance")
    counts = train_df["sentiments"].value_counts().sort_index()
    total = int(counts.sum())
    for label, name in ((0, "negative"), (1, "positive")):
        print(f"  {label} {name}: {counts[label]:>6,}  ({counts[label] / total:.1%})")
    print(f"\n  Always-positive baseline: {counts[1] / total:.1%} accuracy, "
          f"zero negatives found.\n  -> report macro F1, never accuracy alone.")

    chart(
        "Training labels (count)",
        lambda: plt.bar(["0 negative", "1 positive"],
                        [int(counts[0]), int(counts[1])],
                        color=[POS, NEG], orientation="horizontal"),
        bars=2,
    )

    # --- review length ----------------------------------------------------
    heading("3. Review length")
    words = train_df["reviews"].str.split().str.len()
    print(f"  median {words.median():.0f} words,  mean {words.mean():.1f},  "
          f"95th pct {words.quantile(0.95):.0f},  max {words.max()}")

    chart(
        "Words per review (tail clipped at 300)",
        lambda: plt.hist(words.clip(upper=300).tolist(), bins=40, color=NEG),
    )

    print("  Coverage if BERT truncates at max_length:")
    print(f"  {'max_length':>10} | {'by words':>9} | {'est. subword':>13}")
    for n in (128, 192, 256, 384, 512):
        print(f"  {n:>10} | {(words <= n).mean():>8.1%} | "
              f"{(words * 1.3 <= n).mean():>12.1%}")
    print("\n  -> 256 keeps 98.8% of reviews whole; 512 costs ~4x the attention")
    print("     compute for about 2% more coverage.")

    # Length by class, as two overlaid lines rather than two histograms -
    # two sets of bars in one terminal chart are unreadable.
    heading("3.1 Length by class")
    edges = np.linspace(0, 200, 26)
    centres = ((edges[:-1] + edges[1:]) / 2).tolist()
    series = {}
    for label, name, colour in ((0, "negative", NEG), (1, "positive", POS)):
        w = words[train_df["sentiments"] == label].clip(upper=200)
        share = np.histogram(w, bins=edges)[0] / len(w) * 100
        series[name] = (share.tolist(), colour)
        print(f"  {name}: median {w.median():>5.0f} words")

    def draw_lengths():
        for name, (share, colour) in series.items():
            plt.plot(centres, share, color=colour, label=name)

    chart("% of each class at that length (overlap => not a usable signal)",
          draw_lengths)

    # --- vocabulary -------------------------------------------------------
    heading("4. What the frequent words look like")
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(lowercase=True, stop_words="english",
                          ngram_range=(1, 2), min_df=2, max_features=20000)
    X = vec.fit_transform(train_df["reviews"])
    vocab = np.array(vec.get_feature_names_out())
    y = train_df["sentiments"].values

    pos_counts = np.asarray(X[y == 1].sum(axis=0)).ravel()
    neg_counts = np.asarray(X[y == 0].sum(axis=0)).ravel()
    is_unigram = np.array([" " not in t for t in vocab])

    rate_pos = pos_counts / int((y == 1).sum()) * 100
    rate_neg = neg_counts / int((y == 0).sum()) * 100
    top_pos = [i for i in np.argsort(-rate_pos) if is_unigram[i]][:10]
    top_neg = [i for i in np.argsort(-rate_neg) if is_unigram[i]][:10]
    shared = set(vocab[top_pos]) & set(vocab[top_neg])

    print(f"  top-10 words shared by both classes: {len(shared)} of 10")
    print(f"  {sorted(shared)}")
    print("  -> raw frequency barely separates the classes; this is what the")
    print("     IDF term in TF-IDF is for.")

    chart(
        "Most frequent words, negative reviews (per 100 reviews)",
        lambda: plt.bar(vocab[top_neg].tolist(), rate_neg[top_neg].tolist(),
                        color=NEG, orientation="horizontal"),
        bars=len(top_neg),
    )
    chart(
        "Most frequent words, positive reviews (per 100 reviews)",
        lambda: plt.bar(vocab[top_pos].tolist(), rate_pos[top_pos].tolist(),
                        color=POS, orientation="horizontal"),
        bars=len(top_pos),
    )

    # --- what actually discriminates --------------------------------------
    heading("5. Terms that actually discriminate (log-odds ratio)")
    V = len(vocab)
    log_odds = (
        np.log((pos_counts + 1) / (pos_counts.sum() + V))
        - np.log((neg_counts + 1) / (neg_counts.sum() + V))
    )
    scores = np.where((pos_counts + neg_counts) >= 15, log_odds, 0.0)

    k = 10
    neg_idx = np.argsort(scores)[:k]
    pos_idx = np.argsort(-scores)[:k]

    # Two separate charts rather than one diverging chart: plotext mis-renders a
    # horizontal bar chart that crosses zero. The diverging version is figure
    # fig05 in results/figures/, which is the one that goes in the report.
    # Magnitudes are plotted, with the direction stated in each title.
    chart(
        "Strongest NEGATIVE terms (|log-odds|, bigger = more negative)",
        lambda: plt.bar(vocab[neg_idx].tolist()[::-1],
                        np.abs(scores[neg_idx]).tolist()[::-1],
                        color=NEG, orientation="horizontal"),
        bars=k,
    )
    chart(
        "Strongest POSITIVE terms (log-odds, bigger = more positive)",
        lambda: plt.bar(vocab[pos_idx].tolist()[::-1],
                        scores[pos_idx].tolist()[::-1],
                        color=POS, orientation="horizontal"),
        bars=k,
    )

    print("  Strongest negative signal is the customer mentioning a RETURN.")
    print("  Brand names leak in on the positive side (nike, clarks) - a")
    print("  generalisation risk worth checking in the error analysis.")

    print("\nFull write-up and the 200 dpi report figures:")
    print("  notebooks/01_explore_data.ipynb  ->  results/figures/")


if __name__ == "__main__":
    main()
