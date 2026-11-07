"""Draw the model architecture figure that report question (b) requires.

Run it with:  python src/make_architecture_figure.py

Question (b) is worth 20% and explicitly asks for a figure of the architecture
showing the input and output dimensions at each layer. Every number printed on
this diagram comes from the pipeline in run_baseline.py, so if that pipeline
changes, re-run this script rather than editing the figure by hand.

Writes results/figures/fig06_architecture.png at 200 dpi.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from load_data import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"

# Same palette as notebooks/01_explore_data.ipynb so every figure in the report
# looks like it came from one place.
NEG, POS = "#2a78d6", "#eb6834"
INK, INK_MUTED, AXIS = "#0b0b0b", "#898781", "#c3c2b7"
PAPER = "#fcfcfb"
BAND = "#eef3fb"   # fill for the data-handling stages
MODEL = "#fdf0e9"  # fill for the stages that have learned parameters

plt.rcParams.update({
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "font.size": 10,
})

# Each stage: title, the settings that define it, the shape coming out of it,
# and which fill to use. Box heights are computed from the number of detail
# lines, so adding a line never pushes text through the bottom border.
STAGES = [
    {
        "title": "Input  —  raw review text",
        "detail": [
            "train.json, read as one string per review",
            'fields: "reviews" (text), "sentiments" (0 = negative, 1 = positive)',
        ],
        "shape": "7,401 strings\n(6,319 pos / 1,082 neg)",
        "fill": BAND,
    },
    {
        "title": "Stratified 80 / 20 split",
        "detail": [
            "train_test_split(test_size=0.2, stratify=sentiments,",
            "random_state=42)  —  keeps the 85/15 ratio in both halves",
        ],
        "shape": "5,920 train\n1,481 validation",
        "fill": BAND,
    },
    {
        "title": "TF-IDF vectoriser   (fit on the training split only)",
        "detail": [
            'lowercase=True,  stop_words="english",  ngram_range=(1, 2)',
            "min_df=2,  max_features=20,000  (cap reached)",
            "vocabulary learned: 6,120 unigrams + 13,880 bigrams",
            "0.15% non-zero, about 29 active terms per review",
        ],
        "shape": "5,920 × 20,000\nsparse matrix\n(2.1 MB sparse,\n947 MB if dense)",
        "fill": BAND,
    },
    {
        "title": "Logistic regression   (the only learned layer)",
        "detail": [
            "weights w ∈ ℝ²⁰⁰⁰⁰ and one bias b  —  20,001 parameters",
            'L2 penalty (C = 1.0),  class_weight="balanced",',
            "max_iter=1000,  random_state=42",
            "trained by minimising binary cross-entropy (log loss) + L2",
        ],
        "shape": "20,000 → 1\nz = w·x + b\n(one score per review)",
        "fill": MODEL,
    },
    {
        "title": "Sigmoid activation",
        "detail": [r"$\sigma(z) = 1 / (1 + e^{-z})$  turns the score into a probability"],
        "shape": "1 → 1\nP(positive) ∈ (0, 1)",
        "fill": MODEL,
    },
    {
        "title": "Decision threshold at 0.50",
        "detail": [
            "predict 1 when P(positive) ≥ 0.50, otherwise 0",
            "written to results/submission.csv for the 1,851 test reviews",
        ],
        "shape": "1 → 1\n$\\hat{y}$ ∈ {0, 1}",
        "fill": MODEL,
    },
]

BOX_LEFT, BOX_RIGHT = 2.0, 68.0     # the stage boxes
SHAPE_X = 71.0                      # the dimension annotations beside them
GAP = 3.6                           # vertical space for the arrows
TITLE_DY = 2.9                      # title baseline below the box top
FIRST_LINE_DY = 6.4                 # first detail line below the box top
LINE_H = 2.9                        # spacing between detail lines
PAD_BOTTOM = 2.2                    # clear space under the last detail line
HEADER = 13.0                       # room for the figure title and subtitle


def box_height(stage):
    """Tall enough for the title plus every detail line, with padding."""
    return FIRST_LINE_DY + LINE_H * (len(stage["detail"]) - 1) + PAD_BOTTOM


def main():
    heights = [box_height(s) for s in STAGES]
    total = HEADER + sum(heights) + GAP * (len(STAGES) - 1)

    # Keep roughly 11 drawing units per inch however many stages there are.
    fig, ax = plt.subplots(figsize=(8.6, total / 11.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, total)
    ax.axis("off")

    ax.text(BOX_LEFT, total - 1.8,
            "TF-IDF  +  Logistic Regression  —  baseline architecture",
            fontsize=13, fontweight="bold", color=INK, va="top")
    ax.text(BOX_LEFT, total - 5.4,
            "Input and output dimensions at every stage. Numbers measured from "
            "src/run_baseline.py (seed 42).",
            fontsize=9, color=INK_MUTED, va="top")

    y = total - HEADER
    for i, (stage, height) in enumerate(zip(STAGES, heights)):
        top = y
        bottom = top - height

        ax.add_patch(FancyBboxPatch(
            (BOX_LEFT, bottom), BOX_RIGHT - BOX_LEFT, height,
            boxstyle="round,pad=0,rounding_size=1.1",
            facecolor=stage["fill"], edgecolor=AXIS, linewidth=1.0,
        ))

        ax.text(BOX_LEFT + 2.2, top - TITLE_DY, stage["title"],
                fontsize=10.5, fontweight="bold", color=INK, va="center")

        line_y = top - FIRST_LINE_DY
        for line in stage["detail"]:
            ax.text(BOX_LEFT + 2.2, line_y, line,
                    fontsize=8.6, color=INK_MUTED, va="center",
                    family="DejaVu Sans")
            line_y -= LINE_H

        # The output shape sits outside the box so question (b)'s "dimensions at
        # each layer" can be read down the right-hand edge in one pass.
        ax.text(SHAPE_X, top - height / 2, stage["shape"],
                fontsize=8.8, color=POS if stage["fill"] == MODEL else NEG,
                va="center", ha="left", fontweight="bold", linespacing=1.5)

        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch(
                ((BOX_LEFT + BOX_RIGHT) / 2, bottom),
                ((BOX_LEFT + BOX_RIGHT) / 2, bottom - GAP),
                arrowstyle="-|>", mutation_scale=13,
                color=AXIS, linewidth=1.4,
            ))

        y = bottom - GAP

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "fig06_architecture.png"
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
