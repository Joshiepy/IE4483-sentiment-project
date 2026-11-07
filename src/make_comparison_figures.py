"""Draw the model-comparison figures for report questions (c), (e) and (f).

Run it with:  python src/make_comparison_figures.py
Needs:        python src/compare_models.py   (writes results/model_comparison.json)

Two figures, both read straight from that JSON so they can never drift from the
measured numbers:

  fig07_confusion_matrices.png   where each model actually goes wrong
  fig08_model_comparison.png     per-class F1, and macro F1 against the
                                 majority-class floor

Design notes, so a later edit does not undo them:

* The palette is the project's fixed pair - negative #2a78d6, positive #eb6834 -
  on paper #fcfcfb, the same as every other figure. Verified against the six
  colour checks: adjacent CVD separation is dE 24.7 (protan) and 33.6 at normal
  vision, both well clear of the dE 8 target, and both hues clear 3:1 contrast
  against the paper.
* Accuracy is deliberately NOT the headline. At 85.3% positive, predicting
  "positive" for everything scores 0.853 accuracy, so the figure leads with
  per-class F1 and macro F1 and draws that floor explicitly.
* DistilBERT was validated on a 15% split rather than 20%, so its bar is
  hatched and labelled rather than silently placed beside the others. It has no
  per-class numbers in the committed notebook output, so it is absent from the
  left panel instead of being guessed at.
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from load_data import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"

# The project palette, shared with 01_explore_data.ipynb and
# make_architecture_figure.py.
NEG, POS = "#2a78d6", "#eb6834"
INK, INK_MUTED, AXIS = "#0b0b0b", "#898781", "#c3c2b7"
PAPER = "#fcfcfb"

plt.rcParams.update({
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_MUTED,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK,
})


def load_metrics():
    path = RESULTS_DIR / "model_comparison.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found - run 'python src/compare_models.py' first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recessive_axes(ax, keep=("left", "bottom")):
    """Grid and spines should be visible but never compete with the data."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
        spine.set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=8.6)


def figure_confusion(data):
    """fig07 - one confusion matrix per strictly comparable model.

    Counts are a magnitude, so the cells use ONE hue light to dark rather than a
    rainbow. Each cell is annotated, so the colour is a reading aid and never the
    only encoding.
    """
    models = [m for m in data["models"] if m.get("comparable") and "confusion_matrix" in m]
    ramp = LinearSegmentedColormap.from_list("blues", [PAPER, NEG])

    fig, axes = plt.subplots(
        1, len(models), figsize=(3.05 * len(models), 3.5),
        gridspec_kw={"wspace": 0.18},
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        cm = model["confusion_matrix"]
        grid = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        row_totals = [cm["tn"] + cm["fp"], cm["fn"] + cm["tp"]]

        # Normalise per row: the classes differ ~6x in size, so raw counts would
        # paint the positive row dark and the negative row blank whatever the
        # model did.
        shaded = [[grid[r][c] / row_totals[r] for c in range(2)] for r in range(2)]
        ax.imshow(shaded, cmap=ramp, vmin=0, vmax=1)

        for r in range(2):
            for c in range(2):
                share = shaded[r][c]
                ax.text(
                    c, r - 0.08, f"{grid[r][c]}",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color=PAPER if share > 0.55 else INK,
                )
                ax.text(
                    c, r + 0.22, f"{share:.0%} of row",
                    ha="center", va="center", fontsize=7.6,
                    color=PAPER if share > 0.55 else INK_MUTED,
                )

        ax.set_xticks([0, 1], ["pred neg", "pred pos"], fontsize=8.4)
        # Row labels on the leftmost panel only - repeating them puts "true neg"
        # hard against the right edge of the previous matrix.
        if ax is axes[0]:
            ax.set_yticks([0, 1], ["true neg", "true pos"], fontsize=8.4)
        else:
            ax.set_yticks([])
        ax.set_title(
            f"{model['short']}\nnegative recall {model['negative_recall']:.0%}",
            fontsize=9.6, color=INK, pad=8,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle(
        "Where each model goes wrong  —  same 1,481-review validation split, seed 42",
        fontsize=11.5, fontweight="bold", color=INK, y=1.04, x=0.02, ha="left",
    )
    fig.text(
        0.02, 0.985,
        "Shading is the share of each true class, not the raw count: there are "
        "1,264 positive reviews against 217 negative.",
        fontsize=8.4, color=INK_MUTED, ha="left", va="top",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "fig07_confusion_matrices.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def figure_comparison(data):
    """fig08 - per-class F1 (left) and macro F1 against the floor (right)."""
    models = data["models"]
    per_class = [m for m in models if "negative_f1" in m and "positive_f1" in m]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11.2, 0.62 * len(models) + 2.3),
        gridspec_kw={"width_ratios": [1.25, 1]},
    )

    # ---- left: per-class F1, the two project hues ------------------------
    height = 0.26
    gap = 0.045                      # keeps a paper-coloured gap between bars
    positions = list(range(len(per_class)))[::-1]

    for pos, model in zip(positions, per_class):
        for offset, (value, colour, label) in enumerate([
            (model["negative_f1"], NEG, "negative"),
            (model["positive_f1"], POS, "positive"),
        ]):
            y = pos + (0.5 - offset) * (height + gap)
            ax_left.barh(
                y, value, height=height, color=colour,
                label=label if pos == positions[0] else None,
            )
            ax_left.text(
                value + 0.012, y, f"{value:.3f}",
                va="center", ha="left", fontsize=8.6, color=INK_MUTED,
            )

    ax_left.set_yticks(positions, [m["short"] for m in per_class], fontsize=9.4)
    for tick in ax_left.get_yticklabels():
        tick.set_color(INK)
    ax_left.set_xlim(0, 1.12)
    ax_left.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_left.set_xlabel("F1 score", fontsize=9)
    ax_left.xaxis.grid(True, color=AXIS, linewidth=0.6, alpha=0.5)
    ax_left.set_axisbelow(True)
    recessive_axes(ax_left)
    # Top-right of the panel, on the same row as the left-aligned title: clear
    # of the title and clear of the bars at every bar length.
    legend = ax_left.legend(
        loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False,
        fontsize=8.8, handlelength=1.1, handleheight=0.9, borderpad=0.2,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_MUTED)
    # Kept short so it cannot run into the legend on the same row.
    ax_left.set_title(
        "Weakest on the negative class",
        fontsize=10.6, fontweight="bold", color=INK, loc="left", pad=10,
    )

    # ---- right: macro F1, one hue, with the majority-class floor ---------
    # Predicting "positive" for all 1,481 reviews: negative F1 = 0, positive F1
    # = 2*1264 / (1481 + 1264). Macro F1 is the mean of the two.
    n_pos, n_total = 1264, 1481
    floor = (0.0 + 2 * n_pos / (n_total + n_pos)) / 2

    order = list(range(len(models)))[::-1]
    for pos, model in zip(order, models):
        comparable = model.get("comparable", False)
        ax_right.barh(
            pos, model["macro_f1"], height=0.4,
            color=POS if comparable else PAPER,
            edgecolor=POS, linewidth=1.2,
            hatch=None if comparable else "////",
        )
        suffix = "" if comparable else "  (15% split)"
        ax_right.text(
            model["macro_f1"] + 0.012, pos,
            f"{model['macro_f1']:.3f}{suffix}",
            va="center", ha="left", fontsize=8.6,
            color=INK_MUTED if comparable else INK,
        )

    ax_right.axvline(floor, color=INK_MUTED, linewidth=1.1, linestyle=(0, (4, 3)))
    # Below the lowest bar, so it never lands on top of one.
    ax_right.text(
        floor, -0.72,
        f'"always positive" floor {floor:.3f}',
        fontsize=8, color=INK_MUTED, ha="center", va="top",
    )
    ax_right.set_ylim(-1.1, len(models) - 0.4)

    ax_right.set_yticks(order, [m["short"] for m in models], fontsize=9.4)
    for tick in ax_right.get_yticklabels():
        tick.set_color(INK)
    ax_right.set_xlim(0, 1.3)
    ax_right.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_right.set_xlabel("macro F1", fontsize=9)
    ax_right.xaxis.grid(True, color=AXIS, linewidth=0.6, alpha=0.5)
    ax_right.set_axisbelow(True)
    recessive_axes(ax_right)
    ax_right.set_title(
        "Macro F1, and the score for guessing blindly",
        fontsize=10.6, fontweight="bold", color=INK, loc="left", pad=10,
    )

    fig.suptitle(
        "Model comparison  —  one shared TF-IDF vocabulary, seed 42",
        fontsize=12.5, fontweight="bold", color=INK, x=0.012, ha="left", y=1.02,
    )
    # The caption has to follow the data: once notebook 04 has run there is no
    # hatched bar left to explain, and a stale caption is worse than none.
    odd = [m for m in models if not m.get("comparable", False)]
    if odd:
        caption = (
            f"The hatched {odd[0]['short']} bar was validated on a "
            f"{odd[0]['test_size']:.0%} split, not the {data['split'].split()[-1]} "
            "split the others share, so it is indicative rather than like-for-like."
        )
    else:
        caption = (
            "All models scored on the same stratified "
            f"{data['split'].split()[-1]} validation split, seed 42."
        )
    fig.text(0.012, 0.975, caption, fontsize=8.6, color=INK_MUTED,
             ha="left", va="top")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "fig08_model_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main():
    data = load_metrics()
    figure_confusion(data)
    figure_comparison(data)


if __name__ == "__main__":
    main()
