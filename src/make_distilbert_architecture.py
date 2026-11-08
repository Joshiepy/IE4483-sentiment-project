"""Draw the DistilBERT architecture figure for report question (b).

Writes results/figures/fig09_distilbert_architecture.png at 200 dpi.

Every size shown is the configuration used in notebooks/04_distilbert_20split.ipynb:
distilbert-base-uncased (6 layers, 12 heads, hidden size 768, feed-forward 3072),
max_length=256, and DistilBertForSequenceClassification's head
(pre-classifier 768 -> 768 + ReLU, dropout 0.2, classifier 768 -> 2).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "fig09_distilbert_architecture.png"

PAPER, INK, MUTED = "#fcfcfb", "#0b0b0b", "#555555"
BOX_FILL, BOX_EDGE = "#ebf2fc", "#3a5f9c"
HEAD_FILL, HEAD_EDGE = "#fdeee6", "#c4552a"

# (title, detail, output shape, is_head)
ROW1 = [
    ("Review text", "raw string", "1 review", False),
    ("WordPiece tokeniser", "lowercase, add [CLS] / [SEP],\ntruncate or pad", "256 token ids\n+ attention mask", False),
    ("Embeddings", "token + position", "256 x 768", False),
    ("Transformer encoder", "6 layers, 12 heads,\nfeed-forward 3072", "256 x 768", False),
]
ROW2 = [
    ("[CLS] vector", "first position of\nthe last layer", "768", True),
    ("Pre-classifier", "Linear 768 -> 768\n+ ReLU", "768", True),
    ("Dropout", "p = 0.2", "768", True),
    ("Classifier", "Linear 768 -> 2", "2 logits", True),
    ("Prediction", "softmax + argmax;\ncross-entropy in training", "0 / 1", True),
]


def draw_row(ax, items, y, x0, width, gap):
    centres = []
    for i, (title, detail, shape, head) in enumerate(items):
        x = x0 + i * (width + gap)
        ax.add_patch(FancyBboxPatch(
            (x, y), width, 1.5, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=HEAD_FILL if head else BOX_FILL,
            edgecolor=HEAD_EDGE if head else BOX_EDGE, linewidth=1.2))
        ax.text(x + width / 2, y + 1.15, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=INK)
        ax.text(x + width / 2, y + 0.62, detail, ha="center", va="center",
                fontsize=7.8, color=MUTED, linespacing=1.3)
        ax.text(x + width / 2, y - 0.25, shape, ha="center", va="top",
                fontsize=8, color=INK, style="italic", linespacing=1.2)
        if i:
            ax.annotate("", xy=(x - 0.03, y + 0.75), xytext=(x - gap + 0.03, y + 0.75),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
        centres.append((x, x + width))
    return centres


fig, ax = plt.subplots(figsize=(11, 5.2))
fig.patch.set_facecolor(PAPER)
ax.set_facecolor(PAPER)
ax.set_xlim(0, 11)
ax.set_ylim(-0.6, 6.4)
ax.axis("off")

ax.text(0.1, 6.15, "DistilBERT fine-tuned for sentiment (distilbert-base-uncased, 66,955,010 parameters)",
        fontsize=12, fontweight="bold", color=INK, va="center")
ax.text(0.1, 5.75, "Italic text under each block is the shape of its output for one review. "
        "Blue: pretrained backbone. Orange: classification head.",
        fontsize=8.5, color=MUTED, va="center")

r1 = draw_row(ax, ROW1, 3.55, 0.35, 2.25, 0.5)
r2 = draw_row(ax, ROW2, 0.55, 0.2, 1.85, 0.32)

# Connector from the encoder (end of row 1) down to the [CLS] vector (start of row 2).
x_enc = (r1[-1][0] + r1[-1][1]) / 2
x_cls = (r2[0][0] + r2[0][1]) / 2
ax.plot([x_enc, x_enc], [3.55 - 0.75, 2.6], color=INK, lw=1.2)
ax.plot([x_enc, x_cls], [2.6, 2.6], color=INK, lw=1.2)
ax.annotate("", xy=(x_cls, 2.08), xytext=(x_cls, 2.6),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=PAPER)
print("wrote", OUT)
