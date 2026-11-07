# results/

Everything here is **produced by code** — never edit these files by hand.
Re-run the script instead.

| File | Produced by |
|------|-------------|
| `submission.csv` | `python src/run_baseline.py` |
| `figures/fig01` - `fig05` | `notebooks/01_explore_data.ipynb` |
| `figures/fig06_architecture.png` | `python src/make_architecture_figure.py` |

`figures/` holds the 200 dpi PNGs for the report:

| Figure | Shows | Report question |
|---|---|---|
| `fig01_class_balance.png` | 85/15 positive-negative split | (a), (d) |
| `fig02_review_length.png` | length distribution + the `max_length=256` choice | (a), (b) |
| `fig03_length_by_class.png` | length is not a label leak | (a) |
| `fig04_top_words.png` | frequent words barely differ between classes | (a) |
| `fig05_distinctive_words.png` | log-odds ratio - the terms that do discriminate | (a), (e) |
| `fig06_architecture.png` | the TF-IDF + LogReg pipeline, dimensions at each stage | (b) |

`submission.csv` is the file submitted to NTULearn alongside the report.
Columns: `reviews`, `predictions` (0 = negative, 1 = positive).
