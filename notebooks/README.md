# notebooks/

Use notebooks for **exploring and experimenting**. Once something works and you want
to re-run it reliably, move it into `src/` as a script.

| Notebook | What it is | Where to run it |
|----------|------------|-----------------|
| `01_explore_data.ipynb` | Phase 1 - class balance, review lengths, vocabulary. Writes the report figures to `results/figures/` | anywhere (CPU, ~15 s) |
| `tfidf_features_example.ipynb` | Provided by the course - how to turn raw review text into TF-IDF features | anywhere (CPU) |
| `colab_bert_training.ipynb` | Fine-tunes DistilBERT and writes `submission.csv` | **Google Colab with a T4 GPU** |
| `02_distilbert.ipynb` | **Ryan.** Fine-tunes DistilBERT end to end and writes `results/submission_distilbert.csv`. Val acc 0.9487, macro F1 0.8951 | CPU works but is slow; a T4 GPU is much faster |

Suggested notebooks to add as you go:
- `02_error_analysis.ipynb` - Phase 4, contrasting TF-IDF and BERT on the same reviews

## Running the Colab notebook

1. [colab.research.google.com](https://colab.research.google.com) -> `File` -> `Upload notebook`
2. **`Runtime` -> `Change runtime type` -> T4 GPU** - without this it takes about an
   hour per epoch instead of two minutes
3. Run the cells in order; the first cell confirms the GPU is active

## A warning about notebooks in git

Notebooks save their *output* inside the file. If two people run the same notebook and
both commit, git sees the whole file as changed and the merge conflict is unreadable JSON.

To avoid it: agree who owns each notebook, or clear outputs before committing
(`Kernel` -> `Restart & Clear Output`).

Owners so far: `02_distilbert.ipynb` is **Ryan's** - its outputs are committed on purpose,
because the printed per-epoch metrics are the evidence for report question (c). Do not
re-run and re-commit it without telling him.

`02_distilbert.ipynb` reads `train.json` and `test.json` from the **current working
directory**, not from `data/`. Launch Jupyter from the repo root and the paths will not
resolve - either copy the two files next to the notebook or change the paths to
`../data/`. Left as Ryan ran it so the committed outputs stay reproducible.
