# notebooks/

Use notebooks for **exploring and experimenting**. Once something works and you want
to re-run it reliably, move it into `src/` as a script.

| Notebook | Owner | What it is | Where to run it |
|----------|-------|------------|-----------------|
| `01_explore_data.ipynb` | Joshua | Phase 1 - class balance, review lengths, vocabulary. Writes `results/figures/fig01` - `fig05` | anywhere (CPU, ~15 s) |
| `02_svm_mlp.ipynb` | Jayce | Linear SVM and a two-hidden-layer MLP on the same TF-IDF features. Writes `results/part_b/` | anywhere (CPU, ~10 s) |
| `03_distilbert.ipynb` | Ryan | Fine-tunes DistilBERT end to end and writes `results/submission_distilbert.csv` | CPU works but is slow; a T4 GPU is much faster |
| `04_distilbert_20split.ipynb` | Ryan | **Not run yet.** The same DistilBERT at `test_size=0.2`, so it lands on the split the other three share. Writes `results/distilbert_metrics.json` | **Google Colab with a T4 GPU** |
| `tfidf_features_example.ipynb` | course | Provided by the course - how to turn raw review text into TF-IDF features | anywhere (CPU) |
| `colab_bert_training.ipynb` | Joshua | **Superseded by `03_distilbert.ipynb`.** An empty Colab scaffold written before Ryan trained the real model; kept only for the Colab setup cells | Google Colab, T4 GPU |

## The validation splits do not match

This matters for any table that puts these models side by side:

| Notebook / script | Split | Seed |
|---|---|---|
| `src/run_baseline.py` | stratified **80/20** | 42 |
| `02_svm_mlp.ipynb` | stratified **80/20** | 42 |
| `03_distilbert.ipynb` | stratified **85/15** | 42 |

The baseline, the SVM and the MLP all score on the *same* 1,481-review validation
split, so those three are directly comparable. DistilBERT used a 15% split, so its
0.9487 is measured on different reviews and is **not** strictly comparable to the
other three. `notebooks/04_distilbert_20split.ipynb` exists to close that gap.

## Running the Colab notebooks

1. [colab.research.google.com](https://colab.research.google.com) -> `File` -> `Upload notebook`
2. **`Runtime` -> `Change runtime type` -> T4 GPU** - without this it takes about an
   hour per epoch instead of two minutes
3. Run the cells in order; the first cell confirms the GPU is active

## A warning about notebooks in git

Notebooks save their *output* inside the file. If two people run the same notebook and
both commit, git sees the whole file as changed and the merge conflict is unreadable JSON.

To avoid it: agree who owns each notebook, or clear outputs before committing
(`Kernel` -> `Restart & Clear Output`). The Owner column above is that agreement.

`02_svm_mlp.ipynb` and `03_distilbert.ipynb` keep their outputs **on purpose** - the
printed metrics are the evidence for report questions (c) and (f). Do not re-run and
re-commit them without telling their owner.

## Paths

`03_distilbert.ipynb` reads `train.json` and `test.json` from the **current working
directory**, not from `data/`. Launch Jupyter from the repo root and the paths will not
resolve - either copy the two files next to the notebook or change the paths to
`../data/`. Left as Ryan ran it so the committed outputs stay reproducible.

`01_explore_data.ipynb` and `02_svm_mlp.ipynb` both search upwards for `data/`, so they
work from either the repo root or `notebooks/`.
