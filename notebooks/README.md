# notebooks/

| Notebook | Owner | What it is | Where to run it |
|----------|-------|------------|-----------------|
| `01_explore_data.ipynb` | Joshua | Phase 1 - class balance, review lengths, vocabulary. Writes `results/figures/fig01` - `fig05` | anywhere (CPU, ~15 s) |
| `02_svm_mlp.ipynb` | Jayce | **Section 2b shows the cleaned data** (raw vs cleaned reviews, saved to `results/cleaned_train.csv` / `cleaned_val.csv`) and proves it is exactly what the models use; then Linear SVM and a two-hidden-layer MLP on those TF-IDF features. Writes `results/part_b/` | anywhere (CPU, ~10 s) |
| `04_distilbert_20split.ipynb` | Ryan / Joshua | **Superseded by `05`.** The same DistilBERT on the shared 20% split, run once on a T4 (0.9446 / 0.8893). Kept as evidence of GPU-to-GPU variation | Google Colab, GPU |
| `05_distilbert_tuning.ipynb` | Joshua | **The final DistilBERT.** Tuning runs for question (c) (class weights, learning rate, threshold, seeds) and the submitted `results/submission.csv` (run `baseline`: lr 2e-5, seed 42, 0.9507 / 0.9001) | **Google Colab with a GPU** (L4, about 4 min per run) |
| `tfidf_features_example.ipynb` | course | Provided by the course - how to turn raw review text into TF-IDF features | anywhere (CPU) |

Older runs are in `archive/` (see `archive/README.md`), including the first DistilBERT
notebook, which used a 15% validation split and is not used in the report.

## Validation split

Every model in the report uses the same stratified 80/20 split, `random_state=42`:
5,920 training / 1,481 validation reviews.

| Notebook / script | Split | Seed |
|---|---|---|
| `src/run_baseline.py` | stratified **80/20** | 42 |
| `02_svm_mlp.ipynb` | stratified **80/20** | 42 |
| `04_distilbert_20split.ipynb`, `05_distilbert_tuning.ipynb` | stratified **80/20** | 42 (05 also 1, 2) |

## Running the Colab notebooks

1. [colab.research.google.com](https://colab.research.google.com) -> `File` -> `Upload notebook`
2. **`Runtime` -> `Change runtime type` -> GPU** - without this it takes about an
   hour per epoch instead of a minute
3. Upload `data/train.json` and `data/test.json` to the Colab Files panel
4. Run the cells in order; the first cell confirms the GPU and the data files

## A warning about notebooks in git

Notebooks save their *output* inside the file. If two people run the same notebook and
both commit, git sees the whole file as changed and the merge conflict is unreadable JSON.
Agree who owns each notebook (the Owner column above), or clear outputs before committing.

`02_svm_mlp.ipynb` keeps its outputs **on purpose** - the printed metrics are the numbers
quoted in the report. Re-running it with a newer scikit-learn gives slightly different SVM/MLP
numbers (see the library-version note in the top-level `README.md`), so do not re-run and
re-commit it without telling its owner.

## Paths

`01_explore_data.ipynb` and `02_svm_mlp.ipynb` search upwards for `data/`, so they work from
either the repo root or `notebooks/`. The Colab notebooks read `train.json` and `test.json`
from the Colab working directory (upload them first).
