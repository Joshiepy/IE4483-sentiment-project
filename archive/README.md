# archive/

Old work kept for the record. **Nothing in this folder is used in the final report or
`results/submission.csv`.** The final model is `notebooks/05_distilbert_tuning.ipynb`.

| File | What it is | Why it is archived |
|---|---|---|
| `03_distilbert_15split.ipynb` | First DistilBERT run (Ryan). 0.9487 acc / 0.8951 macro F1 | Used a **15%** validation split (`test_size=0.15`), so it cannot be compared with the other models, which all use the shared 20% split. Its CSV column was also named `sentiments`, not `predictions`. |
| `submission_distilbert_15split.csv` | Test predictions from the notebook above (1,598 positive) | Same run; superseded |
| `colab_bert_training.ipynb` | Early Colab DistilBERT template | Never executed; no saved outputs |
