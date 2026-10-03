# data/ — provided by the course, do not edit

| File | What it is |
|------|------------|
| `train.json` | 7,401 labelled reviews. Fields: `reviews` (raw text), `sentiments` (0 = negative, 1 = positive) |
| `test.json` | 1,851 unlabelled reviews. Field: `reviews`. These are what we predict. |
| `submission_example.csv` | The course's example submission — shows the required column format |

Source: Blitzer, Dredze & Pereira, *Biographies, Bollywood, Boom-boxes and Blenders:
Domain Adaptation for Sentiment Classification*, ACL 2007.

**Note:** `test.json` has no labels, so we cannot measure test accuracy ourselves.
That's why the code holds out 20% of `train.json` as a validation set.
