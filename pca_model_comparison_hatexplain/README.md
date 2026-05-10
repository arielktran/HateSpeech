# HateXplain BERT Embedding Experiment

This repo evaluates a pretrained HateXplain BERT classifier and compares it with a simpler PCA + logistic regression classifier trained on BERT CLS embeddings.

The overall goal is to test whether the representation learned by `Hate-speech-CNERG/bert-base-uncased-hatexplain` can support a lightweight downstream classifier that performs similarly to, or slightly better than, the model's original classification head.

## Project Structure

```text
.
+-- main.py                     # End-to-end experiment runner
+-- requirements.txt            # Python dependencies
+-- data/
|   +-- df_final.csv            # Cleaned HateXplain-style input data
+-- outputs/                    # Generated splits, metrics, predictions, and figures
+-- src/
    +-- bootstrap.py            # Bootstrap macro-F1 comparison
    +-- config.py               # Model name and label mapping
    +-- data.py                 # Data loading, cleaning, preprocessing, splitting
    +-- embedding_utils.py      # CLS embedding extraction
    +-- evaluate.py             # BERT classifier evaluation
    +-- model.py                # Hugging Face model/tokenizer loading
    +-- pca_model.py            # PCA + logistic regression model selection
    +-- visualization.py        # Plots and confusion matrices
```

## Experiment Pipeline

1. Load the pretrained HateXplain BERT model and tokenizer from Hugging Face.
2. Load the cleaned dataset from `data/df_final.csv`.
3. Keep examples with one of three majority labels: `hatespeech`, `normal`, or `offensive`.
4. Convert labels to numeric IDs:
   - `hatespeech`: `0`
   - `normal`: `1`
   - `offensive`: `2`
5. Create stratified train, validation, and test splits.
6. Evaluate the pretrained BERT classification head on the test split.
7. Extract final-layer CLS embeddings from BERT for all train, validation, and test examples.
8. Train PCA + logistic regression pipelines using several PCA dimensions.
9. Select the PCA dimension with the best validation macro F1.
10. Evaluate the selected PCA + logistic regression model on the test split.
11. Save predictions, metrics, plots, and bootstrap comparison outputs to `outputs/`.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The first run may need internet access to download the Hugging Face model:

```text
Hate-speech-CNERG/bert-base-uncased-hatexplain
```

## Running the Repo

Run the full experiment from the repo root:

```bash
python3 main.py
```

If your environment exposes Python as `python`, this is equivalent:

```bash
python main.py
```

The script creates `outputs/` if needed. Some expensive intermediate artifacts are cached:

- `outputs/bert_hatexplain_cleaned_results.csv`
- `outputs/X_train_cls.npy`
- `outputs/X_val_cls.npy`
- `outputs/X_test_cls.npy`
- `outputs/y_train.npy`
- `outputs/y_val.npy`
- `outputs/y_test.npy`

If these files already exist, `main.py` reuses them instead of recomputing BERT predictions and embeddings.

## Data

The expected input file is:

```text
data/df_final.csv
```

The code expects at least these columns:

- `post_text`: text to classify
- `majority_label`: one of `hatespeech`, `normal`, or `offensive`

Optional columns such as `post_id` and `post_tokens` are used when present.

In the current generated outputs, the split sizes are:

| Split | Examples |
|---|---:|
| Train | 15,383 |
| Validation | 1,923 |
| Test | 1,923 |

## Outputs

The experiment writes these key files:

| File | Purpose |
|---|---|
| `outputs/train_split.csv` | Training split |
| `outputs/val_split.csv` | Validation split |
| `outputs/test_split.csv` | Test split |
| `outputs/bert_hatexplain_cleaned_results.csv` | BERT head predictions on the test set |
| `outputs/pca_model_selection_results.csv` | Validation results for each PCA dimension |
| `outputs/pca_logreg_test_results.csv` | Test predictions from the selected PCA + logistic regression model |
| `outputs/pca_test_metrics.json` | Final PCA model metrics and classification report |
| `outputs/bootstrap_results.json` | Bootstrap macro-F1 comparison |
| `outputs/model_comparison.png` | Accuracy and macro-F1 comparison chart |
| `outputs/bert_confusion_matrix.png` | BERT confusion matrix |
| `outputs/pca_confusion_matrix.png` | PCA + logistic regression confusion matrix |
| `outputs/pca_scatter_test.png` | 2D PCA projection of test CLS embeddings |

## Results

The latest run loaded the saved BERT evaluation results and saved CLS embeddings, then reran PCA + logistic regression model selection, visualization, and bootstrap comparison.

Run summary:

```text
Train size: 15383
Val size: 1923
Test size: 1923

Train embeddings: (15383, 768)
Val embeddings: (1923, 768)
Test embeddings: (1923, 768)
```

The run selected `PCA-50 + LogisticRegression` as the best validation model.

Validation model selection:

| Model | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| PCA-5 + LogisticRegression | 0.7369 | 0.7301 |
| PCA-10 + LogisticRegression | 0.7384 | 0.7321 |
| PCA-25 + LogisticRegression | 0.7363 | 0.7306 |
| PCA-50 + LogisticRegression | 0.7462 | 0.7408 |
| PCA-100 + LogisticRegression | 0.7389 | 0.7328 |
| PCA-200 + LogisticRegression | 0.7306 | 0.7249 |

Test set comparison:

| Model | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| BERT classification head | 0.7379 | 0.7339 |
| PCA-50 + LogisticRegression | 0.7488 | 0.7445 |

The PCA + logistic regression model performs slightly better on this saved test run:

```text
Macro F1 difference = PCA model - BERT head = 0.0108
95% bootstrap CI = [-0.0008, 0.0229]
```

Because the confidence interval slightly crosses zero, this result suggests a small improvement, but the bootstrap comparison does not show a clearly separated positive interval.

Latest bootstrap output:

```text
bert_mean_macro_f1 = 0.733591864282461
pca_mean_macro_f1 = 0.7443740844997107
mean_difference = 0.010782220217249555
diff_ci_lower = -0.0008482776286596577
diff_ci_upper = 0.02286217440919197
```

## Reproducing Results

For a clean rerun, delete the cached files in `outputs/` and run:

```bash
python3 main.py
```

The code uses `random_state=42` for dataset splitting, PCA, logistic regression, and bootstrap sampling, so results should be reproducible when using the same data, package versions, and model weights.
