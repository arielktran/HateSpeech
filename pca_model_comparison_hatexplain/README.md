# HateXplain BERT Embedding Experiment

This repo evaluates a pretrained HateXplain BERT classifier and compares it with a simpler PCA + logistic regression classifier trained on BERT CLS embeddings. It also reports target-group performance to show whether aggregate model gains hold across identity groups.

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
    +-- group_analysis.py       # Target-group metric computation
    +-- model.py                # Hugging Face model/tokenizer loading
    +-- pca_model.py            # PCA + logistic regression model selection
    +-- visualization.py        # Plots and confusion matrices
```

## Experiment Pipelines

The code emphasizes two analyses: an overall model comparison and a target-group comparison.

### 1. Model Comparison

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

The current model-selection sweep tests PCA dimensions `5`, `10`, `25`, `50`, `100`, and `200`. Each candidate uses `StandardScaler`, `PCA`, and balanced `LogisticRegression`; selection is based on validation macro F1.

### 2. Target-Group Analysis

After the aggregate model comparison, the pipeline evaluates BERT and PCA + logistic regression on examples associated with specific target groups:

1. Parse the test split's `majority` target-group column.
2. Normalize aliases such as `Homosexual` and `LGBTQ` to `Gay`, and `Refugee/Immigrant` to `Refugee`.
3. Add one binary indicator column per supported target group.
4. Keep target groups with at least `30` test examples.
5. Compute accuracy and macro F1 for both models within each group.
6. Save `outputs/group_metrics.csv` and `outputs/group_macro_f1_comparison.png`.

The supported target groups are `African`, `Islam`, `Jewish`, `Gay`, `Women`, `Refugee`, `Arab`, `Caucasian`, `Hispanic`, and `Asian`.

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

For target-group analysis, the cleaned CSV should also include:

- `majority`: stringified list or list of target groups for each post

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
| `outputs/bootstrap_differences.npy` | Bootstrap macro-F1 difference samples |
| `outputs/group_metrics.csv` | Per-target-group accuracy and macro-F1 results |
| `outputs/model_comparison.png` | Accuracy and macro-F1 comparison chart |
| `outputs/bert_confusion_matrix.png` | BERT confusion matrix |
| `outputs/pca_confusion_matrix.png` | PCA + logistic regression confusion matrix |
| `outputs/pca_scatter_test.png` | 2D PCA projection of test CLS embeddings |
| `outputs/group_macro_f1_comparison.png` | Target-group macro-F1 comparison chart |

## Results

Read the results in two parts: the aggregate model comparison and the target-group analysis.

### Overall Model Comparison

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

The central model-selection result is that `PCA-50 + LogisticRegression` was selected as the best validation model. It had the highest validation macro F1 in the sweep, with `0.7408`, and also the highest validation accuracy, with `0.7462`.

Validation model selection:

| Model | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| PCA-5 + LogisticRegression | 0.7369 | 0.7301 |
| PCA-10 + LogisticRegression | 0.7384 | 0.7321 |
| PCA-25 + LogisticRegression | 0.7363 | 0.7306 |
| PCA-50 + LogisticRegression | 0.7462 | 0.7408 |
| PCA-100 + LogisticRegression | 0.7389 | 0.7328 |
| PCA-200 + LogisticRegression | 0.7306 | 0.7249 |

Test set comparison for the selected model:

| Model | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| BERT classification head | 0.7379 | 0.7339 |
| PCA-50 + LogisticRegression | 0.7488 | 0.7445 |

On the test split, the selected PCA-50 + logistic regression model improves over the pretrained BERT classification head by about `+0.0106` macro F1 and `+0.0109` accuracy.

The bootstrap comparison estimates a similar mean macro-F1 difference:

```text
Macro F1 difference = PCA model - BERT head = 0.0108
95% bootstrap CI = [-0.0008, 0.0229]
```

Because the confidence interval slightly crosses zero, this result suggests a small aggregate improvement, but the bootstrap comparison does not show a clearly separated positive interval.

Latest bootstrap output:

```text
bert_mean_macro_f1 = 0.733591864282461
pca_mean_macro_f1 = 0.7443740844997107
mean_difference = 0.010782220217249555
diff_ci_lower = -0.0008482776286596577
diff_ci_upper = 0.02286217440919197
```

### Target-Group Analysis Results

The latest run also produced target-group metrics and saved the group-level plot to `outputs/group_macro_f1_comparison.png`.

| Group | n | BERT Accuracy | BERT Macro F1 | PCA Accuracy | PCA Macro F1 | PCA - BERT Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| African | 306 | 0.8007 | 0.6500 | 0.8007 | 0.6510 | 0.0010 |
| Islam | 233 | 0.6910 | 0.6539 | 0.7339 | 0.6954 | 0.0415 |
| Jewish | 201 | 0.7910 | 0.5423 | 0.8209 | 0.5803 | 0.0380 |
| Gay | 192 | 0.6406 | 0.6407 | 0.6667 | 0.6610 | 0.0203 |
| Women | 163 | 0.6933 | 0.6737 | 0.7362 | 0.7086 | 0.0348 |
| Refugee | 76 | 0.5921 | 0.5865 | 0.6184 | 0.6331 | 0.0466 |
| Arab | 84 | 0.8333 | 0.6447 | 0.8095 | 0.5696 | -0.0751 |
| Caucasian | 53 | 0.5472 | 0.4756 | 0.5472 | 0.5191 | 0.0435 |
| Hispanic | 38 | 0.9211 | 0.9055 | 0.9211 | 0.9055 | 0.0000 |
| Asian | 42 | 0.6190 | 0.5970 | 0.6429 | 0.6242 | 0.0273 |

In this run, PCA + logistic regression improved group-level macro F1 for 8 of the 10 reported groups, matched BERT for Hispanic examples, and underperformed BERT for Arab examples. The largest positive macro-F1 differences were for Refugee (`+0.0466`), Caucasian (`+0.0435`), Islam (`+0.0415`), Jewish (`+0.0380`), and Women (`+0.0348`). The largest negative difference was for Arab examples (`-0.0751`).

These group results should be reported alongside aggregate test metrics because they show where the aggregate improvement is consistent and where it is not.

## Reproducing Results

For a clean rerun, delete the cached files in `outputs/` and run:

```bash
python3 main.py
```

The code uses `random_state=42` for dataset splitting, PCA, logistic regression, and bootstrap sampling, so results should be reproducible when using the same data, package versions, and model weights.
