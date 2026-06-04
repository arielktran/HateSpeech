# HateXplain BERT Embedding Experiment

This repository compares the original pretrained HateXplain BERT classifier with lightweight logistic-regression classifiers trained on BERT CLS embeddings. It evaluates:

- the original `Hate-speech-CNERG/bert-base-uncased-hatexplain` classification head,
- a logistic-regression baseline trained directly on CLS embeddings with no PCA,
- PCA + logistic-regression models over several PCA dimensions,
- target-group performance and significance tests across identity groups.

The main question is whether BERT's learned representation can support a simpler downstream classifier that performs similarly to, or better than, the original classification head, and whether that performance is consistent across target groups.

## Project Structure

```text
.
+-- main.py                     # End-to-end experiment runner
+-- requirements.txt            # Python dependencies
+-- data/
|   +-- df_final.csv            # Cleaned HateXplain-style input data
+-- outputs/                    # Generated splits, metrics, predictions, and figures
+-- src/
    +-- bootstrap.py            # Aggregate bootstrap macro-F1 comparison
    +-- config.py               # Model name and label mapping
    +-- data.py                 # Data loading, cleaning, preprocessing, splitting
    +-- embedding_utils.py      # CLS embedding extraction
    +-- evaluate.py             # BERT classifier evaluation
    +-- group_analysis.py       # Target-group metrics and significance tests
    +-- model.py                # Hugging Face model/tokenizer loading
    +-- pca_model.py            # Logistic baseline and PCA + logistic model selection
    +-- visualization.py        # Plots and confusion matrices
```

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

If the model is already cached locally, the script can run without downloading it. You may still see a non-fatal Hugging Face background warning about safetensors conversion if network access is unavailable.

## Running the Experiment

Run the full pipeline from the repo root:

```bash
python3 main.py
```

In the local project environment used during development, this command was:

```bash
./hatespeech/bin/python main.py
```

The script creates `outputs/` if needed. Expensive artifacts are cached and reused on later runs:

- `outputs/bert_hatexplain_train_results.csv`
- `outputs/bert_hatexplain_val_results.csv`
- `outputs/bert_hatexplain_cleaned_results.csv`
- `outputs/X_train_cls.npy`
- `outputs/X_val_cls.npy`
- `outputs/X_test_cls.npy`
- `outputs/y_train.npy`
- `outputs/y_val.npy`
- `outputs/y_test.npy`

The first full run can be slow because it evaluates BERT and extracts embeddings. Later runs are much faster if these files already exist.

## Data

The expected input file is:

```text
data/df_final.csv
```

Required columns:

- `post_text`: text to classify
- `majority_label`: one of `hatespeech`, `normal`, or `offensive`

Required for target-group analysis:

- `majority`: stringified list or Python list of target groups for each post

Optional columns such as `post_id` and `post_tokens` are preserved when present.

Current generated split sizes:

| Split | Examples |
|---|---:|
| Train | 15,383 |
| Validation | 1,923 |
| Test | 1,923 |

## Pipeline

The full experiment does the following:

1. Load the pretrained HateXplain BERT model and tokenizer.
2. Load and clean `data/df_final.csv`.
3. Keep examples with majority labels `hatespeech`, `normal`, or `offensive`.
4. Create stratified train, validation, and test splits.
5. Evaluate the original BERT classification head on train, validation, and test.
6. Extract final-layer CLS embeddings for train, validation, and test.
7. Train a logistic-regression baseline on CLS embeddings without PCA.
8. Train PCA + logistic-regression models for PCA dimensions `5`, `10`, `25`, `50`, `100`, and `200`.
9. Select the PCA model with the best validation macro F1.
10. Evaluate the selected PCA model on the test set.
11. Generate comparison tables, plots, bootstrap tests, and target-group analyses.

The logistic-regression models use `StandardScaler` and balanced `LogisticRegression`. PCA models add a PCA step before logistic regression.

## Key Outputs

| File | Purpose |
|---|---|
| `outputs/train_split.csv` | Training split |
| `outputs/val_split.csv` | Validation split |
| `outputs/test_split.csv` | Test split |
| `outputs/bert_hatexplain_train_results.csv` | BERT predictions on the train split |
| `outputs/bert_hatexplain_val_results.csv` | BERT predictions on the validation split |
| `outputs/bert_hatexplain_cleaned_results.csv` | BERT predictions on the test split |
| `outputs/baseline_validation_metrics.json` | BERT and no-PCA logistic baseline metrics |
| `outputs/validation_model_comparison.csv` | Train, validation, and test metrics for BERT, no-PCA logistic regression, and all PCA models |
| `outputs/pca_model_selection_results.csv` | No-PCA logistic baseline plus PCA model-selection results |
| `outputs/pca_performance_vs_k.csv` | PCA-only performance table across k |
| `outputs/pca_logreg_test_results.csv` | Test predictions from the selected PCA + logistic-regression model |
| `outputs/pca_test_metrics.json` | Selected PCA model test metrics and classification report |
| `outputs/bootstrap_results.json` | Aggregate paired bootstrap macro-F1 comparison |
| `outputs/bootstrap_differences.npy` | Aggregate bootstrap macro-F1 difference samples |
| `outputs/group_metrics.csv` | Per-target-group metrics plus bootstrap CIs for group-level PCA-minus-BERT macro-F1 deltas |
| `outputs/group_delta_pairwise_significance.csv` | Pairwise significance tests comparing PCA-minus-BERT macro-F1 deltas across target groups |
| `outputs/group_performance_heterogeneity_significance.csv` | Omnibus permutation tests for whether accuracy and macro F1 vary across target groups |

## Visual Outputs

| File | Purpose |
|---|---|
| `outputs/model_comparison.png` | Test accuracy and macro-F1 comparison chart |
| `outputs/bert_confusion_matrix.png` | BERT confusion matrix on the test split |
| `outputs/pca_confusion_matrix.png` | Selected PCA + logistic-regression confusion matrix on the test split |
| `outputs/pca_performance_vs_k.png` | Train/test performance versus PCA dimension |
| `outputs/pca_scatter_test.png` | PC1 vs PC2 test CLS embedding scatter, colored by label class |
| `outputs/pca_scatter_test_pc1_pc3.png` | PC1 vs PC3 label-class scatter |
| `outputs/pca_scatter_test_pc2_pc3.png` | PC2 vs PC3 label-class scatter |
| `outputs/pca_scatter_test_pc1_pc4.png` | PC1 vs PC4 label-class scatter |
| `outputs/pca_scatter_test_pc2_pc4.png` | PC2 vs PC4 label-class scatter |
| `outputs/pca_scatter_test_pc3_pc4.png` | PC3 vs PC4 label-class scatter |
| `outputs/pca_scatter_test_target_groups_pc1_pc2.png` | PC1 vs PC2 scatter colored by 10 target groups |
| `outputs/pca_scatter_test_target_groups_pc1_pc3.png` | PC1 vs PC3 scatter colored by 10 target groups |
| `outputs/pca_scatter_test_target_groups_pc2_pc3.png` | PC2 vs PC3 scatter colored by 10 target groups |
| `outputs/group_macro_f1_comparison.png` | Target-group macro-F1 comparison chart |

The directory `outputs/pca_target_group_scatters/` contains per-group PCA scatter plots for PC1 vs PC2, PC1 vs PC3, and PC2 vs PC3.

## Results

### Validation and Test Comparison

The central comparison table is `outputs/validation_model_comparison.csv`. It includes train, validation, and test accuracy and macro F1 for all baseline and PCA models.

Latest verified run:

| Model | PCA dim | Train Acc. | Train Macro F1 | Val Acc. | Val Macro F1 | Test Acc. | Test Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT-HateXplain baseline | none | 0.7233 | 0.7179 | 0.7301 | 0.7243 | 0.7379 | 0.7339 |
| LogisticRegression baseline (no PCA) | none | 0.7671 | 0.7623 | 0.7103 | 0.7028 | 0.7207 | 0.7153 |
| PCA-5 + LogisticRegression | 5 | 0.7213 | 0.7148 | 0.7369 | 0.7301 | 0.7379 | 0.7324 |
| PCA-10 + LogisticRegression | 10 | 0.7224 | 0.7164 | 0.7384 | 0.7321 | 0.7426 | 0.7377 |
| PCA-25 + LogisticRegression | 25 | 0.7287 | 0.7235 | 0.7363 | 0.7306 | 0.7389 | 0.7350 |
| PCA-50 + LogisticRegression | 50 | 0.7328 | 0.7273 | 0.7462 | 0.7408 | 0.7488 | 0.7445 |
| PCA-100 + LogisticRegression | 100 | 0.7362 | 0.7305 | 0.7389 | 0.7328 | 0.7389 | 0.7339 |
| PCA-200 + LogisticRegression | 200 | 0.7411 | 0.7357 | 0.7306 | 0.7249 | 0.7353 | 0.7300 |

The best validation model is `PCA-50 + LogisticRegression`, selected by validation macro F1.

On the test set, `PCA-50 + LogisticRegression` improves over the BERT classification head by about:

- `+0.0109` accuracy
- `+0.0106` macro F1

### Aggregate Bootstrap Comparison

The aggregate paired bootstrap compares test macro F1 for the BERT head and the selected PCA model on the same test examples.

Latest bootstrap result:

```text
Macro F1 difference = PCA model - BERT head = 0.0108
95% bootstrap CI = [-0.0008, 0.0229]
```

Because the confidence interval slightly crosses zero, this suggests a small aggregate improvement, but not a clearly separated positive interval under this bootstrap test.

### Target-Group Metrics

Target-group metrics are computed on the test split. Supported target groups are:

```text
African, Islam, Jewish, Gay, Women, Refugee, Arab, Caucasian, Hispanic, Asian
```

Aliases are normalized:

- `Homosexual` and `LGBTQ` -> `Gay`
- `Refugee/Immigrant` -> `Refugee`

Only groups with at least 30 test examples are included.

Latest target-group macro-F1 results:

| Group | n | BERT Macro F1 | PCA Macro F1 | PCA - BERT Macro F1 |
|---|---:|---:|---:|---:|
| African | 306 | 0.6500 | 0.6510 | 0.0010 |
| Islam | 233 | 0.6539 | 0.6954 | 0.0415 |
| Jewish | 201 | 0.5423 | 0.5803 | 0.0380 |
| Gay | 192 | 0.6407 | 0.6610 | 0.0203 |
| Women | 163 | 0.6737 | 0.7086 | 0.0348 |
| Refugee | 76 | 0.5865 | 0.6331 | 0.0466 |
| Arab | 84 | 0.6447 | 0.5696 | -0.0751 |
| Caucasian | 53 | 0.4756 | 0.5191 | 0.0435 |
| Hispanic | 38 | 0.9055 | 0.9055 | 0.0000 |
| Asian | 42 | 0.5970 | 0.6242 | 0.0273 |

In this run, PCA + logistic regression improved group-level macro F1 for 8 of 10 groups, matched BERT for Hispanic examples, and underperformed BERT for Arab examples.

## Target-Group Significance Tests

The repository includes two target-group significance analyses.

### Pairwise Delta Differences Across Groups

Output:

```text
outputs/group_delta_pairwise_significance.csv
```

This tests whether the PCA-minus-BERT macro-F1 delta differs across target groups. For example:

```text
delta_Islam - delta_African
```

where:

```text
delta_group = MacroF1(PCA + LogReg on group) - MacroF1(BERT Head on group)
```

The test uses a paired row-level bootstrap over the full test split. Rows are resampled with replacement, and each row keeps its true label, BERT prediction, PCA prediction, and full target-group membership vector. This preserves overlap among target groups, so examples belonging to multiple groups remain linked across bootstrap samples.

The output includes:

- `delta_difference_group_a_minus_group_b`
- `delta_difference_ci_lower`
- `delta_difference_ci_upper`
- `p_value_two_sided_bootstrap`
- `p_value_fdr_bh`
- `significant_fdr_0_05`
- `overlap_n`
- `overlap_fraction_of_smaller_group`

Latest verified result:

```text
45 pairwise contrasts tested
0 significant after Benjamini-Hochberg FDR correction at 0.05
```

### Omnibus Performance Heterogeneity Across Groups

Output:

```text
outputs/group_performance_heterogeneity_significance.csv
```

This tests whether model performance itself varies significantly across target groups. It is run separately for:

- BERT Head accuracy
- BERT Head macro F1
- PCA + LogReg accuracy
- PCA + LogReg macro F1

For each model and metric, the code computes two heterogeneity statistics across groups:

- `range`: maximum group performance minus minimum group performance
- `weighted_sd`: group-size-weighted standard deviation of group performance

The null distribution is generated by permuting full target-group membership vectors across test examples while keeping true labels and model predictions fixed. This preserves the observed target-group overlap patterns but breaks the association between group membership and model performance.

Latest verified result:

```text
All 8 heterogeneity tests were significant after FDR correction at 0.05.
```

For macro F1:

| Model | Statistic | Observed | Null Mean | p-value | FDR significant |
|---|---|---:|---:|---:|---|
| BERT Head | range | 0.4300 | 0.1514 | 0.000 | True |
| BERT Head | weighted_sd | 0.0684 | 0.0341 | 0.001 | True |
| PCA + LogReg | range | 0.3864 | 0.1470 | 0.000 | True |
| PCA + LogReg | weighted_sd | 0.0658 | 0.0332 | 0.001 | True |

Report-ready wording:

```text
To test whether performance differed significantly across target groups, we used
a permutation test over target-group membership patterns. True labels and model
predictions were held fixed, while each example's full target-group membership
vector was shuffled across test examples. This preserves the distribution of
target-group overlap while breaking the association between target group and
model performance. For each permutation, subgroup accuracy and macro F1 were
recomputed, and heterogeneity across groups was summarized using the range and
weighted standard deviation. The p-value is the fraction of permutations where
heterogeneity was at least as large as observed.
```

## Reproducing Results

For a normal rerun:

```bash
python3 main.py
```

For a clean rerun, delete cached artifacts in `outputs/` and run the same command. A clean rerun will regenerate BERT predictions and CLS embeddings, so it will take substantially longer.

The code uses `random_state=42` for splitting, PCA, logistic regression, bootstrap sampling, and permutation sampling. Results should be reproducible with the same input data, package versions, model weights, and cached Hugging Face model.
