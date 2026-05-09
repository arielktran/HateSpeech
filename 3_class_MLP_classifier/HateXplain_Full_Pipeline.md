# HateXplain: Full Pipeline — From `dataset.json` to BERT-HateXplain + MLP Extension

> Based on: *Mathew et al. (2021), "HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection," AAAI 2021*
> GitHub: https://github.com/hate-alert/HateXplain

---

## Table of Contents

1. [Repository & Data Structure](#1-repository--data-structure)
2. [Stage 1 — Raw Dataset (`dataset.json`)](#2-stage-1--raw-dataset-datasetjson)
3. [Stage 2 — Preprocessing](#3-stage-2--preprocessing)
4. [Stage 3 — Ground Truth Attention Construction](#4-stage-3--ground-truth-attention-construction)
5. [Stage 4 — BERT Tokenization & Input Formatting](#5-stage-4--bert-tokenization--input-formatting)
6. [Stage 5 — BERT-HateXplain Model Architecture](#6-stage-5--bert-hatexplain-model-architecture)
7. [Stage 6 — Dual-Loss Training (Classification + Attention Supervision)](#7-stage-6--dual-loss-training-classification--attention-supervision)
8. [Stage 7 — Inference & Attention Head Extraction](#8-stage-7--inference--attention-head-extraction)
9. [Stage 8 — Evaluation Metrics](#9-stage-8--evaluation-metrics)
10. [Mini-Project Extension: MLP Classifier on BERT Representations](#10-mini-project-extension-mlp-classifier-on-bert-representations)
11. [Full Code Implementation](#11-full-code-implementation)
12. [Architecture Diagram (Text)](#12-architecture-diagram-text)

---

## 1. Repository & Data Structure

```
HateXplain/
├── Data/
│   ├── dataset.json              ← Main annotated dataset (20,148 posts)
│   ├── post_id_divisions.json    ← Pre-defined train/val/test splits (8:1:1)
│   └── (glove.840B.300d.txt)     ← GloVe embeddings (non-BERT models only)
├── Models/
│   ├── bert_model.py             ← BERT + attention supervision
│   ├── birnn_model.py
│   └── cnn_gru_model.py
├── Preprocess/
│   ├── data_loader.py            ← Tokenization, batching, DataLoader
│   └── utils.py                  ← Ground truth attention computation
├── best_model_json/
│   └── bert_base_uncased.json    ← Best hyperparameter config
├── manual_training_inference.py  ← Main entry point
└── testing_*.py                  ← Evaluation scripts
```

---

## 2. Stage 1 — Raw Dataset (`dataset.json`)

### Format

Each entry in `dataset.json` is keyed by a unique `post_id` and has the following structure:

```json
{
  "24198545_gab": {
    "post_id": "24198545_gab",
    "annotators": [
      { "label": "hatespeech", "annotator_id": 4, "target": ["African"] },
      { "label": "hatespeech", "annotator_id": 3, "target": ["African"] },
      { "label": "offensive",  "annotator_id": 5, "target": ["African"] }
    ],
    "rationales": [
      [0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ],
    "post_tokens": ["and","this","is","why","i","end","up","with",
                    "nigger","trainee","doctors","who","can","not",
                    "speak","properly","lack","basic","knowledge","of",
                    "biology","it","truly","scary","if","the","public","only","knew"]
  }
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `post_id` | string | Unique ID encoding source (`_gab` or `_twitter`) |
| `annotators` | list[dict] | Up to 3 crowd-worker annotations |
| `annotators[].label` | string | `"hatespeech"` / `"offensive"` / `"normal"` |
| `annotators[].annotator_id` | int | Worker ID (for agreement computation) |
| `annotators[].target` | list[str] | Communities targeted (e.g., `["African", "Women"]`) |
| `rationales` | list[list[int]] | Per-annotator binary token mask (1 = important token) |
| `post_tokens` | list[str] | Pre-tokenized post (whitespace-split, emojis preserved) |

### Dataset Statistics

| Source | Hateful | Offensive | Normal | Undecided | Total |
|---|---|---|---|---|---|
| Twitter | 708 | 2,328 | 5,770 | 249 | 9,055 |
| Gab | 5,227 | 3,152 | 2,044 | 670 | 11,093 |
| **Total** | **5,935** | **5,480** | **7,814** | **919** | **20,148** |

> The 919 "undecided" posts (where all three annotators disagreed) are **excluded** from training and evaluation.

### `post_id_divisions.json`

```json
{
  "train": ["24198545_gab", "tweet_id_2", ...],
  "val":   ["tweet_id_x", ...],
  "test":  ["gab_id_y", ...]
}
```

Splits are **stratified** — class proportions are preserved across train/val/test. Ratio: **8:1:1**.

---

## 3. Stage 2 — Preprocessing

### 3.1 Label Encoding

```
"hatespeech"  →  0
"offensive"   →  1  (sometimes called "normal" in Hugging Face version)
"normal"      →  2
```

Majority voting across the 3 annotators determines the final label. In case of a tie (all three different → undecided), the post is dropped.

```python
from collections import Counter

def get_majority_label(annotators):
    labels = [a['label'] for a in annotators]
    counts = Counter(labels)
    most_common = counts.most_common(1)[0]
    if most_common[1] < 2:        # no majority
        return None               # undecided → skip
    return most_common[0]

LABEL_MAP = {"hatespeech": 0, "offensive": 1, "normal": 2}
```

### 3.2 Target Community Encoding

Target communities mentioned by at least 2/3 annotators are kept (majority vote). Only communities present in ≥100 posts are included:

```
African, Islam, Jewish, Gay, Women, Refugee, Arab, Caucasian, Hispanic, Asian
```

### 3.3 Text Cleaning

The tokens in `post_tokens` are already whitespace-tokenized. Key preprocessing steps:
- Replace `@username` mentions → `<user>` token (done during collection)
- Preserve emojis (carry semantic signal for hate/offensive classification)
- Remove posts with URLs, images, or video references (done during collection)
- No further lowercasing for BERT (uses `bert-base-uncased`, so lowercase internally)

### 3.4 Train/Val/Test Split

```python
import json

dataset    = json.load(open("Data/dataset.json"))
divisions  = json.load(open("Data/post_id_divisions.json"))

train_ids  = divisions["train"]
val_ids    = divisions["val"]
test_ids   = divisions["test"]
```

---

## 4. Stage 3 — Ground Truth Attention Construction

This is the core innovation that enables attention supervision in BERT-HateXplain.

### 4.1 From Binary Masks to Soft Attention

Each post has 2–3 annotator rationale masks (binary vectors over tokens). These are averaged and then converted to a soft probability distribution.

```python
import numpy as np
from scipy.special import softmax

def compute_ground_truth_attention(rationales, label, sentence_length, tau=1.0):
    """
    rationales:      list of binary lists, one per annotator
    label:           int (0=hate, 1=offensive, 2=normal)
    sentence_length: number of tokens
    tau:             temperature for softmax sharpening
    """
    if label == 2:
        # Normal label → uniform distribution (no rationale)
        return np.ones(sentence_length) / sentence_length

    if len(rationales) == 0:
        return np.ones(sentence_length) / sentence_length

    # Average binary masks across annotators
    avg_attention = np.mean(rationales, axis=0)   # shape: (sentence_length,)

    # Apply temperature-scaled softmax to concentrate mass on rationale tokens
    # tau is tuned on the validation set (paper uses tau > 1 to sharpen)
    scaled = avg_attention / tau
    soft_attention = softmax(scaled)              # shape: (sentence_length,)

    return soft_attention
```

**Why temperature scaling?** Without it, the difference between rationale tokens (e.g., 0.6 average) and non-rationale tokens (e.g., 0.1 average) is small. Temperature sharpening makes the distribution more concentrated on the important tokens, giving the model a clearer supervision signal.

### 4.2 BERT Token Alignment

BERT's WordPiece tokenizer splits words into subword tokens (e.g., `"trainee"` → `["train", "##ee"]`). The ground truth attention is defined at the **word level** (from `post_tokens`), so it must be aligned to subword tokens.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def align_attention_to_subwords(word_tokens, word_attention):
    """
    word_tokens:   list of original whitespace-split words
    word_attention: soft attention vector (one value per word)
    Returns: subword_attention aligned to BERT tokenization
    """
    subword_attention = []

    for word, attn_val in zip(word_tokens, word_attention):
        subword_pieces = tokenizer.tokenize(word)
        # Distribute the word's attention equally across its subword pieces
        per_piece = attn_val / len(subword_pieces)
        subword_attention.extend([per_piece] * len(subword_pieces))

    return subword_attention   # length = number of subword tokens
```

After alignment:
- Prepend `0.0` for `[CLS]` token
- Append `0.0` for `[SEP]` token
- Re-normalize via softmax so the full sequence sums to 1

---

## 5. Stage 4 — BERT Tokenization & Input Formatting

### 5.1 Tokenization

```python
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 128   # Paper sets max token length to 128

def encode_post(word_tokens, word_attention_gt, max_len=128):
    """
    Returns input_ids, attention_mask, token_type_ids, gt_attention
    all padded/truncated to max_len.
    """
    # Join words back to string → let BERT tokenizer handle subwords
    text = " ".join(word_tokens)

    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids      = encoding["input_ids"]       # (1, max_len)
    attention_mask = encoding["attention_mask"]   # (1, max_len)  — 1=real, 0=pad
    token_type_ids = encoding["token_type_ids"]   # (1, max_len)  — all 0s for single-seq

    # Align ground truth attention to subword tokens, pad to max_len
    aligned_gt = align_attention_to_subwords(word_tokens, word_attention_gt)
    # Prepend CLS (0.0), append SEP (0.0), re-softmax
    aligned_gt = [0.0] + aligned_gt + [0.0]
    # Truncate or pad with zeros
    aligned_gt = aligned_gt[:max_len]
    aligned_gt += [0.0] * (max_len - len(aligned_gt))
    # Re-normalize (ignore CLS/SEP/PAD positions)
    aligned_gt = np.array(aligned_gt, dtype=np.float32)
    # (Optional: re-softmax over non-pad positions only)

    return input_ids, attention_mask, token_type_ids, torch.tensor(aligned_gt)
```

### 5.2 DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class HateXplainDataset(Dataset):
    def __init__(self, post_ids, dataset, tokenizer, max_len=128, tau=1.0):
        self.samples = []

        for pid in post_ids:
            entry = dataset[pid]
            label_str = get_majority_label(entry["annotators"])
            if label_str is None:
                continue   # skip undecided

            label     = LABEL_MAP[label_str]
            tokens    = entry["post_tokens"]
            rationales = entry["rationales"]

            gt_attn = compute_ground_truth_attention(
                rationales, label, len(tokens), tau=tau
            )
            input_ids, attn_mask, type_ids, gt_attn_tensor = encode_post(
                tokens, gt_attn, max_len
            )
            self.samples.append({
                "input_ids":      input_ids.squeeze(0),
                "attention_mask": attn_mask.squeeze(0),
                "token_type_ids": type_ids.squeeze(0),
                "gt_attention":   gt_attn_tensor,
                "label":          torch.tensor(label, dtype=torch.long),
                "post_id":        pid
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Usage
train_dataset = HateXplainDataset(train_ids, dataset, tokenizer, max_len=128)
train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

---

## 6. Stage 5 — BERT-HateXplain Model Architecture

### 6.1 Base BERT

The paper uses `bert-base-uncased`:
- 12 transformer encoder layers
- 12 attention heads per layer
- Hidden size: 768
- 110M parameters

### 6.2 Architecture Overview

```
Input: [CLS] token_1 token_2 ... token_N [SEP] [PAD] ... [PAD]
        ↓
   BERT Encoder (12 layers, each with 12 attention heads)
        ↓
  Last layer hidden states: (batch, max_len, 768)
  Last layer attention weights: (batch, 12_heads, max_len, max_len)
        ↓
   CLS token representation: hidden_states[:, 0, :]  →  (batch, 768)
        ↓
   Dropout (p=0.1, 0.2, or 0.5 — tuned on val set)
        ↓
   Linear(768 → 3)   →   logits
        ↓
   Softmax → predicted class probabilities (hate / offensive / normal)
```

### 6.3 Attention Supervision Head

For the `-HateXplain` variant, x out of 12 attention heads in the **last encoder layer** are designated as **supervised heads** (x ∈ {1, 6, 12}, tuned as hyperparameter).

For each supervised head h, the attention weights from the `[CLS]` token to all other tokens are extracted:

```
attn_weights[h][:, 0, :]   →   shape: (batch, max_len)
```

This is the model's learned importance distribution over input tokens from the perspective of `[CLS]`.

### 6.4 PyTorch Model Code

```python
import torch
import torch.nn as nn
from transformers import BertModel

class BertHateXplain(nn.Module):
    def __init__(self, num_classes=3, dropout=0.1, num_supervised_heads=6,
                 attention_lambda=100.0):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            output_attentions=True   # ← required to get attention weights back
        )

        self.dropout       = nn.Dropout(dropout)
        self.classifier    = nn.Linear(768, num_classes)
        self.num_sup_heads = num_supervised_heads
        self.attn_lambda   = attention_lambda

    def forward(self, input_ids, attention_mask, token_type_ids,
                gt_attention=None):
        """
        gt_attention: (batch, max_len) — ground truth soft attention
                      None during pure inference
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # CLS representation
        cls_output  = outputs.last_hidden_state[:, 0, :]   # (batch, 768)
        cls_dropped = self.dropout(cls_output)
        logits      = self.classifier(cls_dropped)          # (batch, 3)

        # Attention weights from all layers
        # outputs.attentions: tuple of (batch, num_heads, seq_len, seq_len)
        # One tensor per layer — we use the LAST layer
        last_layer_attn = outputs.attentions[-1]            # (batch, 12, max_len, max_len)

        # Extract CLS→all_tokens attention for the supervised heads
        # We use the FIRST x heads
        supervised_attn = last_layer_attn[:, :self.num_sup_heads, 0, :]
        # supervised_attn shape: (batch, num_sup_heads, max_len)

        # Compute attention supervision loss (if gt_attention provided)
        attn_loss = None
        if gt_attention is not None:
            attn_loss = self._compute_attention_loss(supervised_attn, gt_attention)

        return logits, supervised_attn, attn_loss

    def _compute_attention_loss(self, supervised_attn, gt_attention):
        """
        Cross-entropy between each supervised head's attention and ground truth.
        supervised_attn: (batch, num_sup_heads, max_len)
        gt_attention:    (batch, max_len)
        """
        batch_size, num_heads, seq_len = supervised_attn.shape
        gt_expanded = gt_attention.unsqueeze(1).expand_as(supervised_attn)
        # gt_expanded: (batch, num_sup_heads, max_len)

        # Flatten heads dimension → average cross-entropy across all supervised heads
        loss_fct = nn.CrossEntropyLoss(reduction='none')

        # Reshape: treat each head independently
        pred_flat = supervised_attn.reshape(batch_size * num_heads, seq_len)
        gt_flat   = gt_expanded.reshape(batch_size * num_heads, seq_len)

        # Cross-entropy on soft targets: -sum(gt * log(pred))
        log_pred  = torch.log(pred_flat + 1e-8)
        ce_loss   = -(gt_flat * log_pred).sum(dim=-1)   # (batch * num_heads,)
        attn_loss = ce_loss.mean()

        return attn_loss
```

---

## 7. Stage 6 — Dual-Loss Training (Classification + Attention Supervision)

### 7.1 Total Loss

```
L_total = L_pred + λ * L_att

Where:
  L_pred = CrossEntropyLoss(logits, true_label)     ← 3-class classification
  L_att  = mean CE across supervised heads vs GT attention
  λ      = attention_lambda (default: 100, tuned on val set)
```

**Why λ=100?** The classification loss operates over 3 classes (relatively small values), while the attention loss operates over 128 token positions (larger values per step). λ=100 balances their magnitudes so neither dominates training.

### 7.2 Training Loop

```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attention_mask"].to(device)
        type_ids   = batch["token_type_ids"].to(device)
        gt_attn    = batch["gt_attention"].to(device)   # (batch, max_len)
        labels     = batch["label"].to(device)

        optimizer.zero_grad()

        logits, supervised_attn, attn_loss = model(
            input_ids, attn_mask, type_ids, gt_attention=gt_attn
        )

        # Classification loss
        loss_fct = nn.CrossEntropyLoss()
        pred_loss = loss_fct(logits, labels)

        # Total loss
        if attn_loss is not None:
            loss = pred_loss + model.attn_lambda * attn_loss
        else:
            loss = pred_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# Setup
model     = BertHateXplain(num_classes=3, dropout=0.1,
                            num_supervised_heads=6, attention_lambda=100.0)
model     = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader) // 10,
    num_training_steps=len(train_loader) * NUM_EPOCHS
)
```

### 7.3 Hyperparameters (Best from Paper)

| Hyperparameter | BERT-HateXplain Value |
|---|---|
| Base model | `bert-base-uncased` |
| Max token length | 128 |
| Learning rate | 2e-5 |
| Optimizer | Adam |
| Dropout | 0.1 / 0.2 / 0.5 (tuned) |
| Attention lambda (λ) | 100 |
| Supervised heads (x) | 1 / 6 / 12 (tuned) |
| Batch size | 16 |
| Epochs | varies (early stop on val Macro F1) |

---

## 8. Stage 7 — Inference & Attention Head Extraction

### 8.1 Inference on a New Post

```python
def predict(model, text, tokenizer, device, max_len=128):
    """
    Run BERT-HateXplain on a raw text string.
    Returns: predicted_label, class_probabilities, token_importance_scores
    """
    model.eval()

    # Tokenize
    encoding = tokenizer(
        text, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids  = encoding["input_ids"].to(device)
    attn_mask  = encoding["attention_mask"].to(device)
    type_ids   = encoding["token_type_ids"].to(device)

    with torch.no_grad():
        logits, supervised_attn, _ = model(input_ids, attn_mask, type_ids,
                                            gt_attention=None)

    probs          = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred_label     = probs.argmax()
    label_names    = ["hatespeech", "offensive", "normal"]

    # Average supervised head attention over CLS→tokens
    # supervised_attn: (1, num_sup_heads, max_len)
    token_importance = supervised_attn.squeeze(0).mean(dim=0).cpu().numpy()
    # token_importance: (max_len,) — importance of each subword token

    # Map back to readable tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    # Filter out [PAD] tokens
    real_len    = attn_mask[0].sum().item()
    tokens      = tokens[:real_len]
    importance  = token_importance[:real_len]

    return {
        "predicted_label": label_names[pred_label],
        "probabilities":   dict(zip(label_names, probs.tolist())),
        "tokens":          tokens,
        "token_importance": importance.tolist()
    }
```

### 8.2 Example Output

```python
result = predict(model, "I hate all muslims they should leave our country", tokenizer, device)

# result = {
#   "predicted_label": "hatespeech",
#   "probabilities": {"hatespeech": 0.87, "offensive": 0.10, "normal": 0.03},
#   "tokens": ["[CLS]", "i", "hate", "all", "muslims", "they", "should", "leave",
#              "our", "country", "[SEP]"],
#   "token_importance": [0.01, 0.02, 0.18, 0.04, 0.31, 0.05, 0.08, 0.09, 0.03, 0.07, 0.12]
# }
```

The `token_importance` scores replace the human annotator rationales. Tokens like `"hate"` and `"muslims"` will have high importance scores — the model has learned to attend to them from the supervised attention training.

---

## 9. Stage 8 — Evaluation Metrics

### 9.1 Performance Metrics

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_performance(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            logits, _, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device)
            )
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            labels = batch["label"].numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc    = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    # AUROC: one-vs-rest, macro averaged
    auroc  = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")

    return {"accuracy": acc, "macro_f1": macro_f1, "auroc": auroc}
```

### 9.2 Explainability: Plausibility (IOU F1, Token F1, AUPRC)

Measures how well the model's token importance aligns with human annotator rationales.

```python
def iou_f1(pred_rationale_mask, gt_rationale_mask):
    """Both masks are binary lists over tokens."""
    pred_set = set(i for i, v in enumerate(pred_rationale_mask) if v == 1)
    gt_set   = set(i for i, v in enumerate(gt_rationale_mask) if v == 1)

    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0

    intersection = len(pred_set & gt_set)
    union        = len(pred_set | gt_set)
    iou = intersection / union
    return 1.0 if iou >= 0.5 else 0.0   # binary match for F1 computation
```

### 9.3 Explainability: Faithfulness (Comprehensiveness, Sufficiency)

```python
def comprehensiveness(model, input_ids, attn_mask, type_ids, rationale_mask,
                      predicted_class, device):
    """
    How much does removing the rationale tokens hurt the prediction?
    High comprehensiveness → rationale tokens were critical.
    """
    with torch.no_grad():
        logits_full, _, _ = model(input_ids, attn_mask, type_ids)
        prob_full = torch.softmax(logits_full, dim=-1)[0, predicted_class].item()

        # Mask out rationale tokens (set to [PAD] id or zero out attention mask)
        masked_ids  = input_ids.clone()
        for idx, is_rationale in enumerate(rationale_mask):
            if is_rationale:
                masked_ids[0, idx] = tokenizer.pad_token_id

        logits_masked, _, _ = model(masked_ids, attn_mask, type_ids)
        prob_masked = torch.softmax(logits_masked, dim=-1)[0, predicted_class].item()

    return prob_full - prob_masked   # positive → rationale was important


def sufficiency(model, input_ids, attn_mask, type_ids, rationale_mask,
                predicted_class, device):
    """
    Can the model still predict the class using ONLY the rationale tokens?
    Low sufficiency (close to 0) → rationale alone is enough.
    """
    with torch.no_grad():
        logits_full, _, _ = model(input_ids, attn_mask, type_ids)
        prob_full = torch.softmax(logits_full, dim=-1)[0, predicted_class].item()

        # Keep ONLY rationale tokens, mask everything else
        rationale_only_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        for idx, is_rationale in enumerate(rationale_mask):
            if is_rationale:
                rationale_only_ids[0, idx] = input_ids[0, idx]

        logits_rationale, _, _ = model(rationale_only_ids, attn_mask, type_ids)
        prob_rationale = torch.softmax(logits_rationale, dim=-1)[0, predicted_class].item()

    return prob_full - prob_rationale   # lower → rationale is sufficient
```

---

## 10. Mini-Project Extension: MLP Classifier on BERT Representations

### 10.1 Concept

Instead of (or in addition to) the BERT linear classification head, we extract the `[CLS]` token representation (768-dim) from BERT-HateXplain and feed it into a standalone **Multi-Layer Perceptron (MLP)**. This allows:

1. **Frozen BERT as a feature extractor** — faster iteration, less compute
2. **The model-generated attention scores** (128-dim, averaged over supervised heads) can be concatenated with the CLS embedding → richer 896-dim feature
3. **Classification of new, unseen prompts** without relying on human annotators
4. **Prediction of important tokens** — the supervised attention heads produce token importance scores automatically

### 10.2 Feature Extraction

```python
def extract_features(model, loader, device):
    """
    Extract (CLS embedding, attention importance, label) triples
    from the dataset using the trained BERT-HateXplain model.
    """
    model.eval()
    all_cls, all_attn_importance, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            type_ids   = batch["token_type_ids"].to(device)
            labels     = batch["label"]

            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attn_mask,
                token_type_ids=type_ids
            )

            # CLS embedding (768-dim)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Average supervised head attention → token importance (128-dim)
            last_attn = outputs.attentions[-1]   # (batch, 12, 128, 128)
            sup_attn  = last_attn[:, :model.num_sup_heads, 0, :].mean(dim=1)
            # sup_attn: (batch, 128)
            sup_attn  = sup_attn.cpu().numpy()

            all_cls.append(cls_emb)
            all_attn_importance.append(sup_attn)
            all_labels.extend(labels.numpy().tolist())

    cls_features   = np.vstack(all_cls)               # (N, 768)
    attn_features  = np.vstack(all_attn_importance)   # (N, 128)
    combined       = np.hstack([cls_features, attn_features])  # (N, 896)

    return combined, np.array(all_labels)
```

### 10.3 MLP Architecture

```python
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=896, hidden_dims=[512, 256], num_classes=3,
                 dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)   # returns logits
```

### 10.4 MLP Training

```python
from torch.utils.data import TensorDataset, DataLoader as TDL

# Extract features from trained BERT-HateXplain
X_train, y_train = extract_features(bert_model, train_loader, device)
X_val,   y_val   = extract_features(bert_model, val_loader, device)
X_test,  y_test  = extract_features(bert_model, test_loader, device)

# Normalize
from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Build PyTorch dataset
def make_tensor_loader(X, y, batch_size=64, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return TDL(ds, batch_size=batch_size, shuffle=shuffle)

mlp_train = make_tensor_loader(X_train, y_train)
mlp_val   = make_tensor_loader(X_val,   y_val,   shuffle=False)
mlp_test  = make_tensor_loader(X_test,  y_test,  shuffle=False)

# Train MLP
mlp_model = MLPClassifier(input_dim=896, hidden_dims=[512, 256], num_classes=3)
mlp_model  = mlp_model.to(device)
mlp_opt    = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
loss_fct   = nn.CrossEntropyLoss()

for epoch in range(20):
    mlp_model.train()
    for X_batch, y_batch in mlp_train:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = mlp_model(X_batch)
        loss   = loss_fct(logits, y_batch)
        mlp_opt.zero_grad()
        loss.backward()
        mlp_opt.step()
```

### 10.5 Inference on a New Prompt (End-to-End)

```python
def classify_new_prompt(text, bert_model, mlp_model, scaler, tokenizer,
                         device, max_len=128):
    """
    Full pipeline inference for a new, unseen text prompt.
    No human annotators required.

    Returns:
      - predicted label
      - class probabilities
      - token-level importance scores (model-generated rationale)
    """
    bert_model.eval()
    mlp_model.eval()

    # 1. Tokenize
    encoding = tokenizer(text, max_length=max_len, padding="max_length",
                          truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attn_mask = encoding["attention_mask"].to(device)
    type_ids  = encoding["token_type_ids"].to(device)

    with torch.no_grad():
        # 2. BERT forward pass
        outputs   = bert_model.bert(input_ids=input_ids,
                                     attention_mask=attn_mask,
                                     token_type_ids=type_ids)

        # 3. Extract CLS embedding (768-dim)
        cls_emb   = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (1, 768)

        # 4. Extract supervised head attention (128-dim)
        last_attn = outputs.attentions[-1]   # (1, 12, 128, 128)
        sup_attn  = last_attn[:, :bert_model.num_sup_heads, 0, :].mean(dim=1)
        sup_attn_np = sup_attn.cpu().numpy()   # (1, 128)

        # 5. Combine features
        combined  = np.hstack([cls_emb, sup_attn_np])   # (1, 896)
        combined  = scaler.transform(combined)
        combined_t = torch.tensor(combined, dtype=torch.float32).to(device)

        # 6. MLP classification
        logits    = mlp_model(combined_t)
        probs     = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    pred_label  = int(probs.argmax())
    label_names = ["hatespeech", "offensive", "normal"]

    # 7. Token importance from attention (model-generated rationale)
    real_len   = attn_mask[0].sum().item()
    tokens     = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())[:real_len]
    importance = sup_attn_np[0][:real_len]

    # Identify top-K important tokens
    top_k      = 5
    top_indices = importance.argsort()[::-1][:top_k]
    important_tokens = [(tokens[i], round(float(importance[i]), 4))
                        for i in top_indices]

    return {
        "text":               text,
        "predicted_label":    label_names[pred_label],
        "probabilities":      {k: round(float(v), 4)
                               for k, v in zip(label_names, probs)},
        "important_tokens":   important_tokens,   # model-generated rationale
        "all_token_scores":   list(zip(tokens, importance.round(4).tolist()))
    }
```

### 10.6 Example Outputs

```python
r1 = classify_new_prompt(
    "Those immigrants are ruining our country and need to be deported",
    bert_model, mlp_model, scaler, tokenizer, device
)
# {
#   "predicted_label": "hatespeech",
#   "probabilities":   {"hatespeech": 0.83, "offensive": 0.13, "normal": 0.04},
#   "important_tokens": [("immigrants", 0.21), ("ruining", 0.18),
#                        ("deported", 0.15), ("country", 0.08), ("those", 0.04)]
# }

r2 = classify_new_prompt(
    "I love all people regardless of their background",
    bert_model, mlp_model, scaler, tokenizer, device
)
# {
#   "predicted_label": "normal",
#   "probabilities":   {"hatespeech": 0.02, "offensive": 0.04, "normal": 0.94},
#   "important_tokens": [("love", 0.19), ("people", 0.15), ("background", 0.09), ...]
# }
```

---

## 11. Full Code Implementation

### 11.1 Environment Setup

```bash
# Python 3.8+
pip install torch transformers scikit-learn numpy scipy datasets

# Download HateXplain data
git clone https://github.com/hate-alert/HateXplain.git
cd HateXplain
```

### 11.2 Complete `run_pipeline.py`

```python
"""
HateXplain + MLP Extension — Complete Pipeline
============================================
Stages:
  1. Load dataset.json + post_id_divisions.json
  2. Preprocess → majority labels, ground truth attention
  3. Tokenize → BERT input format
  4. Train BERT-HateXplain (dual loss)
  5. Extract CLS + attention features
  6. Train MLP on extracted features
  7. Inference on new prompts
"""

import json, numpy as np, torch, torch.nn as nn
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.special import softmax
from collections import Counter

# ─── Config ─────────────────────────────────────────────────────────────────
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN            = 128
BATCH_SIZE         = 16
BERT_LR            = 2e-5
BERT_EPOCHS        = 5
MLP_LR             = 1e-3
MLP_EPOCHS         = 20
ATTENTION_LAMBDA   = 100.0
NUM_SUP_HEADS      = 6
DROPOUT            = 0.1
LABEL_MAP          = {"hatespeech": 0, "offensive": 1, "normal": 2}
LABEL_NAMES        = ["hatespeech", "offensive", "normal"]
TAU                = 1.0   # temperature for attention softmax

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ─── Data Loading ────────────────────────────────────────────────────────────
dataset   = json.load(open("Data/dataset.json"))
divisions = json.load(open("Data/post_id_divisions.json"))

# ─── Helper Functions ────────────────────────────────────────────────────────
def get_majority_label(annotators):
    labels  = [a['label'] for a in annotators]
    counts  = Counter(labels)
    top     = counts.most_common(1)[0]
    return top[0] if top[1] >= 2 else None

def compute_gt_attention(rationales, label, length, tau=TAU):
    if label == 2 or not rationales:
        return np.ones(length) / length
    avg = np.mean(rationales, axis=0)
    return softmax(avg / tau)

def align_to_subwords(word_tokens, word_attn):
    sub_attn = []
    for word, val in zip(word_tokens, word_attn):
        pieces = tokenizer.tokenize(word)
        n = max(len(pieces), 1)
        sub_attn.extend([val / n] * n)
    return sub_attn

def encode_sample(word_tokens, word_attn):
    sub_attn = [0.0] + align_to_subwords(word_tokens, word_attn) + [0.0]
    enc = tokenizer(" ".join(word_tokens), max_length=MAX_LEN,
                    padding="max_length", truncation=True, return_tensors="pt")
    sub_attn = sub_attn[:MAX_LEN]
    sub_attn += [0.0] * (MAX_LEN - len(sub_attn))
    return (enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            enc["token_type_ids"].squeeze(0),
            torch.tensor(sub_attn, dtype=torch.float32))

# ─── Dataset Class ───────────────────────────────────────────────────────────
class HXDataset(Dataset):
    def __init__(self, ids):
        self.data = []
        for pid in ids:
            e    = dataset[pid]
            lstr = get_majority_label(e["annotators"])
            if lstr is None: continue
            lbl  = LABEL_MAP[lstr]
            toks = e["post_tokens"]
            gt   = compute_gt_attention(e["rationales"], lbl, len(toks))
            iids, amask, tids, gtattn = encode_sample(toks, gt)
            self.data.append({"input_ids": iids, "attention_mask": amask,
                              "token_type_ids": tids, "gt_attention": gtattn,
                              "label": torch.tensor(lbl, dtype=torch.long)})
    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]

train_dl = DataLoader(HXDataset(divisions["train"]), batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(HXDataset(divisions["val"]),   batch_size=BATCH_SIZE)
test_dl  = DataLoader(HXDataset(divisions["test"]),  batch_size=BATCH_SIZE)

# ─── BERT-HateXplain Model ───────────────────────────────────────────────────
class BertHX(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert       = BertModel.from_pretrained("bert-base-uncased",
                                                     output_attentions=True)
        self.drop       = nn.Dropout(DROPOUT)
        self.clf        = nn.Linear(768, 3)
        self.num_sup    = NUM_SUP_HEADS

    def forward(self, input_ids, attention_mask, token_type_ids, gt_attention=None):
        out      = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        cls      = self.drop(out.last_hidden_state[:, 0, :])
        logits   = self.clf(cls)
        last_attn = out.attentions[-1]           # (B, 12, L, L)
        sup_attn  = last_attn[:, :self.num_sup, 0, :]  # (B, sup, L)

        attn_loss = None
        if gt_attention is not None:
            gt_exp   = gt_attention.unsqueeze(1).expand_as(sup_attn)
            log_pred = torch.log(sup_attn + 1e-8)
            attn_loss = -(gt_exp * log_pred).sum(dim=-1).mean()

        return logits, sup_attn, attn_loss

bert_model = BertHX().to(DEVICE)
optimizer  = torch.optim.AdamW(bert_model.parameters(), lr=BERT_LR)
scheduler  = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=len(train_dl)//10,
    num_training_steps=len(train_dl)*BERT_EPOCHS)
loss_fct   = nn.CrossEntropyLoss()

# ─── BERT Training ───────────────────────────────────────────────────────────
for epoch in range(BERT_EPOCHS):
    bert_model.train()
    total = 0
    for b in train_dl:
        iids = b["input_ids"].to(DEVICE);   amask = b["attention_mask"].to(DEVICE)
        tids = b["token_type_ids"].to(DEVICE); gta = b["gt_attention"].to(DEVICE)
        lbls = b["label"].to(DEVICE)
        logits, _, aloss = bert_model(iids, amask, tids, gta)
        loss = loss_fct(logits, lbls) + ATTENTION_LAMBDA * aloss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total += loss.item()
    print(f"Epoch {epoch+1}: loss={total/len(train_dl):.4f}")

# ─── Feature Extraction ──────────────────────────────────────────────────────
def extract(model, loader):
    model.eval()
    Xcls, Xattn, Ys = [], [], []
    with torch.no_grad():
        for b in loader:
            iids = b["input_ids"].to(DEVICE); amask = b["attention_mask"].to(DEVICE)
            tids = b["token_type_ids"].to(DEVICE)
            out  = model.bert(input_ids=iids, attention_mask=amask, token_type_ids=tids)
            Xcls.append(out.last_hidden_state[:, 0, :].cpu().numpy())
            lattn = out.attentions[-1][:, :model.num_sup, 0, :].mean(1)
            Xattn.append(lattn.cpu().numpy())
            Ys.extend(b["label"].numpy())
    return (np.vstack(Xcls), np.vstack(Xattn), np.array(Ys))

Xc_tr, Xa_tr, y_tr = extract(bert_model, train_dl)
Xc_v,  Xa_v,  y_v  = extract(bert_model, val_dl)
Xc_te, Xa_te, y_te = extract(bert_model, test_dl)

X_tr = np.hstack([Xc_tr, Xa_tr])   # (N, 896)
X_v  = np.hstack([Xc_v,  Xa_v])
X_te = np.hstack([Xc_te, Xa_te])

scaler = StandardScaler().fit(X_tr)
X_tr, X_v, X_te = scaler.transform(X_tr), scaler.transform(X_v), scaler.transform(X_te)

def to_dl(X, y, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=64, shuffle=shuffle)

mlp_tr = to_dl(X_tr, y_tr); mlp_v = to_dl(X_v, y_v, False); mlp_te = to_dl(X_te, y_te, False)

# ─── MLP Classifier ──────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(896, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
    def forward(self, x): return self.net(x)

mlp = MLP().to(DEVICE)
mlp_opt = torch.optim.Adam(mlp.parameters(), lr=MLP_LR)

for epoch in range(MLP_EPOCHS):
    mlp.train()
    for X_b, y_b in mlp_tr:
        logits = mlp(X_b.to(DEVICE))
        loss   = loss_fct(logits, y_b.to(DEVICE))
        mlp_opt.zero_grad(); loss.backward(); mlp_opt.step()

# ─── Inference on New Prompt ─────────────────────────────────────────────────
def classify(text):
    bert_model.eval(); mlp.eval()
    enc = tokenizer(text, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    iids = enc["input_ids"].to(DEVICE); amask = enc["attention_mask"].to(DEVICE)
    tids = enc["token_type_ids"].to(DEVICE)
    with torch.no_grad():
        out   = bert_model.bert(input_ids=iids, attention_mask=amask, token_type_ids=tids)
        cls   = out.last_hidden_state[:, 0, :].cpu().numpy()
        sattn = out.attentions[-1][:, :bert_model.num_sup, 0, :].mean(1).cpu().numpy()
        feat  = scaler.transform(np.hstack([cls, sattn]))
        probs = torch.softmax(mlp(torch.tensor(feat, dtype=torch.float32).to(DEVICE)), 1)
        probs = probs.squeeze(0).cpu().numpy()

    rlen  = amask[0].sum().item()
    toks  = tokenizer.convert_ids_to_tokens(iids[0].cpu().numpy())[:rlen]
    imp   = sattn[0][:rlen]
    top5  = [(toks[i], round(float(imp[i]), 4)) for i in imp.argsort()[::-1][:5]]

    return {"label": LABEL_NAMES[probs.argmax()],
            "probs": {k: round(float(v), 4) for k, v in zip(LABEL_NAMES, probs)},
            "important_tokens": top5}

# Test it
print(classify("Those people don't belong here and should be removed"))
```

---

## 12. Architecture Diagram (Text)

```
dataset.json
    │
    ▼
[Preprocessing]
  majority_vote(annotators) → label ∈ {0,1,2}
  average(rationales)       → word-level binary mask
  temperature_softmax(mask) → soft ground truth attention (word-level)
    │
    ▼
[BERT Tokenization]
  BertTokenizer → subword tokens + [CLS],[SEP],[PAD]
  align GT attention to subword tokens
  → input_ids, attention_mask, token_type_ids, gt_attention  (all len=128)
    │
    ▼
[BERT Encoder — 12 layers, 12 heads, hidden=768]
  Layer 1 → Layer 2 → ... → Layer 12
  Each layer: Multi-Head Self-Attention + FFN + LayerNorm + Residual
    │                             │
    ▼                             ▼
[CLS hidden state]    [Last layer attention weights]
   (batch, 768)        (batch, 12, 128, 128)
    │                             │
    │                    [Supervised heads 0..x-1]
    │                    CLS row: attn[:, 0..x, 0, :]
    │                    → (batch, x, 128)
    │                             │
    ▼                             ▼
[Dropout]              [Attention Supervision Loss]
    │                   CE( pred_attn, gt_attention )
    ▼                             │
[Linear(768→3)]                   │
    │                             │
    ▼                             │
[L_pred]          +    λ * [L_att]  =  L_total
  CrossEntropy                         ↑
  (logits, label)              backward() → AdamW update
    │
    ▼
[Inference on New Text]
    │
    ├─ CLS embedding (768-dim)
    │
    └─ Avg supervised head attention (128-dim) → token importance scores
              (model-generated rationale, no human annotators needed)
    │
    ▼
[Feature Concat: 896-dim]
    │
[StandardScaler normalize]
    │
    ▼
[MLP Classifier: 896→512→256→3]
  ReLU + BatchNorm + Dropout
    │
    ▼
Predicted Label + Probabilities + Important Tokens
```

---

## Key Design Decisions & Insights

**Why supervised attention instead of LIME?**
LIME is a post-hoc method — it perturbs the input after training and observes output changes. Supervised attention is baked into training: the model is explicitly taught which tokens matter, so its representations improve. The paper shows BiRNN-HateXplain [Attn] achieves AUPRC of 0.841 vs 0.648 for the base BiRNN.

**Why CLS + attention as MLP input?**
The CLS token captures global sentence semantics (useful for classification). The attention scores capture token-level salience (useful for interpretability and as an auxiliary signal). Together they give the MLP both "what is the post about" and "which words drove that assessment."

**Why λ=100 for attention loss?**
The attention supervision loss (CE over 128 token positions) is naturally larger in scale than the 3-class classification loss. λ=100 equalizes their gradient contributions so both learning signals shape the model.

**Trade-off: performance vs. explainability**
The paper's key finding: BERT-HateXplain achieves better bias metrics (GMB-Subgroup-AUC: 0.807 vs 0.762 for plain BERT) because the rationale supervision teaches the model what specific hate-relevant words look like — reducing false positives on non-hateful text mentioning minority communities. However, this comes at a cost to plausibility (IOU F1 drops slightly). The MLP extension lets you tune this trade-off independently of BERT's internal heads.
