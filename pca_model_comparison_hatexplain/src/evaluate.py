import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from .data import majority_label, preprocess_example, get_cleaned_label
from .config import ID2LABEL

def evaluate(model, tokenizer, device, dataset):

    y_true, y_pred = [], []
    rows = []

    for ex in tqdm(dataset):

        true_label = majority_label(ex["annotators"])
        if true_label is None:
            continue

        text = preprocess_example(ex)

        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()

        y_true.append(true_label)
        y_pred.append(pred)

        rows.append({
            "text": text,
            "true_label": ID2LABEL[true_label],
            "pred_label": ID2LABEL[pred],
        })

    print("\n=== RESULTS ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))

    print("\nClassification Report:\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(3)]
    ))

    return rows


def evaluate_cleaned(model, tokenizer, device, df):
    y_true, y_pred = [], []
    rows = []

    for _, ex in tqdm(df.iterrows(), total=len(df)):
        true_label = get_cleaned_label(ex)
        text = preprocess_example(ex)

        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()

        y_true.append(true_label)
        y_pred.append(pred)

        rows.append({
            "post_id": ex.get("post_id", None),
            "text": text,
            "true_label": ID2LABEL[true_label],
            "pred_label": ID2LABEL[pred],
        })

    print("\n=== CLEANED DATA RESULTS ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))


    print("\nClassification Report:\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(3)]
    ))

    return rows, y_true, y_pred