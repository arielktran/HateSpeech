from datasets import load_dataset
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import ast

LABEL2ID = {
    "hatespeech": 0,
    "normal": 1,
    "offensive": 2,
}

ID2LABEL = {
    0: "hatespeech",
    1: "normal",
    2: "offensive",
}

def load_hatexplain():
    return load_dataset(
        "Hate-speech-CNERG/hatexplain",
        trust_remote_code=True
    )


def load_cleaned_hatexplain(path="data/df_final.csv"):
    df = pd.read_csv(path)

    # Keep only rows with valid majority labels
    df = df[df["majority_label"].isin(LABEL2ID.keys())].copy()

    # Convert label string to numeric ID
    df["label_id"] = df["majority_label"].map(LABEL2ID)

    # Ensure post_text is clean string
    df["post_text"] = df["post_text"].fillna("").astype(str)

    # Optional: convert post_tokens from stringified list to actual list
    if "post_tokens" in df.columns:
        df["post_tokens"] = df["post_tokens"].apply(_safe_parse_list)

    return df.reset_index(drop=True)


def _safe_parse_list(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []
    

def get_cleaned_label(example):
    """
    For cleaned CSV dataframe rows.
    """
    if isinstance(example, dict):
        return example["label_id"]

    return example.label_id

def majority_label(annotators):
    labels = annotators["label"]

    counts = Counter(labels)
    top_label, top_count = counts.most_common(1)[0]

    return top_label if top_count >= 2 else None

def preprocess_example(example):
    """
    Works for both:
    1. Hugging Face HateXplain examples
    2. cleaned CSV dataframe rows
    """

    # Cleaned CSV row as pandas Series
    if hasattr(example, "post_text"):
        return str(example.post_text)

    # Cleaned CSV row as dict
    if isinstance(example, dict) and "post_text" in example:
        return str(example["post_text"])

    # Original HF dataset format
    if isinstance(example, dict) and "post_tokens" in example:
        return " ".join(example["post_tokens"])

    raise ValueError("Could not preprocess example: missing post_text or post_tokens.")


def split_cleaned_dataset(
    df,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    random_state=42
):
    """
    Stratified train/val/test split using label_id.
    """

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    # First split: train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_size),
        stratify=df["label_id"],
        random_state=random_state
    )

    # Second split: val vs test
    relative_test_size = test_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label_id"],
        random_state=random_state
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )