import numpy as np
import torch
from tqdm import tqdm

from .data import preprocess_example


def extract_cls_embeddings(
    df,
    tokenizer,
    model,
    device,
    batch_size=32,
    max_length=128
):
    """
    Extract final-layer CLS embeddings from BERT-HateXplain.

    Returns:
        embeddings: np.ndarray of shape (num_examples, 768)
        labels: np.ndarray of shape (num_examples,)
        texts: list[str]
    """

    model.eval()

    all_embeddings = []
    all_labels = []
    all_texts = []

    texts = [preprocess_example(row) for _, row in df.iterrows()]
    labels = df["label_id"].to_numpy()

    for start_idx in tqdm(range(0, len(texts), batch_size), desc="Extracting CLS embeddings"):
        batch_texts = texts[start_idx:start_idx + batch_size]
        batch_labels = labels[start_idx:start_idx + batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True
            )

            # Final layer CLS embedding
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]

        all_embeddings.append(cls_embeddings.cpu().numpy())
        all_labels.extend(batch_labels)
        all_texts.extend(batch_texts)

    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)

    return embeddings, labels, all_texts