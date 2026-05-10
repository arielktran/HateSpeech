import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import MODEL_NAME

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    return tokenizer, model, device