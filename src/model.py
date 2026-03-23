from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_tokenizer_and_model(model_name: str, num_labels: int = 2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def predict_phishing_proba(
    texts: Iterable[str],
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int = 16,
) -> np.ndarray:
    model.eval()
    all_probs: list[np.ndarray] = []
    text_list = list(texts)

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.append(probs.detach().cpu().numpy())

    if not all_probs:
        return np.array([], dtype=float)

    return np.concatenate(all_probs)
