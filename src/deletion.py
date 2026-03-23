from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np
import torch

from src.model import predict_phishing_proba

_WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass
class DeletionResult:
    fractions: list[float]
    ranked_curve: list[float]
    random_curve: list[float]


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _join_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)


def _remove_token_indices(tokens: list[str], remove_indices: set[int]) -> str:
    kept = [tok for idx, tok in enumerate(tokens) if idx not in remove_indices]
    return _join_tokens(kept)


def occlusion_importance(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    max_tokens: int = 40,
) -> tuple[list[str], list[float], float]:
    """Score token importance by leave-one-token-out probability drop."""
    tokens = tokenize_words(text)
    if not tokens:
        return [], [], 0.0

    tokens = tokens[:max_tokens]
    base_prob = float(
        predict_phishing_proba([_join_tokens(tokens)], model, tokenizer, device, max_length=max_length)[0]
    )

    perturbed_texts = []
    for idx in range(len(tokens)):
        perturbed_texts.append(_remove_token_indices(tokens, {idx}))

    perturbed_probs = predict_phishing_proba(
        perturbed_texts,
        model,
        tokenizer,
        device,
        max_length=max_length,
    )
    importances = [base_prob - float(p) for p in perturbed_probs]

    return tokens, importances, base_prob


def deletion_curve(
    tokens: list[str],
    ranking: list[int],
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    steps: int = 10,
) -> tuple[list[float], list[float]]:
    n = len(tokens)
    if n == 0:
        return [0.0], [0.0]

    fractions: list[float] = []
    texts: list[str] = []

    for step in range(steps + 1):
        frac = step / steps
        k = min(n, int(round(frac * n)))
        remove = set(ranking[:k])

        fractions.append(frac)
        texts.append(_remove_token_indices(tokens, remove))

    probs = predict_phishing_proba(texts, model, tokenizer, device, max_length=max_length)
    return fractions, [float(p) for p in probs]


def random_deletion_curve(
    tokens: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    rng: random.Random,
    steps: int = 10,
    trials: int = 10,
) -> list[float]:
    n = len(tokens)
    if n == 0:
        return [0.0]

    all_curves = []
    for _ in range(trials):
        shuffled = list(range(n))
        rng.shuffle(shuffled)
        _, curve = deletion_curve(
            tokens,
            shuffled,
            model,
            tokenizer,
            device,
            max_length=max_length,
            steps=steps,
        )
        all_curves.append(curve)

    return list(np.mean(np.array(all_curves), axis=0))
