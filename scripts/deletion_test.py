#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.deletion import deletion_curve, occlusion_importance, random_deletion_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deletion test with random baseline")
    parser.add_argument("--model-dir", default="artifacts/model")
    parser.add_argument("--split-path", default="artifacts/model/test_split.csv")
    parser.add_argument("--output-json", default="artifacts/deletion_test.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    split_df = pd.read_csv(args.split_path)
    if "label" not in split_df.columns or "text" not in split_df.columns:
        raise ValueError("Expected split CSV to contain columns: text,label")

    phishing_df = split_df[split_df["label"] == 1]
    if phishing_df.empty:
        raise ValueError("No phishing-labeled samples found in test split")

    sample_count = min(args.num_samples, len(phishing_df))
    sampled_texts = phishing_df.sample(n=sample_count, random_state=args.seed)["text"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    ranked_curves = []
    random_curves = []
    per_example = []

    for text in sampled_texts:
        tokens, importances, base_prob = occlusion_importance(
            text,
            model,
            tokenizer,
            device,
            max_length=args.max_length,
            max_tokens=args.max_tokens,
        )
        ranking = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

        fractions, ranked_curve = deletion_curve(
            tokens,
            ranking,
            model,
            tokenizer,
            device,
            max_length=args.max_length,
            steps=args.steps,
        )
        random_curve = random_deletion_curve(
            tokens,
            model,
            tokenizer,
            device,
            max_length=args.max_length,
            rng=rng,
            steps=args.steps,
            trials=args.random_trials,
        )

        ranked_curves.append(ranked_curve)
        random_curves.append(random_curve)
        per_example.append(
            {
                "base_prob": base_prob,
                "ranked_curve": ranked_curve,
                "random_curve": random_curve,
            }
        )

    ranked_avg = np.mean(np.array(ranked_curves), axis=0)
    random_avg = np.mean(np.array(random_curves), axis=0)

    results = {
        "num_samples": sample_count,
        "fractions": fractions,
        "ranked_avg_curve": ranked_avg.tolist(),
        "random_avg_curve": random_avg.tolist(),
        "mean_curve_gap": float(np.mean(random_avg - ranked_avg)),
        "examples": per_example,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
