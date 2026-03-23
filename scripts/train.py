#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

from src.data import load_dataset, save_splits, stratified_split
from src.model import build_tokenizer_and_model, compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DistilRoBERTa phishing classifier")
    parser.add_argument("--dataset", default="data/data.csv", help="Path to source dataset CSV")
    parser.add_argument("--output-dir", default="artifacts/model", help="Directory for model artifacts")
    parser.add_argument("--model-name", default="distilroberta-base", help="HF model checkpoint")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    train_df, val_df, test_df = stratified_split(df, args.val_size, args.test_size, args.seed)
    save_splits(train_df, val_df, test_df, output_dir)

    tokenizer, model = build_tokenizer_and_model(args.model_name)

    def preprocess(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(preprocess, batched=True)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False).map(preprocess, batched=True)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False).map(preprocess, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "seed": args.seed,
    }

    optional_fields = {
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "logging_steps": 50,
    }
    for field, value in optional_fields.items():
        if field in ta_params:
            ta_kwargs[field] = value

    # Transformers renamed a few TrainingArguments fields across versions.
    if "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "epoch"
    elif "evaluate_during_training" in ta_params:
        ta_kwargs["evaluate_during_training"] = True

    if "save_strategy" in ta_params:
        ta_kwargs["save_strategy"] = "epoch"

    if "report_to" in ta_params:
        ta_kwargs["report_to"] = []

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_metrics = trainer.evaluate(eval_dataset=test_ds)
    eval_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in eval_metrics.items()}

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)


if __name__ == "__main__":
    main()
