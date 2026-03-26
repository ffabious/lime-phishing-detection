#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.lime_text import LimeTextExplainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LIME explanations for phishing detection"
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts/model",
    )
    parser.add_argument(
        "--text",
        type=str,
    )
    parser.add_argument(
        "--input-file",
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/explanations",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3000,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_dir: str, device: torch.device):
    """Load model and tokenizer from saved artifacts."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def create_predict_fn(model, tokenizer, device, max_length: int, batch_size: int):
    def predict_fn(texts: List[str]) -> np.ndarray:
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            with torch.no_grad():
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs)
    
    return predict_fn


def explain_single_text(
    text: str,
    explainer: LimeTextExplainer,
    output_dir: Path,
    num_features: int,
    positive_only: bool
) -> dict:
    """Explain a single text and save results."""
    explanation = explainer.explain_instance(
        text,
        num_features=num_features,
        positive_only=positive_only
    )
    
    # Create a readable summary
    summary = {
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "original_probability": explanation["original_probability"],
        "prediction": explanation["predicted_class"],
        "top_positive_tokens": [
            {"token": t["token"], "weight": t["weight"]}
            for t in explanation["top_positive_tokens"][:5]
        ],
        "top_negative_tokens": [
            {"token": t["token"], "weight": t["weight"]}
            for t in explanation["top_negative_tokens"][:5]
        ]
    }
    
    return explanation, summary


def explain_batch(
    texts: List[str],
    explainer: LimeTextExplainer,
    output_dir: Path,
    num_features: int,
    positive_only: bool
) -> List[dict]:
    """Explain multiple texts."""
    explanations = []
    
    for idx, text in enumerate(texts):
        print(f"Explaining example {idx + 1}/{len(texts)}...")
        explanation, summary = explain_single_text(
            text, explainer, output_dir, num_features, positive_only
        )
        
        # Save individual explanation
        output_path = output_dir / f"explanation_{idx}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(explanation, f, indent=2, ensure_ascii=False)
        
        explanations.append({
            "index": idx,
            "summary": summary,
            "file": str(output_path)
        })
    
    return explanations


def main() -> None:
    args = parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
    tokenizer, model = load_model_and_tokenizer(args.model_dir, device)
    predict_fn = create_predict_fn(
        model, tokenizer, device, args.max_length, args.batch_size
    )
    
    # Initialize LIME explainer
    print("Initializing LIME explainer...")
    explainer = LimeTextExplainer(
        predict_fn=predict_fn,
        num_samples=args.num_samples,
        random_seed=args.seed
    )
    
    # Process inputs
    texts_to_explain = []
    
    if args.text:
        texts_to_explain = [args.text]
    elif args.input_file:
        df = pd.read_csv(args.input_file)
        if "text" not in df.columns:
            raise ValueError("Input file must have 'text' column")
        texts_to_explain = df["text"].tolist()
    else:
        # Use default example text
        texts_to_explain = [
            "Your account has been suspended. Click here to verify your identity immediately: http://suspicious-link.com",
            "Please find attached the report for Q4 as requested. Let me know if you need any changes.",
            "URGENT: Your PayPal account has been limited. Confirm your details now: https://fake-paypal.com/secure"
        ]
        print("No input provided. Using default examples.")
    
    # Generate explanations
    print(f"Generating explanations for {len(texts_to_explain)} texts...")
    
    if len(texts_to_explain) == 1:
        # Single text
        explanation, summary = explain_single_text(
            texts_to_explain[0],
            explainer,
            output_dir,
            args.num_features,
            args.positive_only
        )
        
        # Save explanation
        output_path = output_dir / "explanation.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(explanation, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_path = output_dir / "explanation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExplanation saved to {output_path}")
        print(f"Summary saved to {summary_path}")
        
        # Print readable summary
        print("\n" + "="*50)
        print("EXPLANATION SUMMARY")
        print("="*50)
        print(f"Prediction: {summary['prediction'].upper()}")
        print(f"Probability: {summary['original_probability']:.4f}")
        print("\nTop phishing-indicating tokens:")
        for token_info in summary['top_positive_tokens'][:5]:
            print(f"  + {token_info['token']:15s} ({token_info['weight']:.4f})")
        print("\nTop safe-indicating tokens:")
        for token_info in summary['top_negative_tokens'][:5]:
            print(f"  - {token_info['token']:15s} ({token_info['weight']:.4f})")
        print("="*50)
        
    else:
        # Batch mode
        explanations = explain_batch(
            texts_to_explain,
            explainer,
            output_dir,
            args.num_features,
            args.positive_only
        )
        
        # Save batch summary
        batch_summary = {
            "num_explanations": len(explanations),
            "explanations": explanations
        }
        
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"\nGenerated {len(explanations)} explanations in {output_dir}")
        print(f"Batch summary saved to {summary_path}")


if __name__ == "__main__":
    main()
