#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize LIME explanations"
    )
    parser.add_argument(
        "--explanation-file",
        type=str,
        default="artifacts/explanations/explanation.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/explanations/figures",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
    )
    return parser.parse_args()


def plot_token_importance(
    positive_tokens: List[Dict],
    negative_tokens: List[Dict],
    output_path: Path
) -> None:
    """Create horizontal bar chart of token importance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Positive tokens
    if positive_tokens:
        tokens_pos = [t["token"][:20] for t in positive_tokens]
        weights_pos = [t["weight"] for t in positive_tokens]

        y_pos = np.arange(len(tokens_pos))
        ax1.barh(y_pos, weights_pos, color='red', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(tokens_pos)
        ax1.set_xlabel("Weight (phishing contribution)")
        ax1.set_title("Tokens Indicating Phishing")
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    else:
        ax1.text(0.5, 0.5, "No positive tokens", ha='center', va='center')
        ax1.set_title("Tokens Indicating Phishing")

    # Negative tokens
    if negative_tokens:
        tokens_neg = [t["token"][:20] for t in negative_tokens]
        weights_neg = [t["weight"] for t in negative_tokens]

        y_neg = np.arange(len(tokens_neg))
        ax2.barh(y_neg, weights_neg, color='green', alpha=0.7)
        ax2.set_yticks(y_neg)
        ax2.set_yticklabels(tokens_neg)
        ax2.set_xlabel("Weight (safe contribution)")
        ax2.set_title("Tokens Indicating Safe")
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, "No negative tokens", ha='center', va='center')
        ax2.set_title("Tokens Indicating Safe")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_highlighted_text(
    tokens: List[str],
    positive_weights: Dict[str, float],
    negative_weights: Dict[str, float],
    output_path: Path
) -> None:
    """Create HTML visualization with highlighted tokens."""
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .text-container { 
                border: 1px solid #ddd; 
                padding: 20px; 
                margin: 20px 0;
                background-color: #f9f9f9;
            }
            .token {
                display: inline-block;
                padding: 2px 4px;
                margin: 2px;
                border-radius: 3px;
            }
            .positive { background-color: #ffcccc; color: #990000; }
            .negative { background-color: #ccffcc; color: #009900; }
            .neutral { background-color: #f0f0f0; color: #666; }
            .legend {
                margin: 20px 0;
                padding: 10px;
                border: 1px solid #ddd;
            }
            .legend-item {
                display: inline-block;
                margin-right: 20px;
            }
            .legend-color {
                display: inline-block;
                width: 20px;
                height: 20px;
                margin-right: 5px;
                vertical-align: middle;
            }
        </style>
    </head>
    <body>
        <h2>LIME Explanation: Token Highlighting</h2>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ffcccc;"></div>
                <span>Phishing-indicating (positive weight)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ccffcc;"></div>
                <span>Safe-indicating (negative weight)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #f0f0f0;"></div>
                <span>Neutral</span>
            </div>
        </div>
        <div class="text-container">
    """

    for token in tokens:
        token_text = token.replace("<", "&lt;").replace(">", "&gt;")
        if token in positive_weights:
            weight = positive_weights[token]
            intensity = min(0.9, abs(weight) / max(positive_weights.values()))
            color_intensity = int(255 - (intensity * 100))
            html_content += f'<span class="token positive" style="background-color: #ff{color_intensity:02x}{color_intensity:02x};">{token_text}</span>'
        elif token in negative_weights:
            weight = negative_weights[token]
            intensity = min(0.9, abs(weight) / max(negative_weights.values()))
            color_intensity = int(255 - (intensity * 100))
            html_content += f'<span class="token negative" style="background-color: #{color_intensity:02x}ff{color_intensity:02x};">{token_text}</span>'
        else:
            html_content += f'<span class="token neutral">{token_text}</span>'

    html_content += """
        </div>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main() -> None:
    args = parse_args()

    # Load explanation
    with open(args.explanation_file, "r", encoding="utf-8") as f:
        explanation = json.load(f)

    if "error" in explanation:
        print(f"Error in explanation: {explanation['error']}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get top tokens
    positive_tokens = explanation["top_positive_tokens"][:args.max_tokens]
    negative_tokens = explanation["top_negative_tokens"][:args.max_tokens]

    # Plot token importance
    plot_path = output_dir / "token_importance.png"
    plot_token_importance(positive_tokens, negative_tokens, plot_path)
    print(f"Token importance plot saved to {plot_path}")

    # Create highlighted text visualization
    # Build weight dictionaries
    positive_weights = {t["token"]: t["weight"] for t in positive_tokens}
    negative_weights = {t["token"]: t["weight"] for t in negative_tokens}

    html_path = output_dir / "highlighted_text.html"
    plot_highlighted_text(
        explanation["tokens"],
        positive_weights,
        negative_weights,
        html_path
    )
    print(f"Highlighted text HTML saved to {html_path}")

    # Print summary
    print("\n" + "="*50)
    print("LIME EXPLANATION VISUALIZATION COMPLETE")
    print("="*50)
    print(f"Original probability: {explanation['original_probability']:.4f}")
    print(f"Prediction: {explanation['predicted_class']}")
    print(f"\nTop {len(positive_tokens)} phishing-indicating tokens:")
    for t in positive_tokens:
        print(f"  + {t['token']:20s} ({t['weight']:.4f})")
    print(f"\nTop {len(negative_tokens)} safe-indicating tokens:")
    for t in negative_tokens:
        print(f"  - {t['token']:20s} ({t['weight']:.4f})")
    print("="*50)


if __name__ == "__main__":
    main()
