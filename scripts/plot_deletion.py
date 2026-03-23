#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot deletion-test curves and summary metrics")
    parser.add_argument("--input-json", default="artifacts/deletion_test.json")
    parser.add_argument("--output-image", default="artifacts/deletion_curve.png")
    parser.add_argument("--output-summary", default="artifacts/deletion_summary.json")
    return parser.parse_args()


def area_under_curve(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def main() -> None:
    args = parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    fractions = np.array(data["fractions"], dtype=float)
    ranked = np.array(data["ranked_avg_curve"], dtype=float)
    random_curve = np.array(data["random_avg_curve"], dtype=float)

    ranked_auc = area_under_curve(fractions, ranked)
    random_auc = area_under_curve(fractions, random_curve)
    auc_improvement = random_auc - ranked_auc

    point_gap = random_curve - ranked
    mean_gap = float(np.mean(point_gap))
    max_gap = float(np.max(point_gap))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fractions, ranked, marker="o", linewidth=2.0, label="Ranked deletion")
    ax.plot(fractions, random_curve, marker="s", linewidth=2.0, label="Random deletion")
    ax.fill_between(fractions, ranked, random_curve, alpha=0.15)

    ax.set_title("Deletion Test: Ranked vs Random")
    ax.set_xlabel("Fraction of removed tokens")
    ax.set_ylabel("Mean phishing probability")
    ax.grid(alpha=0.3)
    ax.legend()

    text_box = (
        f"n={int(data.get('num_samples', 0))}\n"
        f"AUC ranked={ranked_auc:.4f}\n"
        f"AUC random={random_auc:.4f}\n"
        f"AUC gap (random-ranked)={auc_improvement:.4f}\n"
        f"Mean point gap={mean_gap:.4f}"
    )
    ax.text(
        0.02,
        0.03,
        text_box,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    output_image = Path(args.output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_image, dpi=180)
    plt.close(fig)

    summary = {
        "num_samples": int(data.get("num_samples", 0)),
        "ranked_auc": ranked_auc,
        "random_auc": random_auc,
        "auc_gap_random_minus_ranked": auc_improvement,
        "mean_point_gap": mean_gap,
        "max_point_gap": max_gap,
        "interpretation": {
            "better_if": "ranked_auc < random_auc and mean_point_gap > 0",
            "result": bool(ranked_auc < random_auc and mean_gap > 0),
        },
    }

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
