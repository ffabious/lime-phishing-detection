from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_CANDIDATES = ("Email Text", "email_text", "text", "email")
LABEL_CANDIDATES = ("Email Type", "email_type", "label", "target")


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...], kind: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find {kind} column. Tried: {candidates}")


def _normalize_label(raw_label: object) -> int:
    value = str(raw_label).strip().lower()

    if any(token in value for token in ("phish", "spam", "malicious")):
        return 1
    if any(token in value for token in ("safe", "legit", "ham", "benign")):
        return 0

    if value in {"1", "true", "t", "yes"}:
        return 1
    if value in {"0", "false", "f", "no"}:
        return 0

    raise ValueError(f"Unrecognized label value: {raw_label!r}")


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and normalize source CSV into columns: text, label."""
    df = pd.read_csv(csv_path)

    text_col = _resolve_column(df, TEXT_CANDIDATES, "text")
    label_col = _resolve_column(df, LABEL_CANDIDATES, "label")

    out = pd.DataFrame({
        "text": df[text_col].astype(str).str.strip(),
        "label": df[label_col].map(_normalize_label),
    })
    out = out[out["text"].str.len() > 0].dropna().reset_index(drop=True)

    return out


def stratified_split(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size <= 0 or test_size <= 0:
        raise ValueError("val_size and test_size must be > 0")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1")

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=df["label"],
    )

    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)
