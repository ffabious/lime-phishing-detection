# LIME Phishing Detection

- dataset loading and stratified splits
- DistilRoBERTa training and test metrics
- deletion test with random baseline

## Setup

```bash
/usr/local/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Classifier

```bash
PYTHONPATH=. python scripts/train.py \
  --dataset data/data.csv \
  --output-dir artifacts/model \
  --model-name distilroberta-base \
  --epochs 1 \
  --train-batch-size 16 \
  --eval-batch-size 32 \
  --max-length 256
```

Outputs:

- `artifacts/model/train_split.csv`
- `artifacts/model/val_split.csv`
- `artifacts/model/test_split.csv`
- `artifacts/model/test_metrics.json`
- model/tokenizer files in `artifacts/model/`

## Run Deletion Test

```bash
PYTHONPATH=. python scripts/deletion_test.py \
  --model-dir artifacts/model \
  --split-path artifacts/model/test_split.csv \
  --output-json artifacts/deletion_test.json \
  --num-samples 30 \
  --max-tokens 40 \
  --steps 10
```

Output:

- `artifacts/deletion_test.json` with ranked deletion curve vs random baseline

## Visualize Deletion Test

```bash
PYTHONPATH=. python scripts/plot_deletion.py \
  --input-json artifacts/deletion_test.json \
  --output-image artifacts/deletion_curve.png \
  --output-summary artifacts/deletion_summary.json
```

Outputs:

- `artifacts/deletion_curve.png` with ranked vs random curves
- `artifacts/deletion_summary.json` with AUC and gap metrics

## Notes

- The deletion test uses leave-one-token-out occlusion scores to rank tokens.
- This keeps evaluation pipeline independent of the LIME implementation.
- Interpretation: lower ranked curve (and lower ranked AUC) than random indicates better attribution faithfulness.
