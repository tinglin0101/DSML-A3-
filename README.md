# Sentiment Classification Pipeline

A binary sentiment classifier (positive / negative) built on top of a fine-tuned sentence-transformer embedding model and a Random Forest regressor, with self-training to leverage unlabeled test data.

## Overview

| Stage | What happens |
|-------|-------------|
| **Embedding** | Encodes text with `mixedbread-ai/mxbai-embed-large-v1` (1024-dim) |
| **Fine-tuning** | Adapts the sentence-transformer on cosine-similarity pairs sampled from the labeled training set |
| **Self-training** | Iteratively adds high-confidence pseudo-labels from the test set and re-trains (3 rounds, threshold ±0.85) |
| **Prediction** | A `StandardScaler → RandomForestRegressor` pipeline scores each test sample; threshold 0.5 produces the final binary label |

## Requirements

```
numpy
pandas
torch
sentence-transformers
datasets
transformers
scikit-learn
```

Install with:

```bash
pip install numpy pandas torch sentence-transformers datasets transformers scikit-learn
```

## Input Files

| File | Columns | Description |
|------|---------|-------------|
| `train_2022.csv` | `row_id`, `TEXT`, `LABEL` | Labeled training samples (`LABEL` ∈ {0, 1}) |
| `test_no_answer_2022.csv` | `row_id`, `TEXT` | Unlabeled test samples |

## Usage

```bash
python prediction.py
```

Output is written to `result.csv` with columns `row_id` and `LABEL`.

## Self-Training Details

- **Rounds:** 3 (round 0 = labeled data only, rounds 1-2 add pseudo-labels)
- **Pseudo-label threshold:** score ≥ 0.85 → positive, score ≤ 0.15 → negative
- **Pair generation for fine-tuning:** 3 000 pairs per round (50 % same-class, 50 % cross-class)
- **Fine-tuning:** 3 epochs, batch size 8, lr 2e-5, cosine similarity loss

## Output

`result.csv` — one row per test sample:

| Column | Description |
|--------|-------------|
| `row_id` | Original test row identifier |
| `LABEL` | Predicted label (0 = negative, 1 = positive) |

## Result

- kaggle public(33%): acc 0.8379
- kaggle private(67%): acc 0.79724