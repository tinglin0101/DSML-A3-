"""
End-to-End Sentiment Scoring Pipeline  —  v6a-finetune
=======================================================
v6a-finetune: fine-tunes all-mpnet-base-v2 on the training labels
  BEFORE extracting embeddings, using sentence_transformers native training.

  Fine-tuning strategy: CosineSimilarityLoss on sampled pairs.
    - Same label  → target similarity = 1.0
    - Diff label  → target similarity = 0.0
  This teaches the encoder to pull same-sentiment sentences together
  and push opposite-sentiment sentences apart.

Pipeline:
  原始文字
    │
    ▼
  文字切割（Stanza）
    │
    ▼
  微調 all-mpnet-base-v2（CosineSimilarityLoss, N epochs）
    │
    ▼
  用微調後模型產生 768-dim embedding — weighted-sum aggregation
    │
    ▼
  5-fold CV（RF / GBM / ElasticNet）+ 回歸評分
    │
    ▼
  對 test set 推論 → result_v6a-finetune.csv

Usage:
    python v6a-finetune.py
    python v6a-finetune.py --model gbm --ft-epochs 5 --ft-pairs 4000
"""

import argparse
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import stanza
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, mean_squared_error,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ── Constants ─────────────────────────────────────────────────────────────────

SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SBERT_DIM        = 768

VADER_FEATURES = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
AGG_FEATURES   = [f"emb_{i}" for i in range(SBERT_DIM)]
ALL_FEATURES   = AGG_FEATURES + VADER_FEATURES

WEIGHT_SCHEMES = ("sqrt", "uniform", "linear", "log", "decay", "last", "contrast")

TRANSITION_WORDS = {
    "but", "however", "although", "though", "yet", "nevertheless",
    "nonetheless", "except", "while", "whereas",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Text splitting
# ══════════════════════════════════════════════════════════════════════════════

def _build_stanza_pipeline() -> stanza.Pipeline:
    stanza.download("en", verbose=False)
    return stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse", verbose=False)


def split_text(text: str, nlp: stanza.Pipeline) -> list[str]:
    if not isinstance(text, str):
        return [text]

    doc = nlp(text)
    split_indices = []

    for sentence in doc.sentences:
        words = sentence.words
        for word in words:
            word_lower  = word.text.lower()
            lemma_lower = word.lemma.lower() if word.lemma else ""
            start_char  = getattr(word, "start_char", 0)

            if word_lower in TRANSITION_WORDS or lemma_lower in TRANSITION_WORDS:
                if getattr(word, "upos", "") in ("CCONJ", "SCONJ", "ADV"):
                    if start_char > 0:
                        split_indices.append(start_char)
                continue

            if word_lower == "and":
                head_id = word.head if word.head is not None else 0
                if head_id > 0:
                    head_word = words[head_id - 1]
                    if getattr(head_word, "upos", "") in ("VERB", "AUX"):
                        if start_char > 0:
                            split_indices.append(start_char)

    split_indices = sorted(set(split_indices))

    parts, last_idx = [], 0
    for idx in split_indices:
        seg = text[last_idx:idx].strip()
        if seg:
            parts.append(seg)
        last_idx = idx

    final = text[last_idx:].strip()
    if final:
        parts.append(final)

    return parts or [text]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Fine-tune Sentence-BERT
# ══════════════════════════════════════════════════════════════════════════════

def _build_sbert_model() -> SentenceTransformer:
    print(f"  Loading Sentence-BERT '{SBERT_MODEL_NAME}' …")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    print(f"  Embedding dim : {SBERT_DIM}")
    return model


def _sample_pairs(df: pd.DataFrame, n_pairs: int, seed: int = 42) -> list[InputExample]:
    """
    Sample n_pairs InputExample objects for CosineSimilarityLoss.
    Same label → label=1.0, different label → label=0.0.
    Balanced 50/50 between positive and negative pairs.
    """
    rng      = random.Random(seed)
    texts    = df["TEXT"].astype(str).tolist()
    labels   = df["LABEL"].astype(int).tolist()

    pos_idx  = [i for i, l in enumerate(labels) if l == 1]
    neg_idx  = [i for i, l in enumerate(labels) if l == 0]

    examples = []
    half     = n_pairs // 2

    # same-sentiment pairs (target=1.0)
    for _ in range(half):
        a, b = rng.sample(pos_idx if rng.random() < 0.5 else neg_idx, 2)
        examples.append(InputExample(texts=[texts[a], texts[b]], label=1.0))

    # opposite-sentiment pairs (target=0.0)
    for _ in range(n_pairs - half):
        a = rng.choice(pos_idx)
        b = rng.choice(neg_idx)
        examples.append(InputExample(texts=[texts[a], texts[b]], label=0.0))

    rng.shuffle(examples)
    return examples


def finetune_sbert(
    df: pd.DataFrame,
    sbert: SentenceTransformer,
    n_pairs: int    = 3000,
    epochs: int     = 3,
    batch_size: int = 16,
    lr: float       = 2e-5,
    warmup_steps: int = 100,
    output_path: str | None = None,
) -> None:
    """
    Fine-tune *sbert* in-place using CosineSimilarityLoss on sampled text pairs.
    Modifies the model weights; no return value.
    """
    print(f"  Sampling {n_pairs} text pairs for fine-tuning …")
    examples   = _sample_pairs(df, n_pairs)
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss_fn    = losses.CosineSimilarityLoss(sbert)

    total_steps = len(dataloader) * epochs
    print(f"  Fine-tuning: {n_pairs} pairs, {epochs} epochs, "
          f"lr={lr}, batch={batch_size}, warmup={warmup_steps}")

    sbert.fit(
        train_objectives=[(dataloader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=output_path,
        show_progress_bar=True,
    )
    print("  Fine-tuning complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Embedding + VADER helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def _get_vader_scores(vader: SentimentIntensityAnalyzer, text: str) -> dict:
    s = vader.polarity_scores(text)
    return {"vader_neg": s["neg"], "vader_neu": s["neu"],
            "vader_pos": s["pos"], "vader_compound": s["compound"]}


def _get_embedding(sbert: SentenceTransformer, text: str) -> np.ndarray:
    emb = sbert.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Weight schemes
# ══════════════════════════════════════════════════════════════════════════════

def _compute_weights(n: int, scheme: str) -> list:
    if scheme == "uniform":  return [1.0] * n
    if scheme == "linear":   return [float(i + 1) for i in range(n)]
    if scheme == "sqrt":     return [np.sqrt(i + 1) for i in range(n)]
    if scheme == "log":      return [np.log(i + 2) for i in range(n)]
    if scheme == "decay":    return [np.exp(-0.5 * i) for i in range(n)]
    if scheme == "last":
        w = [1.0] * n
        if n > 0: w[-1] = 3.0
        return w
    if scheme == "contrast":
        if n == 1: return [2.0]
        w = [0.5] * n; w[0] = 2.0; w[-1] = 2.0
        return w
    raise ValueError(f"Unknown weight scheme '{scheme}'.")


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE — Steps 1+embedding per row, then weighted sum
# ══════════════════════════════════════════════════════════════════════════════

def process_row(text: str, label: int, row_id: int,
                nlp: stanza.Pipeline, sbert: SentenceTransformer,
                vader: SentimentIntensityAnalyzer,
                weight_scheme: str = "sqrt") -> dict:
    segments = split_text(text, nlp)
    n        = len(segments)
    weights  = _compute_weights(n, weight_scheme)

    agg_emb = np.zeros(SBERT_DIM, dtype=np.float32)
    for i, seg in enumerate(segments):
        agg_emb += _get_embedding(sbert, seg) * weights[i]

    result = {"row_id": row_id, "LABEL": label, "NUM_SPLITS": n}
    for j, val in enumerate(agg_emb):
        result[f"emb_{j}"] = round(float(val), 6)
    result.update(_get_vader_scores(vader, text))
    return result


def run_row_by_row(df: pd.DataFrame,
                   nlp: stanza.Pipeline,
                   sbert: SentenceTransformer,
                   vader: SentimentIntensityAnalyzer,
                   has_label: bool = True,
                   weight_scheme: str = "sqrt") -> pd.DataFrame:
    total = len(df)
    print(f"  Extracting embeddings for {total} rows …")

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f"    progress: {i}/{total}")
        label = int(row["LABEL"]) if has_label else -1
        records.append(
            process_row(str(row["TEXT"]), label, int(row["row_id"]),
                        nlp, sbert, vader, weight_scheme=weight_scheme)
        )

    cols = ["row_id", "LABEL", "NUM_SPLITS"] + AGG_FEATURES + VADER_FEATURES
    result_df = pd.DataFrame(records)[cols]
    print(f"  Done — {len(result_df)} rows embedded")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — 5-Fold CV + Regression Scoring
# ══════════════════════════════════════════════════════════════════════════════

def _make_model(model_name: str):
    model_map = {
        "rf":      RandomForestRegressor(n_estimators=100, random_state=42),
        "elastic": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "gbm":     GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'.")
    return model_map[model_name]


def _print_fold_metrics(fold: int, y_true, y_pred_label, y_score) -> dict:
    cm  = confusion_matrix(y_true, y_pred_label)
    acc = accuracy_score(y_true, y_pred_label)
    pre = precision_score(y_true, y_pred_label, zero_division=0)
    rec = recall_score(y_true, y_pred_label, zero_division=0)
    f1  = f1_score(y_true, y_pred_label, zero_division=0)
    mse = mean_squared_error(y_true, y_score)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    print(f"\n  ── Fold {fold} ──────────────────────────────")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {pre:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"  AUC      : {auc:.4f}  MSE: {mse:.4f}")
    return dict(acc=acc, pre=pre, rec=rec, f1=f1, auc=auc, mse=mse, cm=cm)


def run_cross_validation(agg_df: pd.DataFrame, model_name: str, n_splits: int = 5):
    """
    5-fold stratified CV → OOF metrics, then retrain on full data.
    Returns (output_df, trained_pipe, feat_cols).
    """
    print(f"  Model: {model_name}  |  Features: {SBERT_DIM}-dim SBERT + 4 VADER")
    print(f"  Running {n_splits}-fold stratified cross-validation …")

    X = agg_df[ALL_FEATURES].values
    y = agg_df["LABEL"].values

    skf          = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    oof_scores   = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        pipe = Pipeline([("scaler", StandardScaler()), ("model", _make_model(model_name))])
        pipe.fit(X[train_idx], y[train_idx])
        scores              = np.clip(pipe.predict(X[val_idx]), 0, 1)
        oof_scores[val_idx] = scores
        fold_metrics.append(_print_fold_metrics(fold, y[val_idx], (scores >= 0.5).astype(int), scores))

    oof_labels = (oof_scores >= 0.5).astype(int)
    print("\n  ══ Overall OOF (out-of-fold) ═══════════════════════")
    print(f"  Confusion Matrix:\n{confusion_matrix(y, oof_labels)}")
    for metric in ("acc", "pre", "rec", "f1", "auc", "mse"):
        vals = [m[metric] for m in fold_metrics]
        print(f"  {metric.upper():9s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    print("\n  Retraining on full dataset …")
    final_pipe = Pipeline([("scaler", StandardScaler()), ("model", _make_model(model_name))])
    final_pipe.fit(X, y)
    final_scores = np.clip(final_pipe.predict(X), 0, 1)

    output = agg_df[["row_id", "LABEL"]].copy().rename(columns={"LABEL": "row_label"})
    output["oof_score"]       = oof_scores
    output["oof_pred_label"]  = oof_labels
    output["final_score"]     = final_scores
    output["predicted_label"] = (final_scores >= 0.5).astype(int)
    return output, final_pipe, ALL_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    path = "D:\\[課程]DSML\\DSML-A3-\\"
    parser = argparse.ArgumentParser(
        description="Sentiment pipeline v6a-finetune: fine-tunes all-mpnet-base-v2 first")
    parser.add_argument("--input",       default=path + "train_2022.csv")
    parser.add_argument("--output",      default=path + "row_scores_v6a-finetune.csv")
    parser.add_argument("--model",       default="rf", choices=["rf", "elastic", "gbm"])
    parser.add_argument("--test",        default=path + "test_no_answer_2022.csv")
    parser.add_argument("--test-output", default=path + "result_v6a-finetune.csv")
    parser.add_argument("--weight",      default="sqrt", choices=list(WEIGHT_SCHEMES))
    # Fine-tuning args
    parser.add_argument("--ft-epochs",   default=3,    type=int,
                        help="Fine-tuning epochs (default: 3)")
    parser.add_argument("--ft-lr",       default=2e-5, type=float,
                        help="Fine-tuning learning rate (default: 2e-5)")
    parser.add_argument("--ft-batch",    default=16,   type=int,
                        help="Fine-tuning batch size (default: 16)")
    parser.add_argument("--ft-pairs",    default=3000, type=int,
                        help="Number of sentence pairs for CosineSimilarityLoss (default: 3000)")
    parser.add_argument("--ft-warmup",   default=100,  type=int,
                        help="Warmup steps for fine-tuning (default: 100)")
    parser.add_argument("--ft-save",     default=None, type=str,
                        help="Path to save fine-tuned SBERT model (optional)")
    args = parser.parse_args()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Sentiment Scoring Pipeline  [v6a-finetune]")
    print(sep)
    print(f"  Input      : {args.input}")
    print(f"  Test output: {args.test_output}")
    print(f"  Model      : {args.model}")
    print(f"  Encoder    : {SBERT_MODEL_NAME}  ({SBERT_DIM}-dim, fine-tuned)")
    print(f"  FT epochs  : {args.ft_epochs}  lr={args.ft_lr}  batch={args.ft_batch}")
    print(f"  FT pairs   : {args.ft_pairs}  warmup={args.ft_warmup}")
    print(f"  Weighting  : {args.weight}\n")

    # ── Load training data ────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    df["row_id"] = df["row_id"].astype(int)
    df["LABEL"]  = df["LABEL"].astype(int)
    print(f"  Loaded {len(df)} training rows\n")

    # ── Initialise Stanza + VADER ─────────────────────────────────────────────
    print("── Initialising models ─────────────────────────────────────")
    nlp   = _build_stanza_pipeline()
    vader = _build_vader()

    # ── Load + Fine-tune SBERT ────────────────────────────────────────────────
    sbert = _build_sbert_model()
    print("\n── Fine-tuning all-mpnet-base-v2 ───────────────────────────")
    finetune_sbert(
        df, sbert,
        n_pairs    = args.ft_pairs,
        epochs     = args.ft_epochs,
        batch_size = args.ft_batch,
        lr         = args.ft_lr,
        warmup_steps = args.ft_warmup,
        output_path  = args.ft_save,
    )

    # ── Steps 1+2: row-by-row split → fine-tuned SBERT → weighted sum ────────
    print("\n── Steps 1+2: Split + Embedding (fine-tuned SBERT) ─────────")
    agg_df = run_row_by_row(df, nlp, sbert, vader,
                            has_label=True, weight_scheme=args.weight)

    # ── Step 3: 5-fold CV + regression scoring ────────────────────────────────
    print("\n── Step 3: 5-Fold CV + Regression Scoring ──────────────────")
    row_scores, trained_pipe, feat_cols = run_cross_validation(agg_df, args.model, n_splits=5)

    row_scores.to_csv(args.output, index=False)
    print(f"\n  Saved training results to: {args.output}")

    # ── Process test set ──────────────────────────────────────────────────────
    print(f"\n── Processing test set: {args.test} ────────────────────────")
    test_df = pd.read_csv(args.test)
    test_df["row_id"] = test_df["row_id"].astype(int)
    if "LABEL" not in test_df.columns:
        test_df["LABEL"] = -1
    print(f"  Loaded {len(test_df)} test rows\n")

    test_agg_df = run_row_by_row(test_df, nlp, sbert, vader,
                                 has_label=False, weight_scheme=args.weight)

    X_test      = test_agg_df[feat_cols].values
    test_scores = np.clip(trained_pipe.predict(X_test), 0, 1)
    test_labels = (test_scores >= 0.5).astype(int)

    test_output = test_agg_df[["row_id"]].copy()
    test_output["LABEL"] = test_labels
    test_output.to_csv(args.test_output, index=False)
    print(f"  Saved test predictions to: {args.test_output}")


if __name__ == "__main__":
    main()
