"""
End-to-End Sentiment Scoring Pipeline  —  v4b  (pre-split CSV variant)
=======================================================================
Difference from v4 (test.py):
  - Text splitting (Step 1 / Stanza) is REMOVED.
  - Input CSV must already contain SPLIT_TEXT (a Python list as string)
    and NUM_SPLITS columns produced by an external splitting script.
  - Supported inputs:
      dataset_split/train_2022_split_stanza_v1.csv  (transition-word split)
      dataset_split/train_2022_split_stanza_v2.csv  (alternative split)

  For each row the pipeline reads SPLIT_TEXT, embeds every segment with
  Sentence-BERT, then aggregates:
      agg_emb = Sum_i( emb_i * weight(i) )
  After all rows, Step 2 — regression on aggregated 384-dim features.

Usage:
    python test_presplit.py --input dataset_split/train_2022_split_stanza_v1.csv
    python test_presplit.py --input dataset_split/train_2022_split_stanza_v2.csv --model gbm
"""

import argparse
import ast
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
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


# ── Constants ─────────────────────────────────────────────────────────────────

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_DIM        = 384

AGG_FEATURES  = [f"emb_{i}" for i in range(SBERT_DIM)]
WEIGHT_SCHEMES = ("sqrt", "uniform", "linear", "log", "decay", "last", "contrast")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load pre-split CSV  (replaces Stanza splitting)
# ══════════════════════════════════════════════════════════════════════════════

def load_presplit_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV that already has SPLIT_TEXT (list as string) and NUM_SPLITS.
    SPLIT_TEXT is parsed from its string representation into an actual list.
    """
    df = pd.read_csv(path)

    required = {"row_id", "TEXT", "LABEL", "SPLIT_TEXT", "NUM_SPLITS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    df["row_id"]    = df["row_id"].astype(int)
    df["LABEL"]     = df["LABEL"].astype(int)
    df["NUM_SPLITS"] = df["NUM_SPLITS"].astype(int)

    # Parse SPLIT_TEXT from string → Python list
    def _parse(val):
        if isinstance(val, list):
            return val
        try:
            result = ast.literal_eval(str(val))
            return result if isinstance(result, list) else [str(val)]
        except Exception:
            return [str(val)]

    df["SPLIT_TEXT"] = df["SPLIT_TEXT"].apply(_parse)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Sentence-BERT embedding
# ══════════════════════════════════════════════════════════════════════════════

def _build_sbert_model() -> SentenceTransformer:
    print(f"  Loading Sentence-BERT model '{SBERT_MODEL_NAME}' …")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    print(f"  Embedding dim: {SBERT_DIM}")
    return model


def _get_embedding(sbert: SentenceTransformer, text: str) -> np.ndarray:
    emb = sbert.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE — weighted sum of per-segment embeddings
# ══════════════════════════════════════════════════════════════════════════════

def _compute_weights(n: int, scheme: str) -> list:
    """
    Schemes
    -------
    sqrt     : sqrt(i+1)                — gentle ramp-up
    uniform  : 1                        — all equal
    linear   : i+1                      — strongly favours end
    log      : log(i+2)                 — flatter ramp than sqrt
    decay    : exp(-0.5*i)              — exponential decay; favours opening
    last     : 1 except last seg = 3    — emphasises conclusion
    contrast : first=2, last=2, mid=0.5 — highlights framing and conclusion
    """
    if scheme == "uniform":
        return [1.0] * n
    if scheme == "linear":
        return [float(i + 1) for i in range(n)]
    if scheme == "sqrt":
        return [np.sqrt(i + 1) for i in range(n)]
    if scheme == "log":
        return [np.log(i + 2) for i in range(n)]
    if scheme == "decay":
        return [np.exp(-0.5 * i) for i in range(n)]
    if scheme == "last":
        w = [1.0] * n
        if n > 0:
            w[-1] = 3.0
        return w
    if scheme == "contrast":
        if n == 1:
            return [2.0]
        w = [0.5] * n
        w[0]  = 2.0
        w[-1] = 2.0
        return w
    raise ValueError(f"Unknown weight scheme '{scheme}'. Choose from: {WEIGHT_SCHEMES}")


def process_row(segments: list, label: int, row_id: int,
                sbert: SentenceTransformer,
                weight_scheme: str = "sqrt") -> dict:
    """
    Embed each segment in SPLIT_TEXT and compute the weighted sum.
    Returns a flat dict: row_id, LABEL, NUM_SPLITS, emb_0 … emb_383
    """
    n = len(segments)
    weights = _compute_weights(n, weight_scheme)

    agg_emb = np.zeros(SBERT_DIM, dtype=np.float32)
    for i, seg in enumerate(segments):
        agg_emb += _get_embedding(sbert, str(seg)) * weights[i]

    result = {"row_id": row_id, "LABEL": label, "NUM_SPLITS": n}
    for j, val in enumerate(agg_emb):
        result[f"emb_{j}"] = round(float(val), 6)
    # print(f"    Processed row_id={row_id}  |  LABEL={label}  |  NUM_SPLITS={n}")
    return result


def run_row_by_row(df: pd.DataFrame, sbert: SentenceTransformer,
                   weight_scheme: str = "sqrt",
                   seg_emb_path: str | None = None) -> pd.DataFrame:
    """
    Embed every row using its pre-split SPLIT_TEXT segments.

    If seg_emb_path is given, also saves a per-segment embedding CSV with columns:
        row_id, LABEL, seg_idx, segment_text, emb_0 … emb_383
    """
    total = len(df)
    print(f"  Processing {total} rows (Sentence-BERT embedding per segment) …")

    records     = []
    seg_records = [] if seg_emb_path else None

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f"    progress: {i}/{total}")

        row_id   = int(row["row_id"])
        label    = int(row["LABEL"])
        segments = row["SPLIT_TEXT"]
        n        = len(segments)
        weights  = _compute_weights(n, weight_scheme)

        agg_emb = np.zeros(SBERT_DIM, dtype=np.float32)
        for seg_idx, seg in enumerate(segments):
            emb      = _get_embedding(sbert, str(seg))
            agg_emb += emb * weights[seg_idx]

            if seg_records is not None:
                rec = {"row_id": row_id, "LABEL": label,
                       "seg_idx": seg_idx, "segment_text": str(seg)}
                for j, v in enumerate(emb):
                    rec[f"emb_{j}"] = round(float(v), 6)
                seg_records.append(rec)

        result = {"row_id": row_id, "LABEL": label, "NUM_SPLITS": n}
        for j, val in enumerate(agg_emb):
            result[f"emb_{j}"] = round(float(val), 6)
        records.append(result)

    cols      = ["row_id", "LABEL", "NUM_SPLITS"] + AGG_FEATURES
    result_df = pd.DataFrame(records)[cols]
    print(f"  Done — {len(result_df)} aggregated rows")

    if seg_records is not None and seg_emb_path:
        seg_cols = ["row_id", "LABEL", "seg_idx", "segment_text"] + AGG_FEATURES
        seg_df   = pd.DataFrame(seg_records)[seg_cols]
        seg_df.to_csv(seg_emb_path, index=False)
        print(f"  Per-segment embeddings saved → {seg_emb_path}  ({len(seg_df)} segments)")

    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Regression scoring (5-fold CV)
# ══════════════════════════════════════════════════════════════════════════════

def _make_model(model_name: str):
    model_map = {
        "rf":      RandomForestRegressor(n_estimators=100, random_state=42),
        "elastic": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "gbm":     GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(model_map)}")
    return model_map[model_name]


def _print_fold_metrics(fold: int, y_true: np.ndarray, y_pred_label: np.ndarray,
                        y_score: np.ndarray) -> dict:
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
    5-fold stratified cross-validation on the aggregated SBERT feature matrix.
    Returns: (output_df, trained_pipe_on_full_data, feat_cols)
    """
    print(f"  Model: {model_name}  |  Features: {SBERT_DIM}-dim SBERT embeddings")
    print(f"  Running {n_splits}-fold stratified cross-validation …")

    X = agg_df[AGG_FEATURES].values
    y = agg_df["LABEL"].values

    skf          = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    oof_scores   = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        pipe = Pipeline([("scaler", StandardScaler()), ("model", _make_model(model_name))])
        pipe.fit(X[train_idx], y[train_idx])
        scores              = np.clip(pipe.predict(X[val_idx]), 0, 1)
        pred_labels         = (scores >= 0.5).astype(int)
        oof_scores[val_idx] = scores
        fold_metrics.append(_print_fold_metrics(fold, y[val_idx], pred_labels, scores))

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
    return output, final_pipe, AGG_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment pipeline v4b — uses pre-split CSV (no Stanza)"
    )
    parser.add_argument(
        "--input", default="dataset_split/train_2022_split_stanza_v4.csv",
        help="Pre-split CSV with columns: row_id, TEXT, LABEL, SPLIT_TEXT, NUM_SPLITS"
    )
    parser.add_argument("--output", default="row_scores_v4b.csv",
                        help="Output CSV with row-level scores")
    parser.add_argument("--model", default="rf",
                        choices=["rf", "elastic", "gbm"],
                        help="Regression model (default: rf)")
    parser.add_argument("--weight", default=None,
                        choices=list(WEIGHT_SCHEMES) + ["all"],
                        help="Segment weighting scheme. Use 'all' to run every scheme (default: all)")
    parser.add_argument("--seg_emb", default=None,
                        help="If set, save per-segment SBERT embeddings to this CSV path")
    args = parser.parse_args()

    # 未指定時預設跑全部
    run_all    = (args.weight is None or args.weight == "all")
    schemes    = list(WEIGHT_SCHEMES) if run_all else [args.weight]

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Sentiment Scoring Pipeline  [v4b — pre-split]")
    print(sep)
    print(f"  Input    : {args.input}")
    print(f"  Output   : {args.output}  (suffix _<scheme> added per run)")
    print(f"  Model    : {args.model}")
    print(f"  Encoder  : {SBERT_MODEL_NAME}  ({SBERT_DIM}-dim)")
    print(f"  Schemes  : {schemes}\n")

    # ── Load pre-split CSV ────────────────────────────────────────────────────
    print("── Loading pre-split CSV ───────────────────────────────────")
    df = load_presplit_csv(args.input)
    print(f"  Loaded {len(df)} rows | unique row_ids: {df['row_id'].nunique()}")
    split_counts = df["NUM_SPLITS"].value_counts().sort_index()
    print(f"  NUM_SPLITS distribution:\n{split_counts.to_string()}\n")

    # ── Build Sentence-BERT model once ───────────────────────────────────────
    print("── Initialising Sentence-BERT ──────────────────────────────")
    sbert = _build_sbert_model()

    # ── Pre-compute per-segment embeddings once (weight-independent) ──────────
    # All schemes share the same raw embeddings; only the aggregation differs.
    print("\n── Pre-computing per-segment SBERT embeddings (once) ───────")
    # Store as list-of-arrays: seg_embs[i] = list of np.ndarray for row i
    seg_embs: list[list[np.ndarray]] = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f"    progress: {i}/{total}")
        seg_embs.append([_get_embedding(sbert, str(seg)) for seg in row["SPLIT_TEXT"]])

    if args.seg_emb:
        seg_cols = ["row_id", "LABEL", "seg_idx", "segment_text"] + AGG_FEATURES
        seg_records = []
        for i, (_, row) in enumerate(df.iterrows()):
            for seg_idx, (seg, emb) in enumerate(zip(row["SPLIT_TEXT"], seg_embs[i])):
                rec = {"row_id": int(row["row_id"]), "LABEL": int(row["LABEL"]),
                       "seg_idx": seg_idx, "segment_text": str(seg)}
                for j, v in enumerate(emb):
                    rec[f"emb_{j}"] = round(float(v), 6)
                seg_records.append(rec)
        pd.DataFrame(seg_records)[seg_cols].to_csv(args.seg_emb, index=False)
        print(f"  Per-segment embeddings saved → {args.seg_emb}  ({len(seg_records)} segments)")

    # ── Run each weight scheme ────────────────────────────────────────────────
    summary_rows = []

    for scheme in schemes:
        print(f"\n{'═'*60}")
        print(f"  Weight scheme: {scheme}")
        print(f"{'═'*60}")

        # Aggregate embeddings with this scheme
        records = []
        for i, (_, row) in enumerate(df.iterrows()):
            embs    = seg_embs[i]
            n       = len(embs)
            weights = _compute_weights(n, scheme)
            agg_emb = sum(e * w for e, w in zip(embs, weights))
            result  = {"row_id": int(row["row_id"]), "LABEL": int(row["LABEL"]), "NUM_SPLITS": n}
            for j, val in enumerate(agg_emb):
                result[f"emb_{j}"] = round(float(val), 6)
            records.append(result)

        cols   = ["row_id", "LABEL", "NUM_SPLITS"] + AGG_FEATURES
        agg_df = pd.DataFrame(records)[cols]

        # 5-Fold CV
        print(f"\n── Step 3: 5-Fold CV + Regression Scoring ──────────────────")
        row_scores, _, _ = run_cross_validation(agg_df, args.model)

        # Save per-scheme output
        base, ext      = args.output.rsplit(".", 1) if "." in args.output else (args.output, "csv")
        scheme_path    = f"{base}_{scheme}.{ext}"
        row_scores.to_csv(scheme_path, index=False)
        print(f"  Saved → {scheme_path}")

        # Collect OOF summary metrics for final comparison table
        y_true  = agg_df["LABEL"].values
        y_pred  = (row_scores["oof_score"].values >= 0.5).astype(int)
        y_score = row_scores["oof_score"].values
        summary_rows.append({
            "scheme":    scheme,
            "acc":       round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"), 4),
            "mse":       round(mean_squared_error(y_true, y_score), 4),
            "output":    scheme_path,
        })

    # ── Final comparison table ────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{'═'*60}")
    print("  ── All-scheme OOF Comparison ───────────────────────────────")
    print(summary_df.to_string(index=False))

    summary_path = (args.output.rsplit(".", 1)[0] if "." in args.output else args.output) + "_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
