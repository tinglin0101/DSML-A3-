"""
End-to-End Sentiment Scoring Pipeline  —  v5
=============================================
v5 change: upgraded Sentence-BERT model from all-MiniLM-L6-v2 to all-mpnet-base-v2.
  Model: sentence-transformers/all-mpnet-base-v2
  Outputs a 768-dim dense embedding per segment.

  For each row:
    Step 1 — split text at transition words (Stanza)
    Step 2 — Sentence-BERT embedding per segment  (768-dim)
    Aggregate — Sum( emb_i * weight(i) )  where weight(i) = sqrt(i+1)
  After all rows are aggregated, Step 3 — regression on aggregated features.

Usage:
    python pipeline_v5.py --input dataset_split/train_2022.csv --output row_scores.csv
    python pipeline_v5.py --input dataset_split/train_2022.csv --model rf
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import stanza
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, mean_squared_error,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ── Constants ─────────────────────────────────────────────────────────────────

SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SBERT_DIM        = 768          # embedding dimension for all-mpnet-base-v2

VADER_FEATURES = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]

# Aggregated feature columns: one per embedding dimension
AGG_FEATURES = [f"emb_{i}" for i in range(SBERT_DIM)]

# Combined feature columns fed to the classifier
ALL_FEATURES = AGG_FEATURES + VADER_FEATURES

WEIGHT_SCHEMES = ("sqrt", "uniform", "linear", "log", "decay", "last", "contrast")

TRANSITION_WORDS = {
    "but", "however", "although", "though", "yet", "nevertheless",
    "nonetheless", "except", "while", "whereas",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Text splitting  (unchanged from v3)
# ══════════════════════════════════════════════════════════════════════════════

def _build_stanza_pipeline() -> stanza.Pipeline:
    stanza.download("en", verbose=False)
    return stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse", verbose=False)


def split_text(text: str, nlp: stanza.Pipeline) -> list[str]:
    """Split a single text string at transition words / clause-linking 'and'."""
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
# STEP 2 — Sentence-BERT embedding (single segment)
# ══════════════════════════════════════════════════════════════════════════════

def _build_sbert_model() -> SentenceTransformer:
    """Load Sentence-BERT once; uses GPU automatically if available."""
    print(f"  Loading Sentence-BERT model '{SBERT_MODEL_NAME}' …")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    print(f"  Embedding dim : {SBERT_DIM}")
    return model


def _build_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def _get_vader_scores(vader: SentimentIntensityAnalyzer, text: str) -> dict:
    """Return neg/neu/pos/compound scores for the full original text."""
    scores = vader.polarity_scores(text)
    return {
        "vader_neg":      scores["neg"],
        "vader_neu":      scores["neu"],
        "vader_pos":      scores["pos"],
        "vader_compound": scores["compound"],
    }


def _get_embedding(sbert: SentenceTransformer, text: str) -> np.ndarray:
    """Return a normalised 384-dim embedding for one text segment."""
    emb = sbert.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE — Steps 1+2 per row, then weighted sum
# ══════════════════════════════════════════════════════════════════════════════

def _compute_weights(n: int, scheme: str) -> list:
    """
    Return a list of n weights for segment positions 0..n-1.

    Schemes
    -------
    sqrt     : sqrt(i+1)                — current default; gentle ramp-up
    uniform  : 1                        — all segments equally important
    linear   : i+1                      — linearly increasing; strongly favours end
    log      : log(i+2)                 — flatter ramp than sqrt
    decay    : exp(-0.5*i)              — exponential decay; favours opening clause
    last     : 1 except last seg = 3    — emphasises the concluding segment
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


def process_row(text: str, label: int, row_id: int,
                nlp: stanza.Pipeline, sbert: SentenceTransformer,
                vader: SentimentIntensityAnalyzer,
                weight_scheme: str = "sqrt") -> dict:
    """
    Run Step 1 (split) + Step 2 (Sentence-BERT) on a single row, then aggregate:
        agg_emb = Sum_i( emb_i * weight(i) )
    Also appends VADER scores computed on the full original text.

    Returns a flat dict:  row_id, LABEL, NUM_SPLITS, emb_0…emb_383, vader_neg/neu/pos/compound
    """
    # Step 1 — split
    segments = split_text(text, nlp)
    n = len(segments)

    weights = _compute_weights(n, weight_scheme)

    # Step 2 + aggregate
    agg_emb = np.zeros(SBERT_DIM, dtype=np.float32)
    for i, seg in enumerate(segments):
        emb = _get_embedding(sbert, seg)
        agg_emb += emb * weights[i]

    result = {"row_id": row_id, "LABEL": label, "NUM_SPLITS": n}
    for j, val in enumerate(agg_emb):
        result[f"emb_{j}"] = round(float(val), 6)

    # VADER on the full sentence
    result.update(_get_vader_scores(vader, text))
    return result


def run_row_by_row(df: pd.DataFrame, nlp: stanza.Pipeline,
                   sbert: SentenceTransformer,
                   vader: SentimentIntensityAnalyzer,
                   has_label: bool = True,
                   weight_scheme: str = "sqrt") -> pd.DataFrame:
    """
    Process every row through Steps 1–2 individually and return
    one aggregated-feature row per original row.
    """
    total = len(df)
    print(f"  Processing {total} rows (step 1+2 per row) …")

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f"    progress: {i}/{total}")
        label = int(row["LABEL"]) if has_label else -1
        records.append(
            process_row(str(row["TEXT"]), label, int(row["row_id"]), nlp, sbert, vader,
                        weight_scheme=weight_scheme)
        )

    cols = ["row_id", "LABEL", "NUM_SPLITS"] + AGG_FEATURES + VADER_FEATURES
    result_df = pd.DataFrame(records)[cols]
    print(f"  Done — {len(result_df)} aggregated rows")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Regression scoring
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
    Prints per-fold confusion matrix + metrics, then the mean ± std summary.

    Returns: (output_df, trained_pipe_on_full_data, feat_cols)
    """
    print(f"  Model: {model_name}  |  Features: {SBERT_DIM}-dim SBERT + 4 VADER")
    print(f"  Running {n_splits}-fold stratified cross-validation …")

    feat_cols = ALL_FEATURES
    X = agg_df[feat_cols].values
    y = agg_df["LABEL"].values

    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    oof_scores   = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        pipe = Pipeline([("scaler", StandardScaler()), ("model", _make_model(model_name))])
        pipe.fit(X[train_idx], y[train_idx])
        scores        = np.clip(pipe.predict(X[val_idx]), 0, 1)
        pred_labels   = (scores >= 0.5).astype(int)
        oof_scores[val_idx] = scores
        fold_metrics.append(_print_fold_metrics(fold, y[val_idx], pred_labels, scores))

    # ── Overall OOF metrics ───────────────────────────────────────────────────
    oof_labels = (oof_scores >= 0.5).astype(int)
    print("\n  ══ Overall OOF (out-of-fold) ═══════════════════════")
    print(f"  Confusion Matrix:\n{confusion_matrix(y, oof_labels)}")
    for metric in ("acc", "pre", "rec", "f1", "auc", "mse"):
        vals = [m[metric] for m in fold_metrics]
        print(f"  {metric.upper():9s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    # ── Retrain on full data ──────────────────────────────────────────────────
    print("\n  Retraining on full dataset …")
    final_pipe = Pipeline([("scaler", StandardScaler()), ("model", _make_model(model_name))])
    final_pipe.fit(X, y)
    final_scores  = np.clip(final_pipe.predict(X), 0, 1)

    output = agg_df[["row_id", "LABEL"]].copy().rename(columns={"LABEL": "row_label"})
    output["oof_score"]       = oof_scores
    output["oof_pred_label"]  = oof_labels
    output["final_score"]     = final_scores
    output["predicted_label"] = (final_scores >= 0.5).astype(int)
    return output, final_pipe, feat_cols


def run_regression(agg_df: pd.DataFrame, model_name: str):
    """
    Train a regression model on the aggregated Sentence-BERT feature matrix.

    Returns: (output_df, trained_pipe, feat_cols)
    """
    print(f"  Model: {model_name}  |  Features: {SBERT_DIM}-dim SBERT + 4 VADER")

    feat_cols = ALL_FEATURES
    X = agg_df[feat_cols].values
    y = agg_df["LABEL"].values

    pipe = Pipeline([("scaler", StandardScaler()), ("model", _make_model(model_name))])

    if len(agg_df) >= 10:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        pipe.fit(X_train, y_train)
        val_scores = np.clip(pipe.predict(X_val), 0, 1)
        mse = mean_squared_error(y_val, val_scores)
        try:
            auc = roc_auc_score(y_val, val_scores)
            print(f"  [Regression] train={len(X_train)} | val={len(X_val)} | "
                  f"MSE={mse:.4f} | AUC={auc:.4f}")
        except Exception:
            print(f"  [Regression] train={len(X_train)} | val={len(X_val)} | MSE={mse:.4f}")
        final_scores = np.clip(pipe.predict(X), 0, 1)
    else:
        pipe.fit(X, y)
        final_scores = np.clip(pipe.predict(X), 0, 1)
        mse = mean_squared_error(y, final_scores)
        print(f"  [Regression] rows={len(agg_df)} | MSE={mse:.4f} (too few for split)")

    output = agg_df[["row_id", "LABEL"]].copy().rename(columns={"LABEL": "row_label"})
    output["final_score"]     = final_scores
    output["predicted_label"] = (final_scores >= 0.5).astype(int)
    return output, pipe, feat_cols



# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="End-to-end sentiment scoring pipeline v5")
    parser.add_argument("--input",  default="train_2022.csv",
                        help="Input CSV with columns: row_id, TEXT, LABEL")
    parser.add_argument("--output", default="row_scores_v6.csv",
                        help="Output CSV with row-level scores")
    parser.add_argument("--model",  default="rf",
                        choices=["rf", "elastic", "gbm"],
                        help="Regression model (default: rf)")
    parser.add_argument("--test",   default="test_no_answer_2022.csv",
                        help="Test CSV (row_id, TEXT)")
    parser.add_argument("--test-output", default="result_v6a.csv",
                        help="Output CSV for test predictions")
    parser.add_argument("--weight", default="sqrt",
                        choices=list(WEIGHT_SCHEMES),
                        help="Segment weighting scheme (default: sqrt)")
    args = parser.parse_args()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  End-to-End Sentiment Scoring Pipeline  [v6a]")
    print(sep)
    print(f"  Input    : {args.input}")
    print(f"  Output   : {args.output}")
    print(f"  Model    : {args.model}")
    print(f"  Encoder  : {SBERT_MODEL_NAME}  ({SBERT_DIM}-dim)")
    print(f"  Weighting: {args.weight}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    df["row_id"] = df["row_id"].astype(int)
    df["LABEL"]  = df["LABEL"].astype(int)
    print(f"  Loaded {len(df)} rows | {df['row_id'].nunique()} unique row_ids\n")

    # ── Build NLP + Sentence-BERT models ─────────────────────────────────────
    print("── Initialising models ─────────────────────────────────────")
    nlp   = _build_stanza_pipeline()
    sbert = _build_sbert_model()
    vader = _build_vader()

    # ── Steps 1+2: row-by-row split → SBERT embedding → weighted sum ─────────
    print("\n── Steps 1+2: Split + Sentence-BERT Embedding (row-by-row) ─")
    agg_df = run_row_by_row(df, nlp, sbert, vader, has_label=True, weight_scheme=args.weight)

    # ── Step 3: 5-Fold Cross-Validation + Regression Scoring ─────────────────
    print("\n── Step 3: 5-Fold CV + Regression Scoring ──────────────────")
    row_scores, trained_pipe, feat_cols = run_cross_validation(agg_df, args.model, n_splits=5)

    # ── Save training scores ──────────────────────────────────────────────────
    row_scores.to_csv(args.output, index=False)
    print(f"\n  Saved training results to: {args.output}")

    # ── Process test set ──────────────────────────────────────────────────────
    # print(f"\n── Processing test set: {args.test} ────────────────────────")
    # test_df = pd.read_csv(args.test)
    # test_df["row_id"] = test_df["row_id"].astype(int)
    # if "LABEL" not in test_df.columns:
    #     test_df["LABEL"] = -1
    # print(f"  Loaded {len(test_df)} test rows\n")

    # test_agg_df = run_row_by_row(test_df, nlp, sbert, vader,
    #                              has_label=False, weight_scheme=args.weight)

    # X_test = test_agg_df[feat_cols].values
    # test_scores = np.clip(trained_pipe.predict(X_test), 0, 1)
    # test_labels = (test_scores >= 0.5).astype(int)

    # test_output = test_agg_df[["row_id"]].copy()
    # # test_output["predicted_score"] = test_scores
    # test_output["LABEL"] = test_labels
    # test_output.to_csv(args.test_output, index=False)
    # print(f"  Saved test predictions to: {args.test_output}")

if __name__ == "__main__":
    main()
