"""
End-to-End Sentiment Scoring Pipeline  —  v6c
=============================================
v6c change: combines both encoders from v6a and v6b.
  Encoder A : sentence-transformers/all-mpnet-base-v2     (768-dim SBERT)
  Encoder B : cardiffnlp/twitter-roberta-base-sentiment   (768-dim RoBERTa CLS)

  For each row:
    Step 1 — split text at transition words (Stanza)
    Step 2A — SBERT weighted-sum aggregation        → 768-dim vector
    Step 2B — RoBERTa CLS weighted-sum aggregation  → 768-dim vector
    Concat  — [Step2A | Step2B | VADER]             → 1540-dim feature vector
  After all rows are aggregated, Step 3 — regression on concatenated features.

Usage:
    python v6c.py --input dataset_split/train_2022.csv --output row_scores_v6c.csv
    python v6c.py --input dataset_split/train_2022.csv --model rf
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import stanza
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

SBERT_MODEL_NAME   = "sentence-transformers/all-mpnet-base-v2"
ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
SBERT_DIM          = 768
ROBERTA_DIM        = 768

VADER_FEATURES    = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
SBERT_FEATURES    = [f"sbert_{i}"   for i in range(SBERT_DIM)]
ROBERTA_FEATURES  = [f"roberta_{i}" for i in range(ROBERTA_DIM)]
CATEGORY_FEATURES = ["cat_movie", "cat_game", "cat_product", "cat_other"]

ALL_FEATURES = SBERT_FEATURES + ROBERTA_FEATURES + VADER_FEATURES + CATEGORY_FEATURES  # 1544-dim

# ── Category keywords ─────────────────────────────────────────────────────────

_MOVIE_KW   = ['movie', 'film', 'director', 'comedy', 'drama', 'scene', 'screen',
               'cinema', 'actor', 'actress', 'watch', 'story']
_GAME_KW    = ['game', 'play', 'graphics', 'player', 'multiplayer', 'level',
               'nintendo', 'xbox', 'playstation', 'pc']
_PRODUCT_KW = ['price', 'product', 'quality', 'buy', 'bought', 'use', 'device',
               'dish', 'amazon', 'prime', 'item', 'money', 'work', 'battery',
               'cable', 'cheap']


def categorize(text: str) -> str:
    if not isinstance(text, str):
        return "Other"
    words = text.lower().split()
    text_lower = text.lower()
    counts = {
        "Movie":   sum(1 for kw in _MOVIE_KW   if kw in words or kw in text_lower),
        "Game":    sum(1 for kw in _GAME_KW    if kw in words or kw in text_lower),
        "Product": sum(1 for kw in _PRODUCT_KW if kw in words or kw in text_lower),
    }
    max_cat = max(counts, key=counts.get)
    return max_cat if counts[max_cat] > 0 else "Other"


def _category_onehot(category: str) -> dict:
    return {
        "cat_movie":   int(category == "Movie"),
        "cat_game":    int(category == "Game"),
        "cat_product": int(category == "Product"),
        "cat_other":   int(category == "Other"),
    }

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
# STEP 2A — Sentence-BERT embedding
# ══════════════════════════════════════════════════════════════════════════════

def _build_sbert_model() -> SentenceTransformer:
    print(f"  Loading Sentence-BERT model '{SBERT_MODEL_NAME}' …")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    print(f"  SBERT embedding dim : {SBERT_DIM}")
    return model


def _get_sbert_embedding(sbert: SentenceTransformer, text: str) -> np.ndarray:
    emb = sbert.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2B — RoBERTa CLS embedding
# ══════════════════════════════════════════════════════════════════════════════

def _build_roberta_model() -> tuple:
    print(f"  Loading RoBERTa model '{ROBERTA_MODEL_NAME}' …")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME, output_hidden_states=True
    )
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    print(f"  RoBERTa embedding dim : {ROBERTA_DIM}  (CLS token, last hidden layer)")
    return tokenizer, model


def _get_roberta_embedding(roberta: tuple, text: str) -> np.ndarray:
    tokenizer, model = roberta
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_emb = outputs.hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy()
    norm = np.linalg.norm(cls_emb)
    if norm > 0:
        cls_emb = cls_emb / norm
    return cls_emb.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# VADER
# ══════════════════════════════════════════════════════════════════════════════

def _build_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def _get_vader_scores(vader: SentimentIntensityAnalyzer, text: str) -> dict:
    scores = vader.polarity_scores(text)
    return {
        "vader_neg":      scores["neg"],
        "vader_neu":      scores["neu"],
        "vader_pos":      scores["pos"],
        "vader_compound": scores["compound"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Weight schemes
# ══════════════════════════════════════════════════════════════════════════════

def _compute_weights(n: int, scheme: str) -> list:
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


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE — Steps 1+2A+2B per row, then weighted sum, then concat
# ══════════════════════════════════════════════════════════════════════════════

def process_row(text: str, label: int, row_id: int,
                nlp: stanza.Pipeline,
                sbert: SentenceTransformer,
                roberta: tuple,
                vader: SentimentIntensityAnalyzer,
                weight_scheme: str = "sqrt") -> dict:
    """
    Run Step 1 (split) + Step 2A (SBERT) + Step 2B (RoBERTa) on a single row,
    aggregate each encoder independently via weighted sum, then concatenate:
        features = [sbert_agg | roberta_agg | vader | category_onehot]  →  1544-dim

    Returns a flat dict with: row_id, LABEL, NUM_SPLITS, CATEGORY,
                               sbert_0…sbert_767, roberta_0…roberta_767,
                               vader_neg/neu/pos/compound,
                               cat_movie/cat_game/cat_product/cat_other
    """
    segments = split_text(text, nlp)
    n = len(segments)
    weights = _compute_weights(n, weight_scheme)

    category = categorize(text)
    prefix = category.lower()  # "movie", "game", "product", "other"

    agg_sbert   = np.zeros(SBERT_DIM,   dtype=np.float32)
    agg_roberta = np.zeros(ROBERTA_DIM, dtype=np.float32)

    for i, seg in enumerate(segments):
        labeled_seg = f"{prefix}: {seg}"
        agg_sbert   += _get_sbert_embedding(sbert, labeled_seg)    * weights[i]
        agg_roberta += _get_roberta_embedding(roberta, labeled_seg) * weights[i]

    result = {"row_id": row_id, "LABEL": label, "NUM_SPLITS": n, "CATEGORY": category}

    for j, val in enumerate(agg_sbert):
        result[f"sbert_{j}"] = round(float(val), 6)
    for j, val in enumerate(agg_roberta):
        result[f"roberta_{j}"] = round(float(val), 6)
    result.update(_get_vader_scores(vader, text))
    result.update(_category_onehot(category))
    return result


def run_row_by_row(df: pd.DataFrame,
                   nlp: stanza.Pipeline,
                   sbert: SentenceTransformer,
                   roberta: tuple,
                   vader: SentimentIntensityAnalyzer,
                   has_label: bool = True,
                   weight_scheme: str = "sqrt") -> pd.DataFrame:
    total = len(df)
    print(f"  Processing {total} rows (step 1+2A+2B per row) …")

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f"    progress: {i}/{total}")
        label = int(row["LABEL"]) if has_label else -1
        records.append(
            process_row(str(row["TEXT"]), label, int(row["row_id"]),
                        nlp, sbert, roberta, vader,
                        weight_scheme=weight_scheme)
        )

    cols = ["row_id", "LABEL", "NUM_SPLITS", "CATEGORY"] + SBERT_FEATURES + ROBERTA_FEATURES + VADER_FEATURES + CATEGORY_FEATURES
    result_df = pd.DataFrame(records)[cols]
    print(f"  Done — {len(result_df)} aggregated rows  (feature dim: {len(ALL_FEATURES)})")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Regression / Classification
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
    5-fold stratified cross-validation on the concatenated feature matrix.
    Returns: (output_df, trained_pipe_on_full_data, feat_cols)
    """
    total_dim = len(ALL_FEATURES)
    print(f"  Model: {model_name}  |  Features: {SBERT_DIM}-dim SBERT + {ROBERTA_DIM}-dim RoBERTa + 4 VADER + 4 Category = {total_dim}-dim")
    print(f"  Running {n_splits}-fold stratified cross-validation …")

    X = agg_df[ALL_FEATURES].values
    y = agg_df["LABEL"].values

    skf          = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    oof_scores   = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        pipe = Pipeline([("scaler", StandardScaler()), ("model", _make_model(model_name))])
        pipe.fit(X[train_idx], y[train_idx])
        scores               = np.clip(pipe.predict(X[val_idx]), 0, 1)
        pred_labels          = (scores >= 0.5).astype(int)
        oof_scores[val_idx]  = scores
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
    return output, final_pipe, ALL_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    path = "D:\\[課程]DSML\\DSML-A3-\\"
    parser = argparse.ArgumentParser(description="End-to-end sentiment scoring pipeline v7")
    parser.add_argument("--input",       default=path + "train_2022.csv",
                        help="Input CSV with columns: row_id, TEXT, LABEL")
    parser.add_argument("--output",      default=path + "row_scores_v7.csv",
                        help="Output CSV with row-level scores")
    parser.add_argument("--model",       default="rf",
                        choices=["rf", "elastic", "gbm"],
                        help="Regression model (default: rf)")
    parser.add_argument("--test",        default=path + "test_no_answer_2022.csv",
                        help="Test CSV (row_id, TEXT)")
    parser.add_argument("--test-output", default=path + "result_v7.csv",
                        help="Output CSV for test predictions")
    parser.add_argument("--weight",      default="sqrt",
                        choices=list(WEIGHT_SCHEMES),
                        help="Segment weighting scheme (default: sqrt)")
    args = parser.parse_args()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  End-to-End Sentiment Scoring Pipeline  [v7]")
    print(sep)
    print(f"  Input    : {args.input}")
    print(f"  Output   : {args.output}")
    print(f"  Model    : {args.model}")
    print(f"  Encoder A: {SBERT_MODEL_NAME}  ({SBERT_DIM}-dim)")
    print(f"  Encoder B: {ROBERTA_MODEL_NAME}  ({ROBERTA_DIM}-dim CLS)")
    print(f"  Features : {len(ALL_FEATURES)}-dim total")
    print(f"  Weighting: {args.weight}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    df["row_id"] = df["row_id"].astype(int)
    df["LABEL"]  = df["LABEL"].astype(int)
    print(f"  Loaded {len(df)} rows | {df['row_id'].nunique()} unique row_ids\n")

    # ── Build models ──────────────────────────────────────────────────────────
    print("── Initialising models ─────────────────────────────────────")
    nlp     = _build_stanza_pipeline()
    sbert   = _build_sbert_model()
    roberta = _build_roberta_model()
    vader   = _build_vader()

    # ── Steps 1+2A+2B: split → dual embedding → weighted sum → concat ─────────
    print("\n── Steps 1+2A+2B: Split + Dual Embedding (row-by-row) ──────")
    agg_df = run_row_by_row(df, nlp, sbert, roberta, vader,
                            has_label=True, weight_scheme=args.weight)

    # ── Step 3: 5-Fold CV + Regression Scoring ────────────────────────────────
    print("\n── Step 3: 5-Fold CV + Regression Scoring ──────────────────")
    row_scores, trained_pipe, feat_cols = run_cross_validation(agg_df, args.model, n_splits=5)

    # ── Save training scores ──────────────────────────────────────────────────
    row_scores.to_csv(args.output, index=False)
    print(f"\n  Saved training results to: {args.output}")

    # ── Process test set ──────────────────────────────────────────────────────
    print(f"\n── Processing test set: {args.test} ────────────────────────")
    test_df = pd.read_csv(args.test)
    test_df["row_id"] = test_df["row_id"].astype(int)
    if "LABEL" not in test_df.columns:
        test_df["LABEL"] = -1
    print(f"  Loaded {len(test_df)} test rows\n")

    test_agg_df = run_row_by_row(test_df, nlp, sbert, roberta, vader,
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
