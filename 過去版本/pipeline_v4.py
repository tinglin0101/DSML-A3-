"""
End-to-End Sentiment Scoring Pipeline  —  v4
=============================================
v4 change: BERT emotion classifier replaced by Sentence-BERT embeddings.
  Model: sentence-transformers/all-MiniLM-L6-v2
  Outputs a 384-dim dense embedding per segment.

  For each row:
    Step 1 — split text at transition words (Stanza)
    Step 2 — Sentence-BERT embedding per segment  (384-dim)
    Aggregate — Sum( emb_i * weight(i) )  where weight(i) = sqrt(i+1)
  After all rows are aggregated, Step 3 — regression on aggregated features.

Usage:
    python pipeline_v4.py --input dataset_split/train_2022.csv --output row_scores.csv
    python pipeline_v4.py --input dataset_split/train_2022.csv --model rf
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.pipeline import Pipeline


# ── Constants ─────────────────────────────────────────────────────────────────

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_DIM        = 384          # embedding dimension for all-MiniLM-L6-v2

# Aggregated feature columns: one per embedding dimension
AGG_FEATURES = [f"emb_{i}" for i in range(SBERT_DIM)]

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
    print(f"  Embedding dim: {SBERT_DIM}")
    return model


def _get_embedding(sbert: SentenceTransformer, text: str) -> np.ndarray:
    """Return a normalised 384-dim embedding for one text segment."""
    emb = sbert.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE — Steps 1+2 per row, then weighted sum
# ══════════════════════════════════════════════════════════════════════════════

def process_row(text: str, label: int, row_id: int,
                nlp: stanza.Pipeline, sbert: SentenceTransformer) -> dict:
    """
    Run Step 1 (split) + Step 2 (Sentence-BERT) on a single row, then aggregate:
        agg_emb = Sum_i( emb_i * weight(i) )   weight(i) = sqrt(i + 1)

    Returns a flat dict:  row_id, LABEL, NUM_SPLITS, emb_0, emb_1, …, emb_383
    """
    # Step 1 — split
    segments = split_text(text, nlp)
    n = len(segments)

    # Weights: sqrt(i+1) for each segment position i
    weights = [np.sqrt(i + 1) for i in range(n)]

    # Step 2 + aggregate
    agg_emb = np.zeros(SBERT_DIM, dtype=np.float32)
    for i, seg in enumerate(segments):
        emb = _get_embedding(sbert, seg)
        agg_emb += emb * weights[i]

    result = {"row_id": row_id, "LABEL": label, "NUM_SPLITS": n}
    for j, val in enumerate(agg_emb):
        result[f"emb_{j}"] = round(float(val), 6)
    return result


def run_row_by_row(df: pd.DataFrame, nlp: stanza.Pipeline,
                   sbert: SentenceTransformer,
                   has_label: bool = True) -> pd.DataFrame:
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
            process_row(str(row["TEXT"]), label, int(row["row_id"]), nlp, sbert)
        )

    cols = ["row_id", "LABEL", "NUM_SPLITS"] + AGG_FEATURES
    result_df = pd.DataFrame(records)[cols]
    print(f"  Done — {len(result_df)} aggregated rows")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Regression scoring
# ══════════════════════════════════════════════════════════════════════════════

def run_regression(agg_df: pd.DataFrame, model_name: str):
    """
    Train a regression model on the aggregated Sentence-BERT feature matrix.

    Returns: (output_df, trained_pipe, feat_cols)
    """
    model_map = {
        "rf":      RandomForestRegressor(n_estimators=100, random_state=42),
        "elastic": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "gbm":     GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(model_map)}")
    print(f"  Model: {model_name}  |  Features: {SBERT_DIM}-dim SBERT embeddings")

    feat_cols = AGG_FEATURES
    X = agg_df[feat_cols].values
    y = agg_df["LABEL"].values

    pipe = Pipeline([("scaler", StandardScaler()), ("model", model_map[model_name])])

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
# TEST PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_test(df_test: pd.DataFrame, nlp: stanza.Pipeline,
                 sbert: SentenceTransformer, pipe: Pipeline) -> pd.DataFrame:
    """
    Run Steps 1–2 (row-by-row) on test data, then predict with the saved pipe.
    Returns: [row_id, final_score, predicted_label]
    """
    agg_df      = run_row_by_row(df_test, nlp, sbert, has_label=False)
    X_test      = agg_df[AGG_FEATURES].values
    scores      = np.clip(pipe.predict(X_test), 0, 1)
    pred_labels = (scores >= 0.5).astype(int)

    output = agg_df[["row_id"]].copy()
    # output["final_score"]     = scores
    output["LABEL"] = pred_labels
    print(f"  Predicted {len(output)} test rows | "
          f"label=1: {pred_labels.sum()} | label=0: {(pred_labels == 0).sum()}")
    return output


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="End-to-end sentiment scoring pipeline v4")
    parser.add_argument("--input",  default="dataset_split/train_2022.csv",
                        help="Input CSV with columns: row_id, TEXT, LABEL")
    parser.add_argument("--output", default="row_scores_v4.csv",
                        help="Output CSV with row-level scores")
    parser.add_argument("--model",  default="rf",
                        choices=["rf", "elastic", "gbm"],
                        help="Regression model (default: rf)")
    parser.add_argument("--test",   default="test_no_answer_2022.csv",
                        help="Test CSV (row_id, TEXT)")
    parser.add_argument("--test-output", default="test_predictions_v4.csv",
                        help="Output CSV for test predictions")
    args = parser.parse_args()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  End-to-End Sentiment Scoring Pipeline  [v4]")
    print(sep)
    print(f"  Input    : {args.input}")
    print(f"  Output   : {args.output}")
    print(f"  Model    : {args.model}")
    print(f"  Encoder  : {SBERT_MODEL_NAME}  ({SBERT_DIM}-dim)")
    print(f"  Aggregation: Sum( sbert_emb_i * sqrt(i+1) )\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    df["row_id"] = df["row_id"].astype(int)
    df["LABEL"]  = df["LABEL"].astype(int)
    print(f"  Loaded {len(df)} rows | {df['row_id'].nunique()} unique row_ids\n")

    # ── Build NLP + Sentence-BERT models ─────────────────────────────────────
    print("── Initialising models ─────────────────────────────────────")
    nlp   = _build_stanza_pipeline()
    sbert = _build_sbert_model()

    # ── Steps 1+2: row-by-row split → SBERT embedding → weighted sum ─────────
    print("\n── Steps 1+2: Split + Sentence-BERT Embedding (row-by-row) ─")
    agg_df = run_row_by_row(df, nlp, sbert, has_label=True)

    # ── Step 3: Regression ────────────────────────────────────────────────────
    print("\n── Step 3: Weighted Regression Scoring ─────────────────────")
    row_scores, trained_pipe, feat_cols = run_regression(agg_df, args.model)

    # ── Save ──────────────────────────────────────────────────────────────────
    row_scores.to_csv(args.output, index=False)
    print(f"\n  Saved results to: {args.output}")

    # ── Test prediction ───────────────────────────────────────────────────────
    if args.test:
        print(f"\n{sep}")
        print("  Test Prediction")
        print(sep)
        df_test = pd.read_csv(args.test)
        df_test["row_id"] = df_test["row_id"].astype(int)
        total = len(df_test)
        print(f"  Loaded {total} test rows\n")

        BATCH_SIZE = 2000
        n_batches  = (total + BATCH_SIZE - 1) // BATCH_SIZE
        all_batches = []

        for i in range(n_batches):
            start = i * BATCH_SIZE
            end   = min(start + BATCH_SIZE, total)
            print(f"── Batch {i + 1}/{n_batches}: rows {start}–{end - 1} ──────────────────────")
            batch_df = df_test.iloc[start:end].copy().reset_index(drop=True)
            all_batches.append(predict_test(batch_df, nlp, sbert, trained_pipe))

        test_scores = pd.concat(all_batches, ignore_index=True)
        test_scores.to_csv(args.test_output, index=False)
        print(f"\n  Saved test predictions to: {args.test_output}")
        print(f"\n  Preview (first 10 rows):")
        print(test_scores.head(10).to_string(index=False))

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
