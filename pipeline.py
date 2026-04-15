"""
End-to-End Sentiment Scoring Pipeline
======================================
Step 1 — Split text at transition words using Stanza NLP
Step 2 — Compute NRC emotion features per segment
Step 3 — Weighted single-level regression to produce final row scores

Usage:
    python pipeline.py --input dataset_split/train_2022.csv --output row_scores.csv
    python pipeline.py --input dataset_split/train_2022.csv --model ridge
"""

import ast
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import stanza
from nrclex import NRCLex
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.pipeline import Pipeline


# ── Constants ─────────────────────────────────────────────────────────────────

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "trust", "anticipation"]

EMOTION_COUNT_COLS = {e: e.capitalize() + "_count" for e in EMOTIONS}
EMOTION_FREQ_COLS  = {e: e.capitalize() + "_freq"  for e in EMOTIONS}

EMOTION_FEATURES = (
    [EMOTION_COUNT_COLS[e] for e in EMOTIONS] +
    [EMOTION_FREQ_COLS[e]  for e in EMOTIONS]
)

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

            # General transition words
            if word_lower in TRANSITION_WORDS or lemma_lower in TRANSITION_WORDS:
                if getattr(word, "upos", "") in ("CCONJ", "SCONJ", "ADV"):
                    if start_char > 0:
                        split_indices.append(start_char)
                continue

            # 'and' linking two verb clauses
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


def run_split(df: pd.DataFrame, nlp: stanza.Pipeline) -> pd.DataFrame:
    """Add SPLIT_TEXT and NUM_SPLITS columns to df."""
    print("  Splitting texts …")
    df = df.copy()
    df["SPLIT_TEXT"] = df["TEXT"].apply(lambda x: split_text(x, nlp))
    df["NUM_SPLITS"]  = df["SPLIT_TEXT"].apply(len)
    print(f"  Done — avg splits per row: {df['NUM_SPLITS'].mean():.2f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — NRC emotion analysis
# ══════════════════════════════════════════════════════════════════════════════

def _get_emotion_scores(nrc: NRCLex, text: str) -> dict:
    nrc.load_raw_text(text)
    raw  = nrc.raw_emotion_scores
    freq = nrc.affect_frequencies
    result = {}
    for emo in EMOTIONS:
        result[EMOTION_COUNT_COLS[emo]] = raw.get(emo, 0)
        result[EMOTION_FREQ_COLS[emo]]  = round(freq.get(emo, 0.0), 6)
    return result


def run_emotion_analysis(df: pd.DataFrame, has_label: bool = True) -> pd.DataFrame:
    """
    Expand SPLIT_TEXT into one row per segment, attach emotion features.
    Input df must have: row_id, SPLIT_TEXT (list), NUM_SPLITS.
    LABEL is optional — pass has_label=False for test data (fills LABEL with -1).
    """
    print("  Loading NRC lexicon …")
    nrc = NRCLex()

    records = []
    total   = len(df)
    print(f"  Analysing {total} rows …")

    for idx, row in df.iterrows():
        if idx % 200 == 0:
            print(f"    progress: {idx}/{total}")
        for sub_id, segment in enumerate(row["SPLIT_TEXT"]):
            record = {
                "row_id":     row["row_id"],
                "sub_id":     sub_id,
                "split_text": segment,
                "LABEL":      row["LABEL"] if has_label else -1,
                "NUM_SPLITS": row["NUM_SPLITS"],
            }
            record.update(_get_emotion_scores(nrc, segment))
            records.append(record)

    fixed_cols = ["row_id", "sub_id", "split_text", "LABEL", "NUM_SPLITS"]
    result_df  = pd.DataFrame(records)[
        fixed_cols +
        [EMOTION_COUNT_COLS[e] for e in EMOTIONS] +
        [EMOTION_FREQ_COLS[e]  for e in EMOTIONS]
    ]
    print(f"  Emotion analysis complete — {len(result_df)} segments total")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Weighted regression scoring
# ══════════════════════════════════════════════════════════════════════════════

def build_weighted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row_id, spread all sub_id emotion features into one wide row.
    Each sub_id's features are multiplied by sqrt(rank+1), so later segments
    contribute more strongly.  Missing sub_ids are filled with 0.

    Returns a DataFrame with columns:
      row_id | row_label | sub{sid}_{feature} …
    """
    sorted_sids = sorted(df["sub_id"].unique())
    n_subs      = len(sorted_sids)

    weight_map = (
        {sid: round(np.sqrt(rank + 1), 4) for rank, sid in enumerate(sorted_sids)}
        if n_subs >= 2
        else {sid: 1 for sid in sorted_sids}
    )
    print(f"  sub_id weights: { {sid: weight_map[sid] for sid in sorted_sids} }")

    available_features = [f for f in EMOTION_FEATURES if f in df.columns]

    row_labels = (
        df.groupby("row_id")["LABEL"]
          .agg(lambda x: int(x.mode()[0]))
          .reset_index()
          .rename(columns={"LABEL": "row_label"})
    )
    wide = row_labels.copy()

    for sid in sorted_sids:
        w      = weight_map[sid]
        sub_df = df[df["sub_id"] == sid][["row_id"] + available_features].copy()
        sub_df = sub_df.groupby("row_id")[available_features].mean().reset_index()
        sub_df[available_features] = sub_df[available_features].fillna(0) * w
        sub_df = sub_df.rename(columns={f: f"sub{sid}_{f}" for f in available_features})
        wide   = wide.merge(sub_df, on="row_id", how="left")
        print(f"    sub_id={sid} | weight={w} | rows={len(sub_df)}")

    feat_cols        = [c for c in wide.columns if c not in ("row_id", "row_label")]
    wide[feat_cols]  = wide[feat_cols].fillna(0)
    return wide


def run_regression(wide: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Train a regression model on the wide weighted feature matrix.

    model_name options:
      "ridge"    Ridge (L2)
      "lasso"    Lasso (L1)
      "elastic"  ElasticNet (L1+L2)
      "bayesian" BayesianRidge
      "gbm"      GradientBoostingRegressor

    Returns: [row_id, row_label, final_score, predicted_label]
    """
    model_map = {
        "ridge":    Ridge(alpha=1.0),
        "lasso":    Lasso(alpha=0.01),
        "elastic":  ElasticNet(alpha=0.01, l1_ratio=0.5),
        "bayesian": BayesianRidge(),
        "gbm":      GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(model_map)}")
    print(f"  Model: {model_name}")

    feat_cols = [c for c in wide.columns if c not in ("row_id", "row_label")]
    X = wide[feat_cols].values
    y = wide["row_label"].values

    pipe = Pipeline([("scaler", StandardScaler()), ("model", model_map[model_name])])

    if len(wide) >= 10:
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
        print(f"  [Regression] rows={len(wide)} | MSE={mse:.4f} (too few for split)")

    output = wide[["row_id", "row_label"]].copy()
    output["final_score"]      = final_scores
    output["predicted_label"]  = (final_scores >= 0.5).astype(int)
    return output, pipe, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# TEST PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_test(df_test: pd.DataFrame,
                 nlp: stanza.Pipeline,
                 pipe: Pipeline,
                 train_feat_cols: list[str],
                 train_sorted_sids: list[int]) -> pd.DataFrame:
    """
    Run the full feature-building pipeline on test data (no LABEL),
    align features to training schema, then predict with the saved pipe.

    Returns: [row_id, final_score, predicted_label]
    """
    # Step 1: split
    print("  Splitting test texts …")
    df_test = run_split(df_test, nlp)

    # Step 2: NRC (no label)
    print("\n── Step 2: NRC Emotion Analysis (test) ─────────────────────")
    df_emotion = run_emotion_analysis(df_test, has_label=False)

    # Step 3: build wide features using same weight logic as training
    n_subs     = len(train_sorted_sids)
    weight_map = (
        {sid: round(np.sqrt(rank + 1), 4) for rank, sid in enumerate(train_sorted_sids)}
        if n_subs >= 2 else {sid: 1 for sid in train_sorted_sids}
    )
    print(f"  Using training weight_map: { {sid: weight_map[sid] for sid in train_sorted_sids} }")

    available_features = [f for f in EMOTION_FEATURES if f in df_emotion.columns]

    # Start with just row_ids (no label for test)
    wide = df_emotion[["row_id"]].drop_duplicates().copy()

    for sid in train_sorted_sids:
        w      = weight_map[sid]
        sub_df = df_emotion[df_emotion["sub_id"] == sid][["row_id"] + available_features].copy()
        if sub_df.empty:
            continue
        sub_df = sub_df.groupby("row_id")[available_features].mean().reset_index()
        sub_df[available_features] = sub_df[available_features].fillna(0) * w
        sub_df = sub_df.rename(columns={f: f"sub{sid}_{f}" for f in available_features})
        wide   = wide.merge(sub_df, on="row_id", how="left")

    # Align to training feature columns (add missing cols as 0, keep order)
    for col in train_feat_cols:
        if col not in wide.columns:
            wide[col] = 0.0
    wide[train_feat_cols] = wide[train_feat_cols].fillna(0)

    X_test       = wide[train_feat_cols].values
    scores       = np.clip(pipe.predict(X_test), 0, 1)
    pred_labels  = (scores >= 0.5).astype(int)

    output = wide[["row_id"]].copy()
    output["final_score"]     = scores
    output["predicted_label"] = pred_labels
    print(f"  Predicted {len(output)} test rows | "
          f"label=1: {pred_labels.sum()} | label=0: {(pred_labels==0).sum()}")
    return output


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="End-to-end sentiment scoring pipeline")
    parser.add_argument("--input",  default="dataset_split/train_2022.csv",
                        help="Input CSV with columns: row_id, TEXT, LABEL")
    parser.add_argument("--output", default="row_scores.csv",
                        help="Output CSV with row-level scores")
    parser.add_argument("--model",  default="bayesian",
                        choices=["ridge", "lasso", "elastic", "bayesian", "gbm"],
                        help="Regression model (default: bayesian)")
    parser.add_argument("--test",   default="test_no_answer_2022.csv",
                        help="Test CSV (row_id, TEXT) — train on --input then predict (default: dataset_split/test_no_answer_2022.csv)")
    parser.add_argument("--test-output", default="test_predictions.csv",
                        help="Output CSV for test predictions (default: test_predictions.csv)")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Save train_2022_split.csv and nrc_emotion_result.csv")
    args = parser.parse_args()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  End-to-End Sentiment Scoring Pipeline")
    print(sep)
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Model : {args.model}\n")

    # ── Load ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    df["row_id"] = df["row_id"].astype(int)
    df["LABEL"]  = df["LABEL"].astype(int)
    print(f"  Loaded {len(df)} rows | {df['row_id'].nunique()} unique row_ids\n")

    # ── Step 1: Text splitting ────────────────────────────────────────────────
    print("── Step 1: Text Splitting ──────────────────────────────────")
    nlp    = _build_stanza_pipeline()
    df_split = run_split(df, nlp)

    # if args.save_intermediates:
    #     df_split.to_csv("train_2022_split.csv", index=False)
    #     print("  Saved train_2022_split.csv")

    # ── Step 2: NRC emotion analysis ──────────────────────────────────────────
    print("\n── Step 2: NRC Emotion Analysis ────────────────────────────")
    df_emotion = run_emotion_analysis(df_split)

    # if args.save_intermediates:
    #     df_emotion.to_csv("nrc_emotion_result.csv", index=False, encoding="utf-8-sig")
    #     print("  Saved nrc_emotion_result.csv")

    # ── Step 3: Regression scoring ────────────────────────────────────────────
    print("\n── Step 3: Weighted Regression Scoring ─────────────────────")
    wide       = build_weighted_features(df_emotion)
    sorted_sids = sorted(df_emotion["sub_id"].unique())
    # print(f"\n  Wide feature matrix: {wide.shape[0]} rows × {wide.shape[1] - 2} features\n")
    row_scores, trained_pipe, feat_cols = run_regression(wide, args.model)

    # ── Save & preview ────────────────────────────────────────────────────────
    row_scores.to_csv(args.output, index=False)
    print(f"\n  Saved results to: {args.output}")
    # print(f"\n  Preview (first 10 rows):")
    # print(row_scores.head(10).to_string(index=False))

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
        all_batches = []
        n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(n_batches):
            start = i * BATCH_SIZE
            end   = min(start + BATCH_SIZE, total)
            print(f"── Batch {i + 1}/{n_batches}: rows {start}–{end - 1} ──────────────────────")
            batch_df = df_test.iloc[start:end].copy().reset_index(drop=True)
            batch_scores = predict_test(batch_df, nlp, trained_pipe, feat_cols, sorted_sids)
            all_batches.append(batch_scores)

        test_scores = pd.concat(all_batches, ignore_index=True)
        test_scores.to_csv(args.test_output, index=False)
        print(f"\n  Saved test predictions to: {args.test_output}")
        print(f"\n  Preview (first 10 rows):")
        print(test_scores.head(10).to_string(index=False))

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
