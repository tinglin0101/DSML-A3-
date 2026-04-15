"""
Weighted Single-Level Regression Scoring Pipeline
===================================================
For each row_id, emotion features from all sub_ids are weighted by sub_id rank
(later sub_id → higher weight = rank+1), then concatenated into a single wide
feature vector. One Ridge regression is trained at the row_id level directly.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ── Feature columns ──────────────────────────────────────────────────────────
EMOTION_FEATURES = [
    "Anger_count", "Disgust_count", "Fear_count", "Joy_count",
    "Sadness_count", "Surprise_count", "Trust_count", "Anticipation_count",
    "Anger_freq", "Disgust_freq", "Fear_freq", "Joy_freq",
    "Sadness_freq", "Surprise_freq", "Trust_freq", "Anticipation_freq",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["row_id"] = df["row_id"].astype(int)
    df["sub_id"] = df["sub_id"].astype(int)
    df["LABEL"] = df["LABEL"].astype(int)
    return df


# ── Build weighted wide features ─────────────────────────────────────────────
def build_weighted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row_id, spread all sub_id emotion features into one wide row.
    Each sub_id's features are multiplied by its weight (rank+1 among all
    observed sub_ids, so later sub_ids contribute more strongly).
    Missing sub_ids for a given row_id are filled with 0.

    Returns a DataFrame indexed by row_id with columns:
      sub{sid}_{feature}  (already weight-scaled)
    plus  row_label  (majority-vote LABEL for that row_id).
    """
    sorted_sids = sorted(df["sub_id"].unique())
    n_subs = len(sorted_sids)
    # sqrt(rank+1) keeps later sub_ids weighted higher but with gentler growth
    # e.g. 4 sub_ids → 1.00, 1.41, 1.73, 2.00  (vs linear 1, 2, 3, 4)
    weight_map = {sid: round(np.sqrt(rank + 1), 4) for rank, sid in enumerate(sorted_sids)} \
                 if n_subs >= 2 else {sid: 1 for sid in sorted_sids}

    print(f"  sub_id weights : { {sid: weight_map[sid] for sid in sorted_sids} }")

    available_features = [f for f in EMOTION_FEATURES if f in df.columns]

    # Build one partial DataFrame per sub_id, then merge on row_id
    row_labels = (df.groupby("row_id")["LABEL"]
                    .agg(lambda x: int(x.mode()[0]))
                    .reset_index()
                    .rename(columns={"LABEL": "row_label"}))

    wide = row_labels.copy()

    for sid in sorted_sids:
        w = weight_map[sid]
        sub_df = df[df["sub_id"] == sid][["row_id"] + available_features].copy()

        # Average features if a row_id somehow appears multiple times in same sub_id
        sub_df = sub_df.groupby("row_id")[available_features].mean().reset_index()

        # Apply weight
        sub_df[available_features] = sub_df[available_features].fillna(0) * w

        # Rename columns to sub{sid}_feature
        rename = {f: f"sub{sid}_{f}" for f in available_features}
        sub_df = sub_df.rename(columns=rename)

        wide = wide.merge(sub_df, on="row_id", how="left")
        print(f"  sub_id={sid} | weight={w} | "
              f"rows contributing={len(sub_df)}")

    # Fill rows with no data for a sub_id with 0 (weight × 0 = no contribution)
    feat_cols = [c for c in wide.columns if c not in ("row_id", "row_label")]
    wide[feat_cols] = wide[feat_cols].fillna(0)

    return wide


# ── Single-level regression ──────────────────────────────────────────────────
def single_level_regression(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Train a regression model on the wide weighted feature matrix.
    Model is selected via the global MODEL setting:
      "ridge"    Ridge (L2)            — stable, all features kept
      "lasso"    Lasso (L1)            — sparse, auto feature selection
      "elastic"  ElasticNet (L1+L2)    — balanced between ridge & lasso
      "bayesian" BayesianRidge         — auto-tunes regularisation strength
      "gbm"      GradientBoosting      — non-linear, captures interactions
    Uses 80 / 20 train-val split (when n ≥ 10), otherwise fits on all data.

    Returns a DataFrame: [row_id, row_label, final_score, predicted_label]
    plus validation diagnostics printed to stdout.
    """
    model_map = {
        "ridge":    Ridge(alpha=1.0),
        "lasso":    Lasso(alpha=0.01),
        "elastic":  ElasticNet(alpha=0.01, l1_ratio=0.5),
        "bayesian": BayesianRidge(),
        "gbm":      GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if MODEL not in model_map:
        raise ValueError(f"Unknown MODEL '{MODEL}'. Choose from: {list(model_map)}")
    print(f"  Model: {MODEL}")

    feat_cols = [c for c in wide.columns if c not in ("row_id", "row_label")]
    X = wide[feat_cols].values
    y = wide["row_label"].values

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("model", model_map[MODEL])])

    if len(wide) >= 10:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        pipe.fit(X_train, y_train)
        val_scores = np.clip(pipe.predict(X_val), 0, 1)

        mse = mean_squared_error(y_val, val_scores)
        try:
            auc = roc_auc_score(y_val, val_scores)
            print(f"\n  [Regression] train={len(X_train)} | val={len(X_val)} | "
                  f"MSE={mse:.4f} | AUC={auc:.4f}")
        except Exception:
            print(f"\n  [Regression] train={len(X_train)} | val={len(X_val)} | "
                  f"MSE={mse:.4f}")

        final_scores = np.clip(pipe.predict(X), 0, 1)
    else:
        pipe.fit(X, y)
        final_scores = np.clip(pipe.predict(X), 0, 1)
        mse = mean_squared_error(y, final_scores)
        print(f"\n  [Regression] row_id count={len(wide)} | MSE={mse:.4f} "
              f"(too few rows for split)")

    output = wide[["row_id", "row_label"]].copy()
    output["final_score"] = final_scores
    output["predicted_label"] = (final_scores >= 0.5).astype(int)
    return output


# ── Config ───────────────────────────────────────────────────────────────────
INPUT_PATH  = "nrc_emotion_result.csv"
OUTPUT_PATH = "row_scores.csv"
MODEL       = "bayesian"   # 可選: "ridge" | "lasso" | "elastic" | "bayesian" | "gbm"


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print("  Weighted Single-Level Regression Scoring Pipeline")
    print(f"{'='*60}")
    print(f"  Input : {INPUT_PATH}")
    print(f"  Output: {OUTPUT_PATH}\n")

    df = load_data(INPUT_PATH)
    print(f"  Loaded {len(df)} rows | "
          f"{df['row_id'].nunique()} unique row_ids | "
          f"{df['sub_id'].nunique()} unique sub_ids\n")

    print("── Build Weighted Wide Features ─────────────────────────")
    wide = build_weighted_features(df)
    print(f"\n  Wide feature matrix: {wide.shape[0]} rows × "
          f"{wide.shape[1] - 2} features\n")

    print("── Single-Level Regression ────────────────────────")
    row_scores = single_level_regression(wide)

    row_scores.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  ✓ Saved results to: {OUTPUT_PATH}")
    print(f"\n  Preview (first 10 rows):")
    print(row_scores.head(10).to_string(index=False))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
