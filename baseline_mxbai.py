import random
import warnings
import argparse
import ast
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, mean_squared_error,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = "output_log.txt"
_log_fh  = open(LOG_FILE, "w", encoding="utf-8")

def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    _log_fh.write(text + "\n")
    _log_fh.flush()

# ── Config ────────────────────────────────────────────────────────────────────
base_path = "./"
CFG = {
    "input":       base_path + "train_2022.csv",
    "test":        base_path + "test_no_answer_2022.csv",
    "test_output": base_path + "result_baseline_mxbai.csv",

    "model":       "rf",          # rf | gbm | elastic

    # fine-tune inside each CV fold
    "ft_epochs":   3,
    "ft_lr":       2e-5,
    "ft_batch":    16,
    "ft_pairs":    3000,
    "ft_warmup":   100,
    "ft_save":     base_path + "baseline_mxbai_ft_model",

    "cv_splits":   5,
}

SBERT_MODEL  = "mixedbread-ai/mxbai-embed-large-v1"
SBERT_DIM    = 1024
FEAT_COLS    = [f"emb_{i}" for i in range(SBERT_DIM)]


# ── Model helpers ─────────────────────────────────────────────────────────────
def build_sbert() -> SentenceTransformer:
    log(f"Loading '{SBERT_MODEL}' …")
    model = SentenceTransformer(SBERT_MODEL)
    model = model.float()
    return model


def _make_clf(name: str):
    return {
        "rf":      RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gbm":     GradientBoostingRegressor(n_estimators=200, random_state=42),
        "elastic": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    }[name]


# ── Fine-tuning ───────────────────────────────────────────────────────────────
def _make_pairs(df: pd.DataFrame, n_pairs: int, seed: int = 42) -> list:
    rng    = random.Random(seed)
    texts  = df["TEXT"].astype(str).tolist()
    labels = df["LABEL"].astype(int).tolist()

    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]

    pairs = []
    half  = n_pairs // 2

    # same-class pairs → similarity 1.0
    for _ in range(half):
        pool = pos if rng.random() < 0.5 else neg
        a, b = rng.sample(pool, 2) if len(pool) >= 2 else (pool[0], pool[0])
        pairs.append(InputExample(texts=[texts[a], texts[b]], label=1.0))

    # cross-class pairs → similarity 0.0
    for _ in range(n_pairs - half):
        pairs.append(InputExample(
            texts=[texts[rng.choice(pos)], texts[rng.choice(neg)]], label=0.0
        ))

    rng.shuffle(pairs)
    return pairs


def finetune(sbert: SentenceTransformer, train_df: pd.DataFrame,
             output_path=None) -> None:
    log(f"  Fine-tuning: pairs={CFG['ft_pairs']}  epochs={CFG['ft_epochs']}"
        f"  lr={CFG['ft_lr']}  batch={CFG['ft_batch']}")

    pairs      = _make_pairs(train_df, CFG["ft_pairs"])
    dataloader = DataLoader(pairs, shuffle=True, batch_size=CFG["ft_batch"])
    loss_fn    = losses.CosineSimilarityLoss(sbert)

    sbert.fit(
        train_objectives=[(dataloader, loss_fn)],
        epochs=CFG["ft_epochs"],
        warmup_steps=CFG["ft_warmup"],
        optimizer_params={"lr": CFG["ft_lr"]},
        output_path=output_path,
        show_progress_bar=False,
    )
    log("  Fine-tuning done.")


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed(df: pd.DataFrame, sbert: SentenceTransformer,
          has_label: bool = True) -> pd.DataFrame:
    log(f"  Embedding {len(df)} rows …")
    texts = df["TEXT"].astype(str).tolist()
    embs  = sbert.encode(texts, normalize_embeddings=True,
                         batch_size=64, show_progress_bar=False)

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        rec = {"row_id": int(row["row_id"]),
               "LABEL":  int(row["LABEL"]) if has_label else -1}
        rec.update({f"emb_{j}": float(embs[i, j]) for j in range(SBERT_DIM)})
        records.append(rec)

    return pd.DataFrame(records)[["row_id", "LABEL"] + FEAT_COLS]


# ── Metrics ───────────────────────────────────────────────────────────────────
def print_metrics(fold: int, y_true, y_pred, y_score) -> dict:
    auc = float("nan")
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        pass
    metrics = dict(
        acc=accuracy_score(y_true, y_pred),
        pre=precision_score(y_true, y_pred, zero_division=0),
        rec=recall_score(y_true, y_pred, zero_division=0),
        f1 =f1_score(y_true, y_pred, zero_division=0),
        auc=auc,
        mse=mean_squared_error(y_true, y_score),
        cm =confusion_matrix(y_true, y_pred),
    )
    log(f"\n  ── Fold {fold} ──")
    log(f"  Confusion Matrix:\n{metrics['cm']}")
    log(f"  Acc={metrics['acc']:.4f}  Pre={metrics['pre']:.4f}"
        f"  Rec={metrics['rec']:.4f}  F1={metrics['f1']:.4f}"
        f"  AUC={metrics['auc']:.4f}  MSE={metrics['mse']:.4f}")
    return metrics


# ── Cross-validation ──────────────────────────────────────────────────────────
def cross_validate(train_df: pd.DataFrame, sbert_base: SentenceTransformer):
    import copy

    y       = train_df["LABEL"].astype(int).values
    row_ids = train_df["row_id"].astype(int).values
    n       = CFG["cv_splits"]

    sss        = StratifiedShuffleSplit(n_splits=n, test_size=0.2, random_state=42)
    all_metrics = []
    oof_scores  = np.full(len(y), np.nan)

    for fold, (tr_idx, val_idx) in enumerate(sss.split(np.zeros(len(y)), y), 1):
        log(f"\n══ Fold {fold}/{n}  train={len(tr_idx)}  val={len(val_idx)} ══")

        fold_tr  = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)

        sbert = copy.deepcopy(sbert_base)
        finetune(sbert, fold_tr)

        X_tr  = embed(fold_tr,  sbert, has_label=True)[FEAT_COLS].values
        X_val = embed(fold_val, sbert, has_label=True)[FEAT_COLS].values
        y_tr  = fold_tr["LABEL"].astype(int).values
        y_val = fold_val["LABEL"].astype(int).values

        pipe = Pipeline([("sc", StandardScaler()), ("clf", _make_clf(CFG["model"]))])
        pipe.fit(X_tr, y_tr)

        scores             = np.clip(pipe.predict(X_val), 0, 1)
        oof_scores[val_idx] = scores
        all_metrics.append(print_metrics(fold, y_val, (scores >= 0.5).astype(int), scores))

    log("\n  ══ CV Summary ══")
    for m in ("acc", "pre", "rec", "f1", "auc", "mse"):
        vals = [x[m] for x in all_metrics]
        log(f"  {m.upper():4s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # retrain on full data
    log("\n  Retraining on full dataset …")
    sbert_final = copy.deepcopy(sbert_base)
    finetune(sbert_final, train_df, output_path=CFG["ft_save"])

    X_full     = embed(train_df, sbert_final, has_label=True)[FEAT_COLS].values
    final_pipe = Pipeline([("sc", StandardScaler()), ("clf", _make_clf(CFG["model"]))])
    final_pipe.fit(X_full, y)

    oof_labels = np.where(~np.isnan(oof_scores), (oof_scores >= 0.5).astype(int), -1)
    summary = pd.DataFrame({
        "row_id":         row_ids,
        "label":          y,
        "oof_score":      oof_scores,
        "oof_label":      oof_labels,
        "train_score":    np.clip(final_pipe.predict(X_full), 0, 1),
    })
    return summary, final_pipe, sbert_final


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log(f"  mxbai Sentiment Pipeline")
    log(f"  model={CFG['model']}  encoder={SBERT_MODEL}")
    log(f"  ft_epochs={CFG['ft_epochs']}  ft_lr={CFG['ft_lr']}  ft_pairs={CFG['ft_pairs']}")
    log("=" * 60)

    train_df = pd.read_csv(CFG["input"])
    train_df["row_id"] = train_df["row_id"].astype(int)
    train_df["LABEL"]  = train_df["LABEL"].astype(int)
    log(f"Train: {len(train_df)} rows  |  pos={train_df['LABEL'].sum()}  neg={(train_df['LABEL']==0).sum()}")

    test_df = pd.read_csv(CFG["test"])
    test_df["row_id"] = test_df["row_id"].astype(int)
    if "LABEL" not in test_df.columns:
        test_df["LABEL"] = -1
    log(f"Test : {len(test_df)} rows\n")

    sbert = build_sbert()

    _, pipe, sbert_final = cross_validate(train_df, sbert)

    log("\n── Test set prediction ──")
    test_emb   = embed(test_df, sbert_final, has_label=False)
    scores     = np.clip(pipe.predict(test_emb[FEAT_COLS].values), 0, 1)
    test_out   = test_emb[["row_id"]].copy()
    test_out["LABEL"] = (scores >= 0.5).astype(int)
    test_out.to_csv(CFG["test_output"], index=False)
    log(f"Saved → {CFG['test_output']}  (pos={test_out['LABEL'].sum()}, neg={(test_out['LABEL']==0).sum()})")

    _log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.set:
        for kv in args.set:
            k, v = kv.split("=", 1)
            try:
                CFG[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                CFG[k] = v
    main()
