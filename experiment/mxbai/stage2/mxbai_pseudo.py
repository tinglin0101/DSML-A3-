import random
import warnings
import argparse
import ast
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import (
    SentenceTransformer, InputExample, losses,
    SentenceTransformerTrainer, SentenceTransformerTrainingArguments,
)
from datasets import Dataset
from transformers import TrainerCallback
from transformers.trainer_callback import PrinterCallback
import logging as _pylog
_pylog.getLogger("transformers.trainer").setLevel(_pylog.ERROR)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, mean_squared_error,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)


# from huggingface_hub import login
# login("hugging face token")

# ── base ──────────────────────────────────────────────────────────────────
base_path = "./"


# ── Logging ──────────────────────────────────────────────────────────────────
_log_fh = None

def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    if _log_fh:
        _log_fh.write(text + "\n")
        _log_fh.flush()

# ── Config ────────────────────────────────────────────────────────────────────

CFG = {
    "input":       base_path + "train_2022.csv",
    "test":        base_path + "test_no_answer_2022.csv",
    "test_output": base_path + "result.csv",

    "model":       "rf",          # rf | gbm | elastic

    # fine-tune inside each self-training round
    "ft_epochs":   3,
    "ft_lr":       2e-5,
    "ft_batch":    16,
    "ft_pairs":    3000,
    "ft_warmup":   100,
    "ft_save":      None,

    # pseudo-labeling
    "pseudo_rounds":     2,     # number of self-training iterations after round 0
    "pseudo_threshold":  0.8,  # score >= thr → pseudo pos; score <= 1-thr → pseudo neg
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
class _EpochLog(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            epoch = int(round(logs.get("epoch", 0)))
            grad  = logs.get("grad_norm", float("nan"))
            log(f"    epoch {epoch}  loss={logs['loss']:.5f}  grad_norm={grad:.4f}")


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

    pairs = _make_pairs(train_df, CFG["ft_pairs"])
    ds = Dataset.from_dict({
        "sentence1": [p.texts[0] for p in pairs],
        "sentence2": [p.texts[1] for p in pairs],
        "label":     [float(p.label) for p in pairs],
    })

    train_args = SentenceTransformerTrainingArguments(
        output_dir=output_path or "tmp_ft",
        num_train_epochs=CFG["ft_epochs"],
        per_device_train_batch_size=CFG["ft_batch"],
        learning_rate=CFG["ft_lr"],
        warmup_steps=CFG["ft_warmup"],
        logging_strategy="epoch",
        save_strategy="no",
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=sbert,
        args=train_args,
        train_dataset=ds,
        loss=losses.CosineSimilarityLoss(sbert),
        callbacks=[_EpochLog()],
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()

    if output_path:
        sbert.save(output_path)

    log("  Fine-tuning done.")


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed(df: pd.DataFrame, sbert: SentenceTransformer,
          has_label: bool = True) -> pd.DataFrame:
    log(f"  Embedding {len(df)} rows …")
    texts = df["TEXT"].astype(str).tolist()
    embs  = sbert.encode(texts, normalize_embeddings=True,
                         batch_size=32, show_progress_bar=False)

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        rec = {"row_id": int(row["row_id"]),
               "LABEL":  int(row["LABEL"]) if has_label else -1}
        rec.update({f"emb_{j}": float(embs[i, j]) for j in range(SBERT_DIM)})
        records.append(rec)

    return pd.DataFrame(records)[["row_id", "LABEL"] + FEAT_COLS]


# ── Self-training (pseudo-labeling) ───────────────────────────────────────────
def self_training(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  sbert_base: SentenceTransformer):
    import copy

    augmented_df = train_df.copy()
    pipe = None
    sbert_ft = None
    total_rounds = CFG["pseudo_rounds"] + 1  # round 0 = labeled only; rounds 1..N add pseudo

    for rnd in range(total_rounds):
        is_last = (rnd == total_rounds - 1)
        n_pseudo = len(augmented_df) - len(train_df)
        log(f"\n══ Self-training Round {rnd}/{total_rounds - 1}"
            f"  total_train={len(augmented_df)}"
            f"  (labeled={len(train_df)}, pseudo={n_pseudo}) ══")

        # Always restart from the base model to prevent error accumulation
        sbert_ft = copy.deepcopy(sbert_base)
        out_path = CFG["ft_save"] if is_last else None
        finetune(sbert_ft, augmented_df, output_path=out_path)

        X_emb = embed(augmented_df, sbert_ft, has_label=True)
        y = augmented_df["LABEL"].astype(int).values
        pipe = Pipeline([("sc", StandardScaler()), ("clf", _make_clf(CFG["model"]))])
        pipe.fit(X_emb[FEAT_COLS].values, y)

        if not is_last:
            # Predict unlabeled test set and select high-confidence pseudo-labels
            test_emb = embed(test_df, sbert_ft, has_label=False)
            scores = np.clip(pipe.predict(test_emb[FEAT_COLS].values), 0, 1)

            thr = CFG["pseudo_threshold"]
            pseudo_rows = []
            for idx, (_, row) in enumerate(test_df.iterrows()):
                if scores[idx] >= thr:
                    pseudo_rows.append({"row_id": int(row["row_id"]),
                                        "TEXT": str(row["TEXT"]), "LABEL": 1})
                elif scores[idx] <= 1.0 - thr:
                    pseudo_rows.append({"row_id": int(row["row_id"]),
                                        "TEXT": str(row["TEXT"]), "LABEL": 0})

            n_pos = sum(r["LABEL"] == 1 for r in pseudo_rows)
            n_neg = sum(r["LABEL"] == 0 for r in pseudo_rows)
            log(f"  → Pseudo-labels: {len(pseudo_rows)} selected"
                f"  (pos={n_pos}, neg={n_neg})  threshold={thr}")

            if pseudo_rows:
                pseudo_df = pd.DataFrame(pseudo_rows)
                augmented_df = pd.concat([train_df, pseudo_df], ignore_index=True)
            else:
                log("  → No pseudo-labels selected (threshold too strict); keeping current set")

    return pipe, sbert_ft


# ── Nested 5-Fold Cross-Validation ───────────────────────────────────────────
def nested_cv(train_df: pd.DataFrame, test_df: pd.DataFrame,
              sbert_base: SentenceTransformer, n_folds: int = 5) -> dict:
    import copy
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_all  = train_df["LABEL"].astype(int).values
    results = {k: [] for k in ("auc", "acc", "prec", "rec", "f1")}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(train_df)), y_all)):
        log(f"\n── Nested CV  Fold {fold+1}/{n_folds} ──")
        fold_train = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val   = train_df.iloc[va_idx].reset_index(drop=True)

        sbert = copy.deepcopy(sbert_base)
        # Full self-training on fold_train; pseudo-labels drawn from test_df (val not involved)
        pipe_fold, sbert_ft = self_training(fold_train, test_df, sbert)

        va_emb = embed(fold_val, sbert_ft, has_label=True)
        sc = np.clip(pipe_fold.predict(va_emb[FEAT_COLS].values), 0, 1)
        pr = (sc >= 0.6).astype(int)
        y_va = fold_val["LABEL"].astype(int).values

        results["auc"].append(roc_auc_score(y_va, sc))
        results["acc"].append(accuracy_score(y_va, pr))
        results["prec"].append(precision_score(y_va, pr, zero_division=0))
        results["rec"].append(recall_score(y_va, pr, zero_division=0))
        results["f1"].append(f1_score(y_va, pr, zero_division=0))
        log(f"  → AUC={results['auc'][-1]:.4f}  ACC={results['acc'][-1]:.4f}"
            f"  P={results['prec'][-1]:.4f}  R={results['rec'][-1]:.4f}  F1={results['f1'][-1]:.4f}")

    log(f"\n{'═'*52}")
    log(f"  Nested CV Summary ({n_folds} folds)")
    log(f"{'─'*52}")
    for k, v in results.items():
        log(f"  {k.upper():5s}  mean={np.mean(v):.4f}  std={np.std(v):.4f}")
    log(f"{'═'*52}")
    return {k: float(np.mean(v)) for k, v in results.items()}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _log_fh
    _log_fh = open("log.txt", "w", encoding="utf-8")

    log("=" * 60)
    log(f"  mxbai Pseudo-labeling Pipeline")
    log(f"  model={CFG['model']}  encoder={SBERT_MODEL}")
    log(f"  ft_epochs={CFG['ft_epochs']}  ft_lr={CFG['ft_lr']}  ft_pairs={CFG['ft_pairs']}")
    log(f"  pseudo_rounds={CFG['pseudo_rounds']}  pseudo_threshold={CFG['pseudo_threshold']}")
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

    pipe, sbert_final = self_training(train_df, test_df, sbert)

    log("\n── Nested 5-Fold CV (full self-training per fold, val fold never seen) ──")
    nested_cv(train_df, test_df, sbert)

    log("\n── Final model: full self-training on all labeled data ──")
    pipe, sbert_final = self_training(train_df, test_df, sbert)

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
