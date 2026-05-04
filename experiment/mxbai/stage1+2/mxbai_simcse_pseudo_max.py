import random
import warnings
import argparse
import ast
warnings.filterwarnings("ignore")

import gc
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
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

# from huggingface_hub import login
# login(huggingfacetoken="your token")


# ── base ──────────────────────────────────────────────────────────────────
base_path = "./"
base_name = "mxbai_simcse_pseudo_5fold"


# ── Logging ───────────────────────────────────────────────────────────────
_log_fh = None

def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    if _log_fh:
        _log_fh.write(text + "\n")
        _log_fh.flush()


# ── Config ────────────────────────────────────────────────────────────────
CFG = {
    "input":       base_path + "train_2022.csv",
    "test":        base_path + "test_no_answer_2022.csv",
    "test_output": base_path + "result.csv",

    "model":       "rf",          # rf | gbm | elastic

    # Stage 1: unsupervised SimCSE on train + test texts
    "simcse_epochs":  1,
    "simcse_batch":   64,
    "simcse_lr":      2e-5,

    # Stage 2: supervised fine-tune inside each self-training round
    "ft_epochs":   3,
    "ft_lr":       2e-5,
    "ft_batch":    8,
    "ft_pairs":    3000,
    "ft_warmup":   100,
    "ft_save":     None,          # overridden per fold in main()
    "simcse_save": None,          # overridden in main(); set to "" to disable caching

    # Stage 3: pseudo-labeling rounds
    "pseudo_rounds":          2,     # iterations after round 0
    "pseudo_threshold":       0.85,  # score >= thr → pseudo pos; score <= 1-thr → pseudo neg
    "max_pseudo_per_class":   1600,  # cap positive and negative pseudo-labels each

    # Cross-validation
    "n_folds": 5,
}

SBERT_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
SBERT_DIM   = 1024
FEAT_COLS   = [f"emb_{i}" for i in range(SBERT_DIM)]


# ── Memory helpers ────────────────────────────────────────────────────────
def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log(f"    [mem] GPU freed  allocated={torch.cuda.memory_allocated()//1024**2}MB"
            f"  reserved={torch.cuda.memory_reserved()//1024**2}MB")


# ── Model helpers ─────────────────────────────────────────────────────────
def build_sbert() -> SentenceTransformer:
    log(f"Loading '{SBERT_MODEL}' …")
    model = SentenceTransformer(SBERT_MODEL)
    return model.float()


def _make_clf(name: str):
    return {
        "rf":      RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gbm":     GradientBoostingRegressor(n_estimators=200, random_state=42),
        "elastic": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    }[name]


# ── Metric helpers ───────────────────────────────────────────────────────
def _compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auc":       roc_auc_score(y_true, y_score),
        "pr_auc":    average_precision_score(y_true, y_score),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "accuracy":  accuracy_score(y_true, y_pred),
    }


def _log_metrics(metrics: dict, prefix: str = "") -> None:
    log(f"{prefix}"
        f"AUC={metrics['auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
        f"  F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}"
        f"  Rec={metrics['recall']:.4f}  Acc={metrics['accuracy']:.4f}")


# ── Logging callback ──────────────────────────────────────────────────────
class _EpochLog(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            epoch = int(round(logs.get("epoch", 0)))
            grad  = logs.get("grad_norm", float("nan"))
            log(f"    epoch {epoch}  loss={logs['loss']:.5f}  grad_norm={grad:.4f}")


# ── Pair builder ──────────────────────────────────────────────────────────
def _make_pairs(df: pd.DataFrame, n_pairs: int, seed: int = 42) -> list:
    rng    = random.Random(seed)
    texts  = df["TEXT"].astype(str).tolist()
    labels = df["LABEL"].astype(int).tolist()

    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]

    pairs = []
    half  = n_pairs // 2

    for _ in range(half):
        pool = pos if rng.random() < 0.5 else neg
        a, b = rng.sample(pool, 2) if len(pool) >= 2 else (pool[0], pool[0])
        pairs.append(InputExample(texts=[texts[a], texts[b]], label=1.0))

    for _ in range(n_pairs - half):
        pairs.append(InputExample(
            texts=[texts[rng.choice(pos)], texts[rng.choice(neg)]], label=0.0
        ))

    rng.shuffle(pairs)
    return pairs


# ── Stage 1: Unsupervised SimCSE ──────────────────────────────────────────
def finetune_simcse(sbert: SentenceTransformer, all_texts: list) -> None:
    log(f"  [Stage 1] SimCSE unsupervised"
        f"  n_texts={len(all_texts)}  epochs={CFG['simcse_epochs']}"
        f"  batch={CFG['simcse_batch']}  lr={CFG['simcse_lr']}")

    ds = Dataset.from_dict({
        "anchor":   all_texts,
        "positive": all_texts,   # identical — dropout creates the augmentation
    })

    train_args = SentenceTransformerTrainingArguments(
        output_dir="tmp_simcse",
        num_train_epochs=CFG["simcse_epochs"],
        per_device_train_batch_size=CFG["simcse_batch"],
        learning_rate=CFG["simcse_lr"],
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
        loss=losses.MultipleNegativesRankingLoss(sbert),
        callbacks=[_EpochLog()],
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()
    del trainer, ds, train_args
    _free_memory()
    if CFG.get("simcse_save"):
        sbert.save(CFG["simcse_save"])
        log(f"  [Stage 1] SimCSE model saved → {CFG['simcse_save']}")
    log("  [Stage 1] Done.")


# ── Stage 2: Supervised fine-tune (called inside each pseudo round) ───────
def finetune_supervised(sbert: SentenceTransformer, train_df: pd.DataFrame,
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
        output_dir=output_path or "tmp_sup",
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
    del trainer, ds, train_args, pairs
    _free_memory()

    if output_path:
        sbert.save(output_path)
    log("  Fine-tuning done.")


# ── Embedding ─────────────────────────────────────────────────────────────
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

    del embs
    return pd.DataFrame(records)[["row_id", "LABEL"] + FEAT_COLS]


# ── Stage 3: Self-training with pseudo-labeling ───────────────────────────
def self_training(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  sbert_base: SentenceTransformer):
    import copy

    augmented_df = train_df.copy()
    pipe = None
    sbert_ft = None
    total_rounds = CFG["pseudo_rounds"] + 1

    for rnd in range(total_rounds):
        is_last = (rnd == total_rounds - 1)
        n_pseudo = len(augmented_df) - len(train_df)
        log(f"\n══ Self-training Round {rnd}/{total_rounds - 1}"
            f"  total_train={len(augmented_df)}"
            f"  (labeled={len(train_df)}, pseudo={n_pseudo}) ══")

        # Release previous round's model before creating new copy
        if sbert_ft is not None:
            del sbert_ft
            _free_memory()

        sbert_ft = copy.deepcopy(sbert_base)
        out_path = CFG["ft_save"] if is_last else None
        finetune_supervised(sbert_ft, augmented_df, output_path=out_path)

        X_emb = embed(augmented_df, sbert_ft, has_label=True)
        y = augmented_df["LABEL"].astype(int).values
        if pipe is not None:
            del pipe
        pipe = Pipeline([("sc", StandardScaler()), ("clf", _make_clf(CFG["model"]))])
        pipe.fit(X_emb[FEAT_COLS].values, y)
        del X_emb, y
        _free_memory()

        if not is_last:
            test_emb = embed(test_df, sbert_ft, has_label=False)
            scores = np.clip(pipe.predict(test_emb[FEAT_COLS].values), 0, 1)
            del test_emb
            _free_memory()

            thr           = CFG["pseudo_threshold"]
            max_per_class = CFG.get("max_pseudo_per_class", None)
            test_rows     = test_df.reset_index(drop=True)

            # positive: scores >= thr, ranked by highest score, capped at max_per_class
            pos_idx = np.where(scores >= thr)[0]
            pos_idx = pos_idx[np.argsort(scores[pos_idx])[::-1]]
            if max_per_class is not None:
                pos_idx = pos_idx[:max_per_class]

            # negative: scores <= 1-thr, ranked by lowest score, capped at max_per_class
            neg_idx = np.where(scores <= 1.0 - thr)[0]
            neg_idx = neg_idx[np.argsort(scores[neg_idx])]
            if max_per_class is not None:
                neg_idx = neg_idx[:max_per_class]

            pseudo_rows = []
            for idx in pos_idx:
                row = test_rows.iloc[idx]
                pseudo_rows.append({"row_id": int(row["row_id"]),
                                    "TEXT": str(row["TEXT"]), "LABEL": 1})
            for idx in neg_idx:
                row = test_rows.iloc[idx]
                pseudo_rows.append({"row_id": int(row["row_id"]),
                                    "TEXT": str(row["TEXT"]), "LABEL": 0})
            del scores

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


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    global _log_fh
    _log_fh = open("log.txt", "w", encoding="utf-8")

    log("=" * 60)
    log(f"  mxbai SimCSE + Pseudo-labeling Pipeline  [{CFG['n_folds']}-Fold CV]")
    log(f"  model={CFG['model']}  encoder={SBERT_MODEL}")
    log(f"  simcse_epochs={CFG['simcse_epochs']}  simcse_batch={CFG['simcse_batch']}")
    log(f"  ft_epochs={CFG['ft_epochs']}  ft_lr={CFG['ft_lr']}  ft_pairs={CFG['ft_pairs']}")
    log(f"  pseudo_rounds={CFG['pseudo_rounds']}  pseudo_threshold={CFG['pseudo_threshold']}")
    log(f"  max_pseudo_per_class={CFG['max_pseudo_per_class']}")
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

    # Stage 1: SimCSE once, shared across all folds
    sbert = build_sbert()
    all_texts = (train_df["TEXT"].astype(str).tolist() +
                    test_df["TEXT"].astype(str).tolist())
    log(f"\n  Total texts for SimCSE: {len(all_texts)}"
        f"  (train={len(train_df)}, test={len(test_df)})")
    finetune_simcse(sbert, all_texts)

    # Stage 2+3: supervised fine-tune + pseudo-labeling, each round starts from SimCSE model
    pipe, sbert_final = self_training(train_df, test_df, sbert_base=sbert)


    # ── 5-fold CV ──────────────────────────────────────────────────
    n_folds = CFG["n_folds"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)


    test_scores_list = []
    oof_scores  = np.full(len(train_df), -1.0)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["LABEL"])):
        log(f"\n{'='*60}")
        log(f"  Fold {fold + 1}/{n_folds}"
            f"  train={len(train_idx)}  val={len(val_idx)}")
        log(f"{'='*60}")

        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val   = train_df.iloc[val_idx].reset_index(drop=True)

        pipe, sbert_ft = self_training(fold_train, test_df, sbert_base=sbert)

        # OOF predictions on validation fold
        val_emb = embed(fold_val, sbert_ft, has_label=True)
        oof_scores[val_idx] = np.clip(pipe.predict(val_emb[FEAT_COLS].values), 0, 1)
        fm = _compute_metrics(fold_val["LABEL"].values, oof_scores[val_idx])
        fold_metrics.append(fm)
        _log_metrics(fm, prefix=f"\n  Fold {fold + 1} OOF ── ")
        del val_emb

        # Test predictions for this fold
        test_emb = embed(test_df, sbert_ft, has_label=False)
        fold_test_scores = np.clip(pipe.predict(test_emb[FEAT_COLS].values), 0, 1)
        test_scores_list.append(fold_test_scores)
        del test_emb, sbert_ft, pipe
        _free_memory()

    # ── Final training on full train_df ───────────────────────────────────
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
    all_texts = (train_df["TEXT"].astype(str).tolist() +
                    test_df["TEXT"].astype(str).tolist())
    log(f"\n  Total texts for SimCSE: {len(all_texts)}"
        f"  (train={len(train_df)}, test={len(test_df)})")
    finetune_simcse(sbert, all_texts)

    # Stage 2+3: supervised fine-tune + pseudo-labeling, each round starts from SimCSE model
    pipe, sbert_final = self_training(train_df, test_df, sbert_base=sbert)

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
