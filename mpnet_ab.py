import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, mean_squared_error,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)
from sklearn.pipeline import Pipeline

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_FILE = "mpnet_ab_log.txt"
_log_fh  = open(LOG_FILE, "w", encoding="utf-8")


def log(*args, **kwargs):
    text = " ".join(str(a) for a in args)
    _log_fh.write(text + "\n")
    _log_fh.flush()


def close_log():
    _log_fh.close()


# ── Constants ─────────────────────────────────────────────────────────────────

SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SBERT_DIM        = 768
AGG_FEATURES     = [f"emb_{i}" for i in range(SBERT_DIM)]
ALL_FEATURES     = AGG_FEATURES


# ── Model building ────────────────────────────────────────────────────────────

def build_sbert_model() -> SentenceTransformer:
    log(f"  Loading base Sentence-BERT '{SBERT_MODEL_NAME}' …")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    log(f"  Embedding dim : {SBERT_DIM}")
    return model


def _make_model(model_name: str):
    model_map = {
        "rf":      RandomForestRegressor(n_estimators=100, random_state=42),
        "elastic": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "gbm":     GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(model_map)}")
    return model_map[model_name]


# ── Phase 1: MLM Domain Adaptation ───────────────────────────────────────────

class _TextDataset(TorchDataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def domain_adapt_mlm(
    sbert: SentenceTransformer,
    all_texts: list,
    epochs: int     = 1,
    batch_size: int = 16,
    lr: float       = 3e-5,
    mlm_prob: float = 0.15,
    output_path     = None,
) -> None:
    from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

    log(f"\n── Phase 1: MLM Domain Adaptation ────────────────────────")
    log(f"  Texts    : {len(all_texts)} (Train + Test, no labels)")
    log(f"  Epochs   : {epochs}  |  batch={batch_size}  |  lr={lr}")
    log(f"  MLM prob : {mlm_prob} (隨機遮蔽 {int(mlm_prob*100)}% tokens)")

    transformer_module = sbert[0]
    hf_model_name      = transformer_module.auto_model.config.name_or_path

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    mlm_model = AutoModelForMaskedLM.from_pretrained(hf_model_name)
    mlm_model.mpnet.load_state_dict(
        transformer_module.auto_model.state_dict(), strict=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlm_model.to(device)
    mlm_model.train()

    encodings  = tokenizer(all_texts, truncation=True, padding="max_length",
                           max_length=128, return_tensors="pt")
    dataset    = _TextDataset(encodings)
    collator   = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                 mlm_probability=mlm_prob)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collator)

    optimizer    = torch.optim.AdamW(mlm_model.parameters(), lr=lr)
    total_steps  = len(dataloader) * epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler    = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )

    global_step = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(dataloader, 1):
            batch  = {k: v.to(device) for k, v in batch.items()}
            output = mlm_model(**batch)
            loss   = output.loss
            loss.backward()
            optimizer.step()
            if global_step < warmup_steps:
                scheduler.step()
            optimizer.zero_grad()
            total_loss  += loss.item()
            global_step += 1
            if step % 50 == 0:
                log(f"    epoch {epoch}/{epochs}  step {step}/{len(dataloader)}"
                    f"  loss={total_loss/step:.4f}")
        log(f"  Epoch {epoch} done — avg loss: {total_loss/len(dataloader):.4f}")

    mlm_model.to("cpu")
    transformer_module.auto_model.load_state_dict(
        mlm_model.mpnet.state_dict(), strict=False
    )
    if output_path:
        sbert.save(output_path)
        log(f"  Saved adapted model to: {output_path}")
    log("  Phase 1 complete — encoder adapted to domain vocabulary.\n")


# ── Phase 2: Supervised Fine-tuning ──────────────────────────────────────────

def _sample_pairs(df: pd.DataFrame, n_pairs: int, seed: int = 42) -> list:
    rng    = random.Random(seed)
    texts  = df["TEXT"].astype(str).tolist()
    labels = df["LABEL"].astype(int).tolist()

    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    examples = []
    half     = n_pairs // 2

    for _ in range(half):
        pool = pos_idx if rng.random() < 0.5 else neg_idx
        if len(pool) >= 2:
            a, b = rng.sample(pool, 2)
        else:
            a = b = pool[0]
        examples.append(InputExample(texts=[texts[a], texts[b]], label=1.0))

    for _ in range(n_pairs - half):
        a = rng.choice(pos_idx)
        b = rng.choice(neg_idx)
        examples.append(InputExample(texts=[texts[a], texts[b]], label=0.0))

    rng.shuffle(examples)
    return examples


def finetune_sbert_supervised(
    sbert: SentenceTransformer,
    train_df: pd.DataFrame,
    n_pairs: int      = 3000,
    epochs: int       = 3,
    batch_size: int   = 16,
    lr: float         = 2e-5,
    warmup_steps: int = 100,
    output_path       = None,
) -> None:
    log(f"\n── Phase 2: Supervised Fine-tuning (CosineSimilarityLoss) ──")
    log(f"  Train rows  : {len(train_df)}  |  pairs={n_pairs}")
    log(f"  Epochs      : {epochs}  |  batch={batch_size}  |  lr={lr}")
    log(f"  Strategy    : same label→sim=1.0, diff label→sim=0.0")

    examples   = _sample_pairs(train_df, n_pairs)
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size,
                            collate_fn=lambda b: b)
    loss_fn    = losses.CosineSimilarityLoss(sbert)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sbert.to(device)
    sbert.train()

    optimizer   = torch.optim.AdamW(sbert.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler   = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=min(warmup_steps, total_steps)
    )

    global_step = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(dataloader, 1):
            texts_a = [ex.texts[0] for ex in batch]
            texts_b = [ex.texts[1] for ex in batch]
            labels  = torch.tensor([ex.label for ex in batch],
                                   dtype=torch.float, device=device)
            feats_a = {k: v.to(device) for k, v in sbert.tokenize(texts_a).items()
                       if isinstance(v, torch.Tensor)}
            feats_b = {k: v.to(device) for k, v in sbert.tokenize(texts_b).items()
                       if isinstance(v, torch.Tensor)}
            loss = loss_fn([feats_a, feats_b], labels)
            loss.backward()
            optimizer.step()
            if global_step < warmup_steps:
                scheduler.step()
            optimizer.zero_grad()
            total_loss  += loss.item()
            global_step += 1
            if step % 50 == 0:
                log(f"    epoch {epoch}/{epochs}  step {step}/{len(dataloader)}"
                    f"  loss={total_loss/step:.4f}")
        log(f"  Epoch {epoch} done — avg loss: {total_loss/len(dataloader):.4f}")

    sbert.train(False)
    if output_path:
        sbert.save(output_path)
        log(f"  Saved fine-tuned model to: {output_path}")
    log("  Phase 2 complete — model now clusters same-sentiment sentences.\n")


# ── Embedding extraction ──────────────────────────────────────────────────────

def _get_embedding(sbert: SentenceTransformer, text: str) -> np.ndarray:
    emb = sbert.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32)


def process_row(text: str, label: int, row_id: int,
                sbert: SentenceTransformer) -> dict:
    emb    = _get_embedding(sbert, text)
    result = {"row_id": row_id, "LABEL": label}
    for j, val in enumerate(emb):
        result[f"emb_{j}"] = round(float(val), 6)
    return result


def run_row_by_row(df: pd.DataFrame,
                   sbert: SentenceTransformer,
                   has_label: bool = True) -> pd.DataFrame:
    total = len(df)
    log(f"  Extracting embeddings for {total} rows …")
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            log(f"    progress: {i}/{total}")
        label = int(row["LABEL"]) if has_label else -1
        records.append(process_row(str(row["TEXT"]), label, int(row["row_id"]), sbert))
    cols      = ["row_id", "LABEL"] + AGG_FEATURES
    result_df = pd.DataFrame(records)[cols]
    log(f"  Done — {len(result_df)} rows embedded  (feature dim: {len(ALL_FEATURES)})")
    return result_df


# ── Cross-validation ──────────────────────────────────────────────────────────

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
    log(f"\n  ── Round {fold} ──")
    log(f"  Confusion Matrix:\n{cm}")
    log(f"  Accuracy : {acc:.4f}")
    log(f"  Precision: {pre:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    log(f"  AUC      : {auc:.4f}  MSE: {mse:.4f}")
    return dict(acc=acc, pre=pre, rec=rec, f1=f1, auc=auc, mse=mse, cm=cm)


def run_cross_validation(
    train_df: pd.DataFrame,
    sbert_base: SentenceTransformer,
    model_name: str,
    cfg: dict,
    n_splits: int = 5,
):
    import copy
    from sklearn.model_selection import StratifiedShuffleSplit

    log(f"  Model: {model_name}  |  Features: {SBERT_DIM}-dim SBERT = {len(ALL_FEATURES)}-dim")
    log(f"  Running {n_splits} x 80/20 splits — fine-tune INSIDE each round (no leakage) …")

    y       = train_df["LABEL"].astype(int).values
    row_ids = train_df["row_id"].astype(int).values

    sss          = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    fold_metrics = []
    oof_scores   = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(sss.split(np.zeros(len(y)), y), start=1):
        log(f"\n══ Round {fold}/{n_splits}  (train={len(train_idx)}, val={len(val_idx)}) ══")

        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df   = train_df.iloc[val_idx].reset_index(drop=True)

        sbert = copy.deepcopy(sbert_base)
        finetune_sbert_supervised(
            sbert, fold_train_df,
            n_pairs=cfg["ft_pairs"], epochs=cfg["ft_epochs"],
            batch_size=cfg["ft_batch"], lr=cfg["ft_lr"],
            warmup_steps=cfg["ft_warmup"], output_path=None,
        )

        train_emb_df = run_row_by_row(fold_train_df, sbert, has_label=True)
        val_emb_df   = run_row_by_row(fold_val_df,   sbert, has_label=True)

        X_tr  = train_emb_df[ALL_FEATURES].values
        y_tr  = train_emb_df["LABEL"].values
        X_val = val_emb_df[ALL_FEATURES].values
        y_val = val_emb_df["LABEL"].values

        pipe                = Pipeline([("scaler", StandardScaler()),
                                        ("model", _make_model(model_name))])
        pipe.fit(X_tr, y_tr)
        scores              = np.clip(pipe.predict(X_val), 0, 1)
        oof_scores[val_idx] = scores
        fold_metrics.append(
            _print_fold_metrics(fold, y_val, (scores >= 0.5).astype(int), scores)
        )

    valid_mask = ~np.isnan(oof_scores)
    oof_labels = np.where(valid_mask, (oof_scores >= 0.5).astype(int), -1)

    log("\n  ══ Overall (averaged across 5 rounds) ══")
    for metric in ("acc", "pre", "rec", "f1", "auc", "mse"):
        vals = [m[metric] for m in fold_metrics]
        log(f"  {metric.upper():9s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    log("\n  Retraining on full dataset …")
    sbert_final = copy.deepcopy(sbert_base)
    finetune_sbert_supervised(
        sbert_final, train_df,
        n_pairs=cfg["ft_pairs"], epochs=cfg["ft_epochs"],
        batch_size=cfg["ft_batch"], lr=cfg["ft_lr"],
        warmup_steps=cfg["ft_warmup"], output_path=cfg["ft_save"],
    )
    full_emb_df  = run_row_by_row(train_df, sbert_final, has_label=True)
    X_full       = full_emb_df[ALL_FEATURES].values
    final_pipe   = Pipeline([("scaler", StandardScaler()),
                              ("model", _make_model(model_name))])
    final_pipe.fit(X_full, y)
    final_scores = np.clip(final_pipe.predict(X_full), 0, 1)

    output = pd.DataFrame({
        "row_id":          row_ids,
        "row_label":       y,
        "oof_score":       oof_scores,
        "oof_pred_label":  oof_labels,
        "final_score":     final_scores,
        "predicted_label": (final_scores >= 0.5).astype(int),
    })
    return output, final_pipe, sbert_final, ALL_FEATURES


# ── Pseudo-label ──────────────────────────────────────────────────────────────

def apply_pseudo_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_scores: np.ndarray,
    thresh_low: float,
    thresh_high: float,
) -> tuple:
    mask         = (test_scores <= thresh_low) | (test_scores >= thresh_high)
    pseudo_label = (test_scores >= thresh_high).astype(int)

    pseudo_df          = test_df[mask].copy().reset_index(drop=True)
    pseudo_df["LABEL"] = pseudo_label[mask]

    n_added = int(mask.sum())
    n_pos   = int(pseudo_label[mask].sum())
    n_neg   = n_added - n_pos
    log(f"  Pseudo-label filter: score ≤ {thresh_low} or ≥ {thresh_high}")
    log(f"    Selected {n_added}/{len(test_df)} test samples  (pos={n_pos}, neg={n_neg})")

    augmented = pd.concat([train_df, pseudo_df], ignore_index=True)
    return augmented, n_added


def retrain_with_pseudo(
    augmented_df: pd.DataFrame,
    sbert_base: SentenceTransformer,
    model_name: str,
    cfg: dict,
) -> tuple:
    import copy
    sbert = copy.deepcopy(sbert_base)
    finetune_sbert_supervised(
        sbert, augmented_df,
        n_pairs=cfg["ft_pairs"], epochs=cfg["ft_epochs"],
        batch_size=cfg["ft_batch"], lr=cfg["ft_lr"],
        warmup_steps=cfg["ft_warmup"], output_path=None,
    )
    emb_df = run_row_by_row(augmented_df, sbert, has_label=True)
    X      = emb_df[ALL_FEATURES].values
    y      = emb_df["LABEL"].values
    pipe   = Pipeline([("scaler", StandardScaler()), ("model", _make_model(model_name))])
    pipe.fit(X, y)
    return pipe, sbert


# ── Config ────────────────────────────────────────────────────────────────────

# base_path = "/content/drive/MyDrive/Colab_Notebooks/a5/"
base_path = "./"
CFG = {
    "input":        base_path + "train_2022.csv",
    "test":         base_path + "test_no_answer_2022.csv",
    "test_output":  base_path + "result_mpnet_ab.csv",

    "model":        "rf",

    "skip_da":      False,
    "load_da":      None,
    "da_epochs":    2,
    "da_batch":     16,
    "da_lr":        4e-5,
    "da_mlm_prob":  0.15,
    "da_save":      None,

    "skip_ft":      False,

    "ft_epochs":    3,
    "ft_lr":        2e-5,
    "ft_batch":     16,
    "ft_pairs":     3000,
    "ft_warmup":    100,
    "ft_save":      None,

    # A+B: one-shot pseudo-label (single retrain after CV)
    "pseudo_iters":       1,
    "pseudo_thresh_low":  0.15,
    "pseudo_thresh_high": 0.85,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    c = CFG

    sep = "=" * 60
    log(f"\n{sep}")
    log("  mpnet_ab — Pseudo-Label A+B (one-shot)")
    log("  CV → predict test → high-conf → retrain SBERT + classifier once")
    log(sep)
    log(f"  Input      : {c['input']}")
    log(f"  Test       : {c['test']}")
    log(f"  Model      : {c['model']}")
    log(f"  Encoder    : {SBERT_MODEL_NAME}  ({SBERT_DIM}-dim)")
    if c["load_da"]:
        phase1_desc = f"LOAD from {c['load_da']}"
    elif c["skip_da"]:
        phase1_desc = "SKIPPED"
    else:
        phase1_desc = f"MLM  epochs={c['da_epochs']}  lr={c['da_lr']}  mlm_prob={c['da_mlm_prob']}"
    log(f"  Phase 1    : {phase1_desc}")
    log(f"  Phase 2    : CosineSimilarityLoss (inside CV)  epochs={c['ft_epochs']}  lr={c['ft_lr']}  pairs={c['ft_pairs']}")
    log(f"  Pseudo-L   : A+B  thresh=[≤{c['pseudo_thresh_low']}, ≥{c['pseudo_thresh_high']}]\n")

    train_df = pd.read_csv(c["input"])
    train_df["row_id"] = train_df["row_id"].astype(int)
    train_df["LABEL"]  = train_df["LABEL"].astype(int)
    log(f"  Loaded {len(train_df)} training rows")

    test_df = pd.read_csv(c["test"])
    test_df["row_id"] = test_df["row_id"].astype(int)
    if "LABEL" not in test_df.columns:
        test_df["LABEL"] = -1
    log(f"  Loaded {len(test_df)} test rows\n")

    log("── Initialising models ──")
    sbert = build_sbert_model()

    if c["load_da"]:
        log(f"\n  [Phase 1] 載入已存 checkpoint: {c['load_da']}\n")
        sbert = SentenceTransformer(c["load_da"])
    elif not c["skip_da"]:
        all_texts = (
            train_df["TEXT"].astype(str).tolist() +
            test_df["TEXT"].astype(str).tolist()
        )
        domain_adapt_mlm(
            sbert, all_texts,
            epochs=c["da_epochs"], batch_size=c["da_batch"],
            lr=c["da_lr"], mlm_prob=c["da_mlm_prob"],
            output_path=c["da_save"],
        )
    else:
        log("\n  [Phase 1 skipped — using base model weights]\n")

    if c["skip_ft"]:
        log("  [skip_ft=True，Phase 1 完成後退出]")
        close_log()
        return

    log("\n── Step 3: 5x 80/20 CV with fine-tune inside each round ──")
    row_scores, trained_pipe, sbert_final, feat_cols = run_cross_validation(
        train_df, sbert, c["model"], cfg=c, n_splits=5
    )

    log(f"\n── Processing test set ──")
    test_agg_df = run_row_by_row(test_df, sbert_final, has_label=False)
    X_test      = test_agg_df[feat_cols].values
    test_scores = np.clip(trained_pipe.predict(X_test), 0, 1)

    # ── A+B: single pseudo-label round ───────────────────────────────────────
    log(f"\n── Pseudo-Label A+B  (thresh=[≤{c['pseudo_thresh_low']}, ≥{c['pseudo_thresh_high']}]) ──")
    augmented_df, n_added = apply_pseudo_labels(
        train_df, test_df, test_scores,
        c["pseudo_thresh_low"], c["pseudo_thresh_high"],
    )
    if n_added > 0:
        log("  Retraining SBERT (B) + classifier (A) on augmented data …")
        new_pipe, new_sbert = retrain_with_pseudo(augmented_df, sbert, c["model"], cfg=c)
        test_agg_df = run_row_by_row(test_df, new_sbert, has_label=False)
        X_test      = test_agg_df[feat_cols].values
        new_scores  = np.clip(new_pipe.predict(X_test), 0, 1)
        log(f"  Score shift — mean: {test_scores.mean():.4f} → {new_scores.mean():.4f}  "
            f"std: {test_scores.std():.4f} → {new_scores.std():.4f}")
        test_scores = new_scores
    else:
        log("  No samples passed threshold — skipping retrain.")

    test_labels = (test_scores >= 0.5).astype(int)
    test_output = test_agg_df[["row_id"]].copy()
    test_output["LABEL"] = test_labels
    test_output.to_csv(c["test_output"], index=False)
    log(f"  Saved test predictions to: {c['test_output']}")

    close_log()


if __name__ == "__main__":
    import argparse, ast
    parser = argparse.ArgumentParser(description="mpnet A+B pseudo-label")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE",
                        help="Override CFG keys, e.g. --set model=gbm pseudo_thresh_low=0.1")
    args = parser.parse_args()
    if args.set:
        for kv in args.set:
            key, val = kv.split("=", 1)
            try:
                CFG[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                CFG[key] = val
    main()
