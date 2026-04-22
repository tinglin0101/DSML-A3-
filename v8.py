"""
End-to-End Sentiment Scoring Pipeline  —  v8
=============================================
Two-phase fine-tuning of sentence-transformers/all-mpnet-base-v2:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Phase 1 — Domain Adaptation  (無監督 / Unsupervised)           │
  │  Input : ALL texts (Train + Test), labels ignored               │
  │  Method: MLM — Masked Language Modeling                         │
  │    • 隨機遮蔽句子中 15% 的 token（用 [MASK] 替代）               │
  │    • 模型嘗試預測被遮蔽的原始 token                               │
  │    • Loss = cross-entropy（預測 token vs 原始 token）             │
  │    • 效果：模型學會捕捉領域用詞與語法，而不是死記情緒標籤           │
  └─────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────┐
  │  Phase 2 — Supervised SBERT Fine-tuning  (有監督)               │
  │  Input : Train texts + LABEL                                    │
  │  Method: CosineSimilarityLoss on sentence pairs                 │
  │    • 相同情緒 (same label) → target similarity = 1.0            │
  │    • 不同情緒 (diff label) → target similarity = 0.0            │
  │    • 效果：在向量空間中，同情緒句子靠攏，異情緒句子遠離            │
  └─────────────────────────────────────────────────────────────────┘

Pipeline overview:
  原始文字 (Train + Test)
    │
    ├─ Phase 1: MLM Domain Adaptation (無標籤)
    │       → domain-adapted checkpoint
    │
    ├─ Phase 2: CosineSimilarityLoss Fine-tuning (有標籤 Train pairs)
    │       → sentiment-aware SBERT checkpoint
    │
    ▼
  文字切割（Stanza）
    │
    ▼
  用微調後模型產生 768-dim embedding — weighted-sum aggregation
    │
    ▼
  拼接 VADER features  →  772-dim feature vector
    │
    ▼
  5-fold stratified CV（RF / GBM / ElasticNet）+ 回歸評分
    │
    ▼
  對 test set 推論 → result_v8.csv

Usage:
    python v8.py
    python v8.py --model gbm
    python v8.py --da-epochs 2 --ft-epochs 5 --ft-pairs 4000
    python v8.py --da-mlm-prob 0.2  # mask 20% tokens in Phase 1
    python v8.py --skip-da          # skip Phase 1, only Phase 2
"""

import argparse
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import stanza
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Dataset as TorchDataset
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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ── Constants ─────────────────────────────────────────────────────────────────

SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SBERT_DIM        = 768

VADER_FEATURES = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
AGG_FEATURES   = [f"emb_{i}" for i in range(SBERT_DIM)]
ALL_FEATURES   = AGG_FEATURES + VADER_FEATURES   # 772-dim

WEIGHT_SCHEMES = ("sqrt", "uniform", "linear", "log", "decay", "last", "contrast")

TRANSITION_WORDS = {
    "but", "however", "although", "though", "yet", "nevertheless",
    "nonetheless", "except", "while", "whereas",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Build base SBERT model
# ══════════════════════════════════════════════════════════════════════════════

def _build_sbert_model() -> SentenceTransformer:
    print(f"  Loading base Sentence-BERT '{SBERT_MODEL_NAME}' …")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    print(f"  Embedding dim : {SBERT_DIM}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Domain Adaptation via MLM (Masked Language Modeling)
# ══════════════════════════════════════════════════════════════════════════════
#
# 為什麼用 MLM 而非 TSDAE：
#   all-mpnet-base-v2 是 encoder-only 架構（MPNet），沒有 LMHead decoder，
#   無法用 TSDAE（需要 CausalLM decoder）。
#   MLM 是 encoder-only 模型（BERT/MPNet/RoBERTa）的標準 domain adaptation 方式。
#
# MLM（Masked Language Modeling）運作原理：
#
#   1. 輸入句子 "The movie was great but slow"
#   2. 隨機遮蔽 15% token：
#        80% → 替換為 [MASK]：  "The [MASK] was great but slow"
#        10% → 替換為隨機字：   "The movie was great but cable"
#        10% → 保持不變（讓模型學會不依賴位置）
#   3. 模型用雙向 attention 預測每個 [MASK] 的原始 token
#   4. Loss = cross-entropy（預測 token vs 原始 token）
#
#   訓練後效果：
#   • 模型學會理解領域特定詞彙的語境（如 "graphics", "actor", "battery"）
#   • 完全不需要標籤 → Train 和 Test 的文字都能使用
#   • 只學語言特徵，不會看到情緒標籤
#
# ══════════════════════════════════════════════════════════════════════════════

class _TextDataset(TorchDataset):
    """Simple Dataset that tokenizes a list of strings."""
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def domain_adapt_mlm(
    sbert: SentenceTransformer,
    all_texts: list[str],
    epochs: int     = 1,
    batch_size: int = 16,
    lr: float       = 3e-5,
    mlm_prob: float = 0.15,
    output_path: str | None = None,
) -> None:
    """
    Phase 1: Fine-tune *sbert*'s underlying transformer in-place via MLM.
    Uses a pure PyTorch training loop (no accelerate / Trainer required).
    After training, the updated weights stay inside *sbert* automatically.

    Args:
        sbert       : SentenceTransformer to fine-tune (modified in-place)
        all_texts   : list of raw texts (Train + Test, no labels needed)
        epochs      : training epochs (1 is usually enough for domain adapt)
        batch_size  : per-device batch size
        lr          : learning rate
        mlm_prob    : fraction of tokens to mask (default 0.15 = BERT standard)
        output_path : optional path to save adapted checkpoint
    """
    from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

    print(f"\n── Phase 1: MLM Domain Adaptation ────────────────────────")
    print(f"  Texts    : {len(all_texts)} (Train + Test, no labels)")
    print(f"  Epochs   : {epochs}  |  batch={batch_size}  |  lr={lr}")
    print(f"  MLM prob : {mlm_prob} (隨機遮蔽 {int(mlm_prob*100)}% tokens)")

    transformer_module = sbert[0]
    hf_model_name      = transformer_module.auto_model.config.name_or_path

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    mlm_model = AutoModelForMaskedLM.from_pretrained(hf_model_name)

    # Copy current encoder weights into the MLM model
    mlm_model.mpnet.load_state_dict(
        transformer_module.auto_model.state_dict(), strict=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlm_model.to(device)
    mlm_model.train()

    # Tokenize (max_length=128; reviews are short)
    encodings = tokenizer(
        all_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    dataset    = _TextDataset(encodings)
    collator   = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                  mlm_probability=mlm_prob)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collator)

    optimizer   = torch.optim.AdamW(mlm_model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    warmup_steps = max(1, total_steps // 10)

    # Linear warmup scheduler (no accelerate dependency)
    scheduler = torch.optim.lr_scheduler.LinearLR(
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
                print(f"    epoch {epoch}/{epochs}  step {step}/{len(dataloader)}"
                      f"  loss={total_loss/step:.4f}")
        print(f"  Epoch {epoch} done — avg loss: {total_loss/len(dataloader):.4f}")

    # Write adapted weights back into the SentenceTransformer encoder
    mlm_model.to("cpu")
    transformer_module.auto_model.load_state_dict(
        mlm_model.mpnet.state_dict(), strict=False
    )

    if output_path:
        sbert.save(output_path)
        print(f"  Saved adapted model to: {output_path}")

    print("  Phase 1 complete — encoder adapted to domain vocabulary.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Supervised SBERT Fine-tuning (CosineSimilarityLoss)
# ══════════════════════════════════════════════════════════════════════════════
#
# CosineSimilarityLoss 運作原理：
#
#   輸入：(sentence_A, sentence_B, target_similarity)
#     • target = 1.0  →  A 和 B 是相同情緒（例如都是正面）
#     • target = 0.0  →  A 和 B 是不同情緒（一正一負）
#
#   計算：
#     emb_A = SBERT(sentence_A) → 768-dim 向量
#     emb_B = SBERT(sentence_B) → 768-dim 向量
#     pred_sim = cosine_similarity(emb_A, emb_B)  ∈ [-1, 1]
#     loss = MSE(pred_sim, target_similarity)
#
#   訓練後效果：
#   • 正面評論 → 在向量空間中形成一個「群集」
#   • 負面評論 → 在另一個方向形成另一個「群集」
#   • 後續 ML 模型（RF/GBM）更容易在這兩個群集之間畫出分界
#
# ══════════════════════════════════════════════════════════════════════════════

def _sample_pairs(df: pd.DataFrame, n_pairs: int, seed: int = 42) -> list[InputExample]:
    """
    Sample sentence pairs for CosineSimilarityLoss.
    Same label → label=1.0, different label → label=0.0. Balanced 50/50.
    """
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
    n_pairs: int    = 3000,
    epochs: int     = 3,
    batch_size: int = 16,
    lr: float       = 2e-5,
    warmup_steps: int = 100,
    output_path: str | None = None,
) -> None:
    """
    Phase 2: Fine-tune *sbert* in-place using labeled pairs (CosineSimilarityLoss).
    Pure PyTorch training loop — no accelerate dependency.
    """
    print(f"\n── Phase 2: Supervised Fine-tuning (CosineSimilarityLoss) ──")
    print(f"  Train rows  : {len(train_df)}  |  pairs={n_pairs}")
    print(f"  Epochs      : {epochs}  |  batch={batch_size}  |  lr={lr}")
    print(f"  Strategy    : same label→sim=1.0, diff label→sim=0.0")

    examples   = _sample_pairs(train_df, n_pairs)
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size,
                            collate_fn=lambda b: b)  # keep as list of InputExample
    loss_fn    = losses.CosineSimilarityLoss(sbert)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sbert.to(device)
    sbert.train()

    optimizer  = torch.optim.AdamW(sbert.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    # Linear warmup then constant
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=min(warmup_steps, total_steps)
    )

    global_step = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(dataloader, 1):
            # CosineSimilarityLoss expects: features list + labels tensor
            texts_a  = [ex.texts[0] for ex in batch]
            texts_b  = [ex.texts[1] for ex in batch]
            labels   = torch.tensor([ex.label for ex in batch],
                                    dtype=torch.float, device=device)

            feats_a  = {k: v.to(device) for k, v in sbert.tokenize(texts_a).items()
                        if isinstance(v, torch.Tensor)}
            feats_b  = {k: v.to(device) for k, v in sbert.tokenize(texts_b).items()
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
                print(f"    epoch {epoch}/{epochs}  step {step}/{len(dataloader)}"
                      f"  loss={total_loss/step:.4f}")
        print(f"  Epoch {epoch} done — avg loss: {total_loss/len(dataloader):.4f}")

    sbert.train(False)
    if output_path:
        sbert.save(output_path)
        print(f"  Saved fine-tuned model to: {output_path}")
    print("  Phase 2 complete — model now clusters same-sentiment sentences.\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Text splitting (Stanza)
# ══════════════════════════════════════════════════════════════════════════════

def _build_stanza_pipeline() -> stanza.Pipeline:
    stanza.download("en", verbose=False)
    return stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse", verbose=False)


def split_text(text: str, nlp: stanza.Pipeline) -> list[str]:
    if not isinstance(text, str):
        return [text]

    doc          = nlp(text)
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
# Embedding + VADER helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def _get_vader_scores(vader: SentimentIntensityAnalyzer, text: str) -> dict:
    s = vader.polarity_scores(text)
    return {"vader_neg": s["neg"], "vader_neu": s["neu"],
            "vader_pos": s["pos"], "vader_compound": s["compound"]}


def _get_embedding(sbert: SentenceTransformer, text: str) -> np.ndarray:
    emb = sbert.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Weight schemes
# ══════════════════════════════════════════════════════════════════════════════

def _compute_weights(n: int, scheme: str) -> list:
    if scheme == "uniform":  return [1.0] * n
    if scheme == "linear":   return [float(i + 1) for i in range(n)]
    if scheme == "sqrt":     return [np.sqrt(i + 1) for i in range(n)]
    if scheme == "log":      return [np.log(i + 2) for i in range(n)]
    if scheme == "decay":    return [np.exp(-0.5 * i) for i in range(n)]
    if scheme == "last":
        w = [1.0] * n
        if n > 0: w[-1] = 3.0
        return w
    if scheme == "contrast":
        if n == 1: return [2.0]
        w = [0.5] * n; w[0] = 2.0; w[-1] = 2.0
        return w
    raise ValueError(f"Unknown weight scheme '{scheme}'.")


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE — split → embed → weighted sum per row
# ══════════════════════════════════════════════════════════════════════════════

def process_row(text: str, label: int, row_id: int,
                nlp: stanza.Pipeline, sbert: SentenceTransformer,
                vader: SentimentIntensityAnalyzer,
                weight_scheme: str = "sqrt") -> dict:
    segments = split_text(text, nlp)
    n        = len(segments)
    weights  = _compute_weights(n, weight_scheme)

    agg_emb = np.zeros(SBERT_DIM, dtype=np.float32)
    for i, seg in enumerate(segments):
        agg_emb += _get_embedding(sbert, seg) * weights[i]

    result = {"row_id": row_id, "LABEL": label, "NUM_SPLITS": n}
    for j, val in enumerate(agg_emb):
        result[f"emb_{j}"] = round(float(val), 6)
    result.update(_get_vader_scores(vader, text))
    return result


def run_row_by_row(df: pd.DataFrame,
                   nlp: stanza.Pipeline,
                   sbert: SentenceTransformer,
                   vader: SentimentIntensityAnalyzer,
                   has_label: bool = True,
                   weight_scheme: str = "sqrt") -> pd.DataFrame:
    total = len(df)
    print(f"  Extracting embeddings for {total} rows …")

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f"    progress: {i}/{total}")
        label = int(row["LABEL"]) if has_label else -1
        records.append(
            process_row(str(row["TEXT"]), label, int(row["row_id"]),
                        nlp, sbert, vader, weight_scheme=weight_scheme)
        )

    cols      = ["row_id", "LABEL", "NUM_SPLITS"] + AGG_FEATURES + VADER_FEATURES
    result_df = pd.DataFrame(records)[cols]
    print(f"  Done — {len(result_df)} rows embedded  (feature dim: {len(ALL_FEATURES)})")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — 5-Fold CV + Regression Scoring
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

    print(f"\n  ── Fold {fold} ──────────────────────────────")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {pre:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"  AUC      : {auc:.4f}  MSE: {mse:.4f}")
    return dict(acc=acc, pre=pre, rec=rec, f1=f1, auc=auc, mse=mse, cm=cm)


def run_cross_validation(agg_df: pd.DataFrame, model_name: str, n_splits: int = 5):
    """5-fold stratified CV → OOF metrics, then retrain on full data."""
    print(f"  Model: {model_name}  |  Features: {SBERT_DIM}-dim SBERT + 4 VADER = {len(ALL_FEATURES)}-dim")
    print(f"  Running {n_splits}-fold stratified cross-validation …")

    X = agg_df[ALL_FEATURES].values
    y = agg_df["LABEL"].values

    skf          = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    oof_scores   = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        pipe = Pipeline([("scaler", StandardScaler()), ("model", _make_model(model_name))])
        pipe.fit(X[train_idx], y[train_idx])
        scores              = np.clip(pipe.predict(X[val_idx]), 0, 1)
        oof_scores[val_idx] = scores
        fold_metrics.append(_print_fold_metrics(fold, y[val_idx], (scores >= 0.5).astype(int), scores))

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
    path = ""
    parser = argparse.ArgumentParser(
        description="v8: Two-phase fine-tuning — MLM Domain Adapt + Supervised SBERT")
    parser.add_argument("--input",        default=path + "train_2022.csv")
    parser.add_argument("--output",       default=path + "row_scores_v8.csv")
    parser.add_argument("--model",        default="rf", choices=["rf", "elastic", "gbm"])
    parser.add_argument("--test",         default=path + "test_no_answer_2022.csv")
    parser.add_argument("--test-output",  default=path + "result_v8.csv")
    parser.add_argument("--weight",       default="sqrt", choices=list(WEIGHT_SCHEMES))
    # Phase 1 args
    parser.add_argument("--skip-da",      action="store_true",
                        help="Skip Phase 1 (domain adaptation); jump straight to Phase 2")
    parser.add_argument("--da-epochs",    default=1,    type=int,
                        help="Phase 1 MLM epochs (default: 1)")
    parser.add_argument("--da-batch",     default=8,    type=int,
                        help="Phase 1 batch size (default: 8)")
    parser.add_argument("--da-lr",        default=3e-5, type=float,
                        help="Phase 1 learning rate (default: 3e-5)")
    parser.add_argument("--da-mlm-prob",  default=0.15, type=float,
                        help="Phase 1 MLM mask probability (default: 0.15)")
    parser.add_argument("--da-save",      default=None, type=str,
                        help="Path to save Phase 1 checkpoint (optional)")
    parser.add_argument("--load-da",      default=None, type=str,
                        help="載入已存的 Phase 1 checkpoint，跳過 Phase 1")
    parser.add_argument("--skip-ft",      action="store_true",
                        help="只跑 Phase 1，存檔後退出（搭配 --da-save 使用）")
    # Phase 2 args
    parser.add_argument("--ft-epochs",    default=3,    type=int,
                        help="Phase 2 fine-tuning epochs (default: 3)")
    parser.add_argument("--ft-lr",        default=2e-5, type=float,
                        help="Phase 2 learning rate (default: 2e-5)")
    parser.add_argument("--ft-batch",     default=16,   type=int,
                        help="Phase 2 batch size (default: 16)")
    parser.add_argument("--ft-pairs",     default=3000, type=int,
                        help="Phase 2 number of sentence pairs (default: 3000)")
    parser.add_argument("--ft-warmup",    default=100,  type=int,
                        help="Phase 2 warmup steps (default: 100)")
    parser.add_argument("--ft-save",      default=None, type=str,
                        help="Path to save Phase 2 checkpoint (optional)")
    args = parser.parse_args()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Sentiment Scoring Pipeline  [v8]")
    print("  Two-Phase Fine-Tuning: MLM → CosineSimilarityLoss")
    print(sep)
    print(f"  Input      : {args.input}")
    print(f"  Test       : {args.test}")
    print(f"  Model      : {args.model}")
    print(f"  Encoder    : {SBERT_MODEL_NAME}  ({SBERT_DIM}-dim)")
    if args.load_da:
        phase1_desc = f"LOAD from {args.load_da}"
    elif args.skip_da:
        phase1_desc = "SKIPPED"
    else:
        phase1_desc = f"MLM  epochs={args.da_epochs}  lr={args.da_lr}  mlm_prob={args.da_mlm_prob}"
    print(f"  Phase 1    : {phase1_desc}")
    print(f"  Phase 2    : CosineSimilarityLoss  epochs={args.ft_epochs}  lr={args.ft_lr}  pairs={args.ft_pairs}")
    print(f"  Weighting  : {args.weight}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df = pd.read_csv(args.input)
    train_df["row_id"] = train_df["row_id"].astype(int)
    train_df["LABEL"]  = train_df["LABEL"].astype(int)
    print(f"  Loaded {len(train_df)} training rows")

    test_df = pd.read_csv(args.test)
    test_df["row_id"] = test_df["row_id"].astype(int)
    if "LABEL" not in test_df.columns:
        test_df["LABEL"] = -1
    print(f"  Loaded {len(test_df)} test rows\n")

    # ── Initialise base model + Stanza + VADER ────────────────────────────────
    print("── Initialising models ─────────────────────────────────────")
    sbert = _build_sbert_model()
    nlp   = _build_stanza_pipeline()
    vader = _build_vader()

    # ── Phase 1: MLM Domain Adaptation (Train + Test, no labels) ────────────
    if args.load_da:
        print(f"\n  [Phase 1] 載入已存 checkpoint: {args.load_da}\n")
        sbert = SentenceTransformer(args.load_da)
    elif not args.skip_da:
        all_texts = (
            train_df["TEXT"].astype(str).tolist() +
            test_df["TEXT"].astype(str).tolist()
        )
        domain_adapt_mlm(
            sbert,
            all_texts,
            epochs     = args.da_epochs,
            batch_size = args.da_batch,
            lr         = args.da_lr,
            mlm_prob   = args.da_mlm_prob,
            output_path= args.da_save,
        )
    else:
        print("\n  [Phase 1 skipped — using base model weights]\n")

    if args.skip_ft:
        print("  [--skip-ft 指定，Phase 1 完成後退出]")
        return

    # ── Phase 2: Supervised Fine-tuning (labeled Train pairs) ────────────────
    finetune_sbert_supervised(
        sbert,
        train_df,
        n_pairs      = args.ft_pairs,
        epochs       = args.ft_epochs,
        batch_size   = args.ft_batch,
        lr           = args.ft_lr,
        warmup_steps = args.ft_warmup,
        output_path  = args.ft_save,
    )

    # ── Steps 1+2: Stanza split → fine-tuned embedding → weighted sum ─────────
    print("── Steps 1+2: Split + Embedding (two-phase fine-tuned SBERT) ──")
    agg_df = run_row_by_row(train_df, nlp, sbert, vader,
                            has_label=True, weight_scheme=args.weight)

    # ── Step 3: 5-fold CV + regression scoring ────────────────────────────────
    print("\n── Step 3: 5-Fold CV + Regression Scoring ──────────────────")
    row_scores, trained_pipe, feat_cols = run_cross_validation(agg_df, args.model, n_splits=5)

    row_scores.to_csv(args.output, index=False)
    print(f"\n  Saved training results to: {args.output}")

    # ── Process test set ──────────────────────────────────────────────────────
    print(f"\n── Processing test set ─────────────────────────────────────")
    test_agg_df = run_row_by_row(test_df, nlp, sbert, vader,
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
