# =============================================================
# vBERT_finetune — 使用 Training Set 微調 (Fine-tune) BERT 進行情感分類
#
# 策略：
#   將 train_2022.csv 以 8:2 比例切分：
#     - 80% (train_split)：用來微調 bert-base-uncased 的分類頭
#     - 20% (val_split)  ：用來驗證微調效果（Validation Set）
#
# 模型架構：
#   bert-base-uncased + Linear Classification Head（BertForSequenceClassification）
#
# 新增功能：轉折詞後半段加權（Clause Weighting）
#   當句子中偵測到轉折詞（but、however、yet、although、though、
#   while、whereas、despite、nevertheless、nonetheless）時，
#   將句子在最後一個轉折詞處切割，並將後半段（轉折後的語氣）
#   重複 CLAUSE_WEIGHT 次拼接回原句，以加強後段對 BERT 的影響。
#
# 新增功能：反諷偵測標籤反轉（Irony Detection）
#   使用 twitter-roberta-base-irony 模型，在最終預測階段判斷文字是否為反諷，
#   若反諷機率大於 IRONY_THRESHOLD，則自動反轉該句的預測情感標籤。
#
# 輸出：
#   - vBERT_finetune.csv：對 test_no_answer_2022.csv 的預測結果
#   - 訓練過程每個 epoch 的 train/val loss 與 accuracy
# =============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------------------------------------------------------
# 0. 超參數設定
# ------------------------------------------------------------------
MODEL_NAME   = "bert-base-uncased"
MAX_LENGTH   = 128          # 截斷長度（句子普遍較短，128 已足夠）
BATCH_SIZE   = 16
EPOCHS       = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1         # 前 10% steps 做 linear warmup
RANDOM_SEED   = 42
VAL_SIZE      = 0.2         # 20% 作為 Validation Set

IRONY_THRESHOLD = 0.80      # 反諷機率超過此值則反轉預測標籤 (0.0~1.0，設為大於1.0則關閉)
IRONY_MODEL     = "cardiffnlp/twitter-roberta-base-irony"

# ------------------------------------------------------------------
# 轉折詞後半段加權設定
# ------------------------------------------------------------------
# CLAUSE_WEIGHT: 後半段（轉折詞之後的文字）重複幾次。
#   = 1  → 不做任何加權（等同原句）
#   = 2  → 後半段額外重複 1 次（推薦起始值）
#   = 3  → 後半段額外重複 2 次（加強版）
CLAUSE_WEIGHT = 2

# 主轉折詞清單（無條件切割）
_PIVOT_WORDS = [
    "however", "but", "yet", "although", "though",
    "while", "whereas", "despite", "nevertheless", "nonetheless",
    "still", "even so", "on the other hand", "in contrast",
]
# 次轉折詞清單（'and' 只在後接否定詞時才切割）
_SOFT_PIVOT  = ["and"]
_NEGATION    = ["not", "no", "never", "neither", "nor", "hardly", "barely", "scarcely"]

# ------------------------------------------------------------------
# 1. 固定隨機種子（確保可重現性）
# ------------------------------------------------------------------
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------------------------------------------------------------------
# 1.5 轉折詞後半段加權函式
# ------------------------------------------------------------------
import re

def split_and_weight_text(text: str, weight: int = CLAUSE_WEIGHT) -> str:
    """
    偵測句子中的轉折詞，將句子在「最後一個轉折詞」處切割，
    並將後半段（轉折後的語氣）重複 `weight` 次拼接回原句尾端，
    以提升後段文字對 BERT [CLS] 表示的影響力。

    邏輯：
      1. 尋找主轉折詞（_PIVOT_WORDS）的所有出現位置，取最後一個。
      2. 若無主轉折詞，嘗試次轉折詞（'and'），但僅在其後接否定詞時才切割。
      3. 找到切割點後：
           新文字 = 原句 + ' ' + (後半段 × weight)
      4. 若無任何轉折詞，直接回傳原文。

    Parameters
    ----------
    text   : 原始文字
    weight : 後半段重複次數（= 1 表示不加權）

    Returns
    -------
    加權後的文字字串
    """
    if weight <= 1:
        return text

    text_lower = text.lower()

    # --- 1. 尋找主轉折詞（取最後出現位置）---
    last_pivot_pos = -1
    matched_pivot  = ""

    for pivot in _PIVOT_WORDS:
        # 使用 word boundary 避免誤配（例如 「without」含 'out'）
        pattern = r'\b' + re.escape(pivot) + r'\b'
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            pos = matches[-1].start()
            if pos > last_pivot_pos:
                last_pivot_pos = pos
                matched_pivot  = pivot

    # --- 2. 若無主轉折詞，嘗試 'and' + 否定詞 的組合 ---
    if last_pivot_pos == -1:
        and_pattern = r'\band\b'
        for m in re.finditer(and_pattern, text_lower):
            after_and = text_lower[m.end():].strip()
            first_word = after_and.split()[0] if after_and.split() else ""
            if first_word in _NEGATION:
                last_pivot_pos = m.start()
                matched_pivot  = "and"
                # 取最後一個符合條件的 'and'

    # --- 3. 無轉折詞 → 原文回傳 ---
    if last_pivot_pos == -1:
        return text

    # --- 4. 切割並加權 ---
    # 找到原始大小寫對應的切割位置（使用 lower 找到的 index 直接對應）
    # after_pivot 包含轉折詞本身，讓 BERT 看到完整語意
    after_pivot = text[last_pivot_pos:].strip()

    # 重複後半段（weight - 1）次額外附加（原句已含一次）
    weighted_text = text + (" " + after_pivot) * (weight - 1)
    return weighted_text


def apply_clause_weighting(series: pd.Series, weight: int = CLAUSE_WEIGHT) -> pd.Series:
    """對整個 pd.Series 套用 split_and_weight_text，並回傳新 Series。"""
    return series.apply(lambda t: split_and_weight_text(str(t), weight))


# ------------------------------------------------------------------
# 2. 載入資料
# ------------------------------------------------------------------
print("=" * 60)
print("載入資料...")
print("=" * 60)

# 路徑設定（相對於此 script 的父目錄）
script_dir  = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(script_dir)
train_path  = os.path.join(parent_dir, "train_2022.csv")
test_path   = os.path.join(parent_dir, "test_no_answer_2022.csv")

train_full_df = pd.read_csv(train_path)
test_df       = pd.read_csv(test_path)

print(f"完整 Training Set 大小: {len(train_full_df)}")
print(f"Test Set 大小         : {len(test_df)}")
print(f"Label 分布:\n{train_full_df['LABEL'].value_counts()}\n")

# ------------------------------------------------------------------
# 3. 切分 train / val（8:2）
# ------------------------------------------------------------------
train_df, val_df = train_test_split(
    train_full_df,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=train_full_df["LABEL"],   # 保持正負樣本比例
)

print(f"訓練資料（80%）大小: {len(train_df)}")
print(f"驗證資料（20%）大小: {len(val_df)}")
print(f"訓練資料 label 分布:\n{train_df['LABEL'].value_counts()}")
print(f"驗證資料 label 分布:\n{val_df['LABEL'].value_counts()}\n")

# ------------------------------------------------------------------
# 3.4 備份原始文字（供後續反諷判斷使用）
# ------------------------------------------------------------------
train_df = train_df.copy()
val_df   = val_df.copy()
test_df  = test_df.copy()

train_df["ORIG_TEXT"] = train_df["TEXT"]
val_df["ORIG_TEXT"]   = val_df["TEXT"]
test_df["ORIG_TEXT"]  = test_df["TEXT"]

# ------------------------------------------------------------------
# 3.5 套用轉折詞後半段加權
# ------------------------------------------------------------------
print("=" * 60)
print(f"套用轉折詞後半段加權（CLAUSE_WEIGHT={CLAUSE_WEIGHT}）...")
print("=" * 60)

if CLAUSE_WEIGHT > 1:
    # 計算各資料集中受影響的句子數量（含轉折詞者）
    def _count_pivoted(series):
        return sum(
            1 for t in series
            if split_and_weight_text(str(t), CLAUSE_WEIGHT) != str(t)
        )

    n_train_pivoted = _count_pivoted(train_df["TEXT"])
    n_val_pivoted   = _count_pivoted(val_df["TEXT"])
    n_test_pivoted  = _count_pivoted(test_df["TEXT"])

    train_df["TEXT"] = apply_clause_weighting(train_df["TEXT"])
    val_df["TEXT"]   = apply_clause_weighting(val_df["TEXT"])
    test_df["TEXT"]  = apply_clause_weighting(test_df["TEXT"])

    print(f"  Train 中含轉折詞的句子: {n_train_pivoted}/{len(train_df)} ({n_train_pivoted/len(train_df)*100:.1f}%)")
    print(f"  Val   中含轉折詞的句子: {n_val_pivoted}/{len(val_df)} ({n_val_pivoted/len(val_df)*100:.1f}%)")
    print(f"  Test  中含轉折詞的句子: {n_test_pivoted}/{len(test_df)} ({n_test_pivoted/len(test_df)*100:.1f}%)")
    print(f"  後半段重複次數（weight）: {CLAUSE_WEIGHT}")

    # 顯示加權效果範例（前 3 筆含轉折詞）
    # print("\n  ─ 加權範例（前 3 筆含轉折詞的訓練樣本）─")
    # shown = 0
    # for orig, weighted in zip(
    #     pd.read_csv(train_path)["TEXT"].tolist(),
    #     train_df["TEXT"].tolist()
    # ):
    #     if orig != weighted:
    #         print(f"  原文   : {orig[:120]}")
    #         print(f"  加權後 : {weighted[:180]}")
    #         print()
    #         shown += 1
    #         if shown >= 3:
    #             break
else:
    print(f"  CLAUSE_WEIGHT={CLAUSE_WEIGHT}，跳過加權步驟。")
print()

# ------------------------------------------------------------------
# 4. 自訂 Dataset
# ------------------------------------------------------------------
class SentimentDataset(Dataset):
    """
    將文本 tokenize 成 BERT 所需格式：
      - input_ids     : token id 序列（含 [CLS], [SEP]）
      - attention_mask: 有效 token 標記（padding 位置為 0）
      - token_type_ids: 句對任務用，單句全為 0
      - labels        : 情感標籤（0 或 1）
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts      = texts.tolist() if hasattr(texts, "tolist") else list(texts)
        self.labels     = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "token_type_ids" : encoding["token_type_ids"].squeeze(0),
            "labels"         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TestDataset(Dataset):
    """測試集（無標籤）專用 Dataset。"""
    def __init__(self, texts, tokenizer, max_length):
        self.texts      = texts.tolist() if hasattr(texts, "tolist") else list(texts)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "token_type_ids" : encoding["token_type_ids"].squeeze(0),
        }

# ------------------------------------------------------------------
# 5. 載入 Tokenizer 與 Model
# ------------------------------------------------------------------
print("=" * 60)
print(f"載入 Tokenizer & Model: {MODEL_NAME}")
print("=" * 60)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# BertForSequenceClassification = bert-base-uncased + Linear(hidden=768, num_labels=2)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,           # 二元分類
    hidden_dropout_prob=0.1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"使用裝置: {device}\n")

# ------------------------------------------------------------------
# 6. 建立 DataLoader
# ------------------------------------------------------------------
train_dataset = SentimentDataset(train_df["TEXT"], train_df["LABEL"], tokenizer, MAX_LENGTH)
val_dataset   = SentimentDataset(val_df["TEXT"],   val_df["LABEL"],   tokenizer, MAX_LENGTH)
test_dataset  = TestDataset(test_df["TEXT"],                          tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}\n")

# ------------------------------------------------------------------
# 7. Optimizer + Scheduler
# ------------------------------------------------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# ------------------------------------------------------------------
# 8. 訓練 & 驗證輔助函式
# ------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, device):
    """執行一個 epoch 的訓練，回傳平均 loss 與 accuracy。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        optimizer.zero_grad()

        input_ids       = batch["input_ids"].to(device)
        attention_mask  = batch["attention_mask"].to(device)
        token_type_ids  = batch["token_type_ids"].to(device)
        labels          = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss    = outputs.loss
        logits  = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device):
    """執行一個 epoch 的驗證，回傳 loss、accuracy、所有預測結果及 softmax 機率。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss   = outputs.loss
            logits = outputs.logits
            probs  = torch.softmax(logits, dim=1)   # shape: (batch, 2)

            total_loss += loss.item()
            preds       = torch.argmax(logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())   # 每筆 [p_neg, p_pos]

    return (
        total_loss / len(loader),
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),   # shape: (N, 2)
    )

# def get_irony_scores(texts, batch_size=32, device="cuda"):
#     """使用 twitter-roberta-base-irony 取得反諷分數 (Probability of being ironic)"""
#     print(f"載入反諷模型: {IRONY_MODEL} ...")
#     import gc
#     tok = AutoTokenizer.from_pretrained(IRONY_MODEL)
#     mod = AutoModelForSequenceClassification.from_pretrained(IRONY_MODEL).to(device)
#     mod.eval()
    
#     scores = []
#     with torch.no_grad():
#         for i in range(0, len(texts), batch_size):
#             batch_texts = texts[i:i+batch_size]
#             inputs = tok(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
#             outputs = mod(**inputs)
#             probs = torch.softmax(outputs.logits, dim=-1)
#             # Label 1 is irony
#             scores.extend(probs[:, 1].cpu().tolist())
            
#     # 釋放 GPU 記憶體
#     del mod
#     del tok
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     return np.array(scores)


# ------------------------------------------------------------------
# 9. 主訓練迴圈
# ------------------------------------------------------------------
print("=" * 60)
print("開始微調 BERT...")
print("=" * 60)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc  = 0.0
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc, val_preds, val_labels, _ = eval_epoch(model, val_loader, device)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}  | Val   Acc: {val_acc:.4f}")

    # 儲存最佳模型（以 val_acc 為準）
    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(script_dir, "best_bert_finetune.pt"))
        print(f"  ✓ 儲存最佳模型（Val Acc: {best_val_acc:.4f}）")

# ------------------------------------------------------------------
# 10. 最終 Validation 評估（載入最佳模型）
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("最終 Validation 評估（最佳模型）")
print("=" * 60)

model.load_state_dict(torch.load(os.path.join(script_dir, "best_bert_finetune.pt"), map_location=device))
_, final_val_acc, final_preds, final_labels, final_probs = eval_epoch(model, val_loader, device)

# --- 反諷偵測與標籤反轉 ---
# if IRONY_THRESHOLD <= 1.0:
#     print(f"\n執行 Validation 反諷語氣判斷 (Threshold={IRONY_THRESHOLD})...")
#     irony_scores_val = get_irony_scores(val_df["ORIG_TEXT"].tolist(), device=device)
#     val_flip_mask = irony_scores_val > IRONY_THRESHOLD
#     n_flipped = val_flip_mask.sum()
#     print(f"  Validation 中偵測到 {n_flipped} 筆反諷語氣，進行標籤反轉！")
    
#     # 反轉 pred: 0->1, 1->0
#     final_preds[val_flip_mask] = 1 - final_preds[val_flip_mask]
    
#     # 重新計算 Accuracy
#     correct = (final_preds == final_labels).sum()
#     final_val_acc = correct / len(final_labels)

precision = precision_score(final_labels, final_preds)
recall    = recall_score(final_labels, final_preds)
f1        = f1_score(final_labels, final_preds)
cm        = confusion_matrix(final_labels, final_preds)

print(f"Validation Accuracy : {final_val_acc:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1-Score            : {f1:.4f}")
print(f"\n混淆矩陣:")
print(f"  TP={cm[1,1]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TN={cm[0,0]}")
print(f"\nValidation 預測分布: {pd.Series(final_preds).value_counts().to_dict()}")

# ------------------------------------------------------------------
# 10.5 匯出錯誤樣本（供進階評估使用）
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("匯出錯誤樣本...")
print("=" * 60)

# 還原 val_df 的原始 index，以便對齊 final_preds
val_df_reset = val_df.reset_index(drop=True)

# 找出預測錯誤的位置
error_mask = final_preds != final_labels
error_indices = np.where(error_mask)[0]

if len(error_indices) == 0:
    print("  沒有錯誤樣本（Validation 全部預測正確）")
else:
    error_texts       = val_df_reset.loc[error_indices, "TEXT"].values
    true_labels       = final_labels[error_indices]
    pred_labels       = final_preds[error_indices]
    prob_neg          = final_probs[error_indices, 0]   # P(label=0)
    prob_pos          = final_probs[error_indices, 1]   # P(label=1)
    confidence        = np.max(final_probs[error_indices], axis=1)  # 最高信心分數
    error_type        = np.where(
        (true_labels == 1) & (pred_labels == 0),
        "FN (漏報正類)",
        "FP (誤報正類)"
    )

    # 也保留原始 row_id（若欄位存在）
    error_df_data = {
        "text"        : error_texts,
        "true_label"  : true_labels,
        "pred_label"  : pred_labels,
        "prob_neg"    : prob_neg.round(4),
        "prob_pos"    : prob_pos.round(4),
        "confidence"  : confidence.round(4),
        "error_type"  : error_type,
    }
    # if IRONY_THRESHOLD <= 1.0:
    #     error_df_data["irony_score"] = irony_scores_val[error_indices].round(4)
    #     error_df_data["flipped_by_irony"] = val_flip_mask[error_indices]

    if "row_id" in val_df_reset.columns:
        error_df_data = {"row_id": val_df_reset.loc[error_indices, "row_id"].values, **error_df_data}
    else:
        error_df_data = {"val_index": error_indices, **error_df_data}

    error_df = pd.DataFrame(error_df_data)

    # 依信心分數降序排列（信心高但錯是最值得分析的）
    error_df = error_df.sort_values("confidence", ascending=False).reset_index(drop=True)

    error_path = os.path.join(script_dir, "vBERT_finetune_errors.csv")
    error_df.to_csv(error_path, index=False, encoding="utf-8-sig")

    fn_count = (error_type == "FN (漏報正類)").sum()
    fp_count = (error_type == "FP (誤報正類)").sum()
    print(f"  錯誤樣本總數  : {len(error_df)}")
    print(f"    FN（漏報正類）: {fn_count}")
    print(f"    FP（誤報正類）: {fp_count}")
    print(f"  已儲存至       : vBERT_finetune_errors.csv")
    print(f"  欄位說明       : {' | '.join(error_df.columns)}")

# ------------------------------------------------------------------
# 11. 繪製訓練曲線
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, EPOCHS+1), history["train_loss"], "b-o", label="Train Loss")
axes[0].plot(range(1, EPOCHS+1), history["val_loss"],   "r-o", label="Val Loss")
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, EPOCHS+1), history["train_acc"], "b-o", label="Train Acc")
axes[1].plot(range(1, EPOCHS+1), history["val_acc"],   "r-o", label="Val Acc")
axes[1].set_title("Accuracy Curve")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("BERT Fine-tuning Training Curves", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "vBERT_finetune.png"), dpi=150, bbox_inches="tight")
print("\n訓練曲線已儲存至 vBERT_finetune.png")

# ------------------------------------------------------------------
# 12. 對 Test Set 做最終預測
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("推論 test_no_answer_2022.csv...")
print("=" * 60)

model.eval()
test_preds = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        preds = torch.argmax(outputs.logits, dim=1)
        test_preds.extend(preds.cpu().numpy())

        if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
            print(f"  進度: {(i+1)*BATCH_SIZE}/{len(test_df)} ({min((i+1)*BATCH_SIZE, len(test_df))/len(test_df)*100:.1f}%)")

test_preds = np.array(test_preds)

# --- 反諷偵測與標籤反轉 ---
# if IRONY_THRESHOLD <= 1.0:
#     print(f"\n執行 Test 反諷語氣判斷 (Threshold={IRONY_THRESHOLD})...")
    # irony_scores_test = get_irony_scores(test_df["ORIG_TEXT"].tolist(), device=device)
    # test_flip_mask = irony_scores_test > IRONY_THRESHOLD
    # n_flipped_test = test_flip_mask.sum()
    # print(f"  Test 中偵測到 {n_flipped_test} 筆反諷語氣，進行標籤反轉！")
    
    # test_preds[test_flip_mask] = 1 - test_preds[test_flip_mask]

# # ------------------------------------------------------------------
# # 13. 儲存預測結果
# # ------------------------------------------------------------------
output_df = pd.DataFrame({
    "row_id": test_df["row_id"],
    "label" : test_preds,
})
output_path = os.path.join(script_dir, "vBERT_finetune.csv")
output_df.to_csv(output_path, index=False)

print(f"\n預測結果已儲存至 vBERT_finetune.csv ({len(output_df)} 筆)")
print(f"預測分布: {pd.Series(test_preds).value_counts().to_dict()}")

# # ------------------------------------------------------------------
# # 14. 摘要
# # ------------------------------------------------------------------
print("\n" + "=" * 60)
print("摘要")
print("=" * 60)
print(f"模型        : {MODEL_NAME} (BertForSequenceClassification)")
print(f"微調方式    : 全參數微調（All-layer fine-tuning）")
print(f"訓練資料    : train_2022.csv 的 80%（{len(train_df)} 筆）")
print(f"驗證資料    : train_2022.csv 的 20%（{len(val_df)} 筆）")
print(f"Epochs      : {EPOCHS}")
print(f"Batch Size  : {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Max Length  : {MAX_LENGTH}")
print(f"Best Val Acc: {best_val_acc:.4f}")
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Test 筆數   : {len(output_df)}")
print(f"錯誤樣本    : vBERT_finetune_errors.csv（僅含 Validation 錯誤樣本）")
