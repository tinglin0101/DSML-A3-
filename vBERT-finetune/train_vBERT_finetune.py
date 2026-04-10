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

# ------------------------------------------------------------------
# 1. 固定隨機種子（確保可重現性）
# ------------------------------------------------------------------
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
    """執行一個 epoch 的驗證，回傳 loss、accuracy 及所有預測結果。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

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

            total_loss += loss.item()
            preds       = torch.argmax(logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)


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
    val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, device)

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
_, final_val_acc, final_preds, final_labels = eval_epoch(model, val_loader, device)

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

# ------------------------------------------------------------------
# 13. 儲存預測結果
# ------------------------------------------------------------------
output_df = pd.DataFrame({
    "row_id": test_df["row_id"],
    "label" : test_preds,
})
output_path = os.path.join(script_dir, "vBERT_finetune.csv")
output_df.to_csv(output_path, index=False)

print(f"\n預測結果已儲存至 vBERT_finetune.csv ({len(output_df)} 筆)")
print(f"預測分布: {pd.Series(test_preds).value_counts().to_dict()}")

# ------------------------------------------------------------------
# 14. 摘要
# ------------------------------------------------------------------
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
