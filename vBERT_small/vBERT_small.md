# vBERT_small — DistilBERT 情感分類微調說明文件

## 1. 專案概述

`vBERT_small` 是基於 **DistilBERT**（`distilbert-base-uncased`）的情感分類微調（Fine-tuning）方案。  
相較於完整的 BERT（12 層），DistilBERT 僅使用 **6 層 Transformer**，推理速度約快 60%，參數量減少約 40%，適合在資源有限的環境中快速實驗。

### 任務說明

| 項目 | 內容 |
|------|------|
| 任務類型 | 二元情感分類（Positive / Negative） |
| 基礎模型 | `distilbert-base-uncased` |
| 訓練資料 | `train_2022.csv`（標記資料） |
| 測試資料 | `test_no_answer_2022.csv`（無標籤） |
| 輸出檔案 | `vBERT_small.csv`（預測結果）、`vBERT_small.png`（訓練曲線） |

---

## 2. 模型架構

```
輸入文本
   │
   ▼
DistilBertTokenizer（Subword Tokenization）
   │  - 加入 [CLS]、[SEP] special token
   │  - Padding / Truncation 至 max_length=128
   ▼
DistilBERT Backbone（6 層 Transformer Encoder）
   │  - 輸出 [CLS] token 的 hidden state（dim=768）
   ▼
Linear Classification Head（768 → 2）
   │
   ▼
Softmax → 預測類別（0 = Negative, 1 = Positive）
```

> **注意**：DistilBERT 不支援 `token_type_ids`（原 BERT 用來區分句對），因此本實作不傳入此欄位。

---

## 3. 資料處理流程

### 3.1 資料集切分

```
train_2022.csv（全量標記資料）
       │
       ├─ 80% → train_split（用於微調）
       └─ 20% → val_split（用於驗證）
```

- 使用 `stratify=LABEL` 維持正負樣本比例一致性
- 固定 `random_state=42` 確保可重現性

### 3.2 Dataset 類別

| 類別 | 用途 | 輸出欄位 |
|------|------|----------|
| `SentimentDataset` | 訓練集 / 驗證集（含標籤） | `input_ids`, `attention_mask`, `labels` |
| `TestDataset` | 測試集（無標籤） | `input_ids`, `attention_mask` |

Tokenizer 設定：

```python
tokenizer(
    text,
    truncation=True,       # 超過 max_length 則截斷
    padding="max_length",  # 不足則補 [PAD]
    max_length=128,
    return_tensors="pt",
)
```

---

## 4. 超參數設定

| 超參數 | 數值 | 說明 |
|--------|------|------|
| `MODEL_NAME` | `distilbert-base-uncased` | 預訓練模型 |
| `MAX_LENGTH` | 128 | 輸入序列最大長度（token 數） |
| `BATCH_SIZE` | 16 | 每批次樣本數 |
| `EPOCHS` | 3 | 訓練輪數 |
| `LEARNING_RATE` | `2e-5` | AdamW 初始學習率 |
| `WARMUP_RATIO` | 0.1 | 前 10% steps 做 linear warmup |
| `VAL_SIZE` | 0.2 | 驗證集比例（20%） |
| `RANDOM_SEED` | 42 | 隨機種子 |

---

## 5. 訓練策略

### 5.1 Optimizer：AdamW

```python
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

使用 **AdamW**（Adam + Weight Decay 解耦），相較於標準 Adam 在 fine-tuning 大型預訓練模型時有較佳的泛化表現。

### 5.2 Learning Rate Scheduler：Linear Warmup + Linear Decay

```
Learning Rate
   ▲
   |      /\
   |     /  \
   |    /    \──────────
   |   /
   |──/
   +─────────────────────► Steps
     ↑warmup   ↑decay end
```

- **Warmup 階段**：前 10% steps 線性升溫，避免初期訓練不穩定
- **Decay 階段**：之後線性下降至 0，防止後期學習率過大導致遺忘

### 5.3 梯度裁剪（Gradient Clipping）

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

限制梯度最大範數為 1.0，防止梯度爆炸（Gradient Explosion）。

### 5.4 最佳模型儲存

以每個 epoch 結束後的 **Validation Accuracy** 為標準，儲存最高準確率的模型權重至 `best_vBERT_small.pt`。

---

## 6. 訓練與驗證流程

### 6.1 `train_epoch()`

```
對每個 batch：
  1. optimizer.zero_grad()
  2. Forward pass → 取得 loss（CrossEntropyLoss） 與 logits
  3. loss.backward()
  4. gradient clipping
  5. optimizer.step() + scheduler.step()
  6. 累計 loss 與 accuracy
```

### 6.2 `eval_epoch()`

```
torch.no_grad() 模式（不計算梯度，節省記憶體）
對每個 batch：
  1. Forward pass → 取得 loss 與 logits
  2. argmax(logits) → 預測標籤
  3. 累計 loss、accuracy、all_preds、all_labels
```

回傳：`(avg_loss, accuracy, all_preds_array, all_labels_array)`

---

## 7. 評估指標

最終以載入最佳模型後，在驗證集上計算以下指標：

| 指標 | 說明 |
|------|------|
| **Accuracy** | 整體預測正確率 |
| **Precision** | 預測為 Positive 中真正 Positive 的比例 |
| **Recall** | 所有真正 Positive 中被正確預測的比例 |
| **F1-Score** | Precision 與 Recall 的調和平均數 |
| **Confusion Matrix** | TP / FP / FN / TN 四格分類矩陣 |

混淆矩陣結構：

```
              預測 Negative  預測 Positive
實際 Negative  [ TN           FP ]
實際 Positive  [ FN           TP ]
```

---

## 8. 輸出檔案說明

| 檔案 | 說明 |
|------|------|
| `best_vBERT_small.pt` | 驗證集上最佳 Accuracy 的模型權重（PyTorch state_dict） |
| `vBERT_small.png` | 訓練曲線圖（Loss Curve + Accuracy Curve） |
| `vBERT_small.csv` | Test Set 預測結果，含 `row_id` 與 `label` 兩欄 |

`vBERT_small.csv` 格式：

```csv
row_id,label
0,1
1,0
2,1
...
```

---

## 9. 執行方式

### 環境需求

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib
```

### 執行指令

```bash
cd vBERT-finetune
python train_vBERT_small.py
```

### 目錄結構

```
DSML-A3-/
├── train_2022.csv              ← 標記訓練資料
├── test_no_answer_2022.csv     ← 無標籤測試資料
└── vBERT-finetune/
    ├── train_vBERT_small.py    ← 主程式
    ├── best_vBERT_small.pt     ← 最佳模型權重（執行後產生）
    ├── vBERT_small.csv         ← 預測結果（執行後產生）
    └── vBERT_small.png         ← 訓練曲線（執行後產生）
```

---

## 10. 程式碼結構總覽

| 區段 | 功能 |
|------|------|
| Section 0 | 超參數定義 |
| Section 1 | 固定隨機種子（可重現性） |
| Section 2 | 載入 `train_2022.csv` 與 `test_no_answer_2022.csv` |
| Section 3 | 8:2 切分訓練集與驗證集（stratified） |
| Section 4 | 定義 `SentimentDataset` 與 `TestDataset` |
| Section 5 | 載入 DistilBertTokenizer 與 DistilBertForSequenceClassification |
| Section 6 | 建立 DataLoader（train / val / test） |
| Section 7 | 設定 AdamW Optimizer + Linear Warmup Scheduler |
| Section 8 | 定義 `train_epoch()` 與 `eval_epoch()` |
| Section 9 | 主訓練迴圈（3 epochs，儲存最佳模型） |
| Section 10 | 最終驗證評估（Accuracy / Precision / Recall / F1 / CM） |
| Section 11 | 繪製並儲存訓練曲線圖 |
| Section 12 | 對 Test Set 推論，取得預測標籤 |
| Section 13 | 儲存預測結果為 `vBERT_small.csv` |
| Section 14 | 列印超參數與結果摘要 |

---

## 11. DistilBERT vs BERT 比較

| 特性 | BERT-base | DistilBERT |
|------|-----------|------------|
| Transformer 層數 | 12 | 6 |
| 參數量 | ~110M | ~66M |
| 推理速度 | 基準 | 約快 60% |
| 記憶體需求 | 較高 | 較低 |
| 支援 `token_type_ids` | ✅ | ❌ |
| 在本任務的準確率 | 略高 | 相近（差距約 1-2%） |

DistilBERT 透過**知識蒸餾（Knowledge Distillation）**從 BERT 學習，在多數 NLP 任務上保持 97% 的 BERT 性能，是資源受限場景下的首選。
