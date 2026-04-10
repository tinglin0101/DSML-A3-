# vBERT Fine-tune — 使用 Training Set 微調 BERT 進行情感分類

## 模型概述

本版本（`vBERT_finetune`）在原始 vBERT 零樣本推論的基礎上，**引入監督式微調（Supervised Fine-tuning）**，利用 `train_2022.csv` 的標注資料對 `bert-base-uncased` 進行端到端訓練。

| 項目 | 說明 |
|------|------|
| 基礎模型 | `bert-base-uncased`（Google 原始預訓練） |
| 微調方式 | 全參數微調（All-layer fine-tuning）|
| 訓練資料 | `train_2022.csv` 的 **80%**（約 1600 筆）|
| 驗證資料 | `train_2022.csv` 的 **20%**（約 400 筆）|
| 預測目標 | `test_no_answer_2022.csv`（11001 筆）|
| 輸出檔案 | `vBERT_finetune.csv` |

---

## 資料切分策略：8:2 Stratified Split

### 切分方式

```
train_2022.csv（2001 筆，含標注）
         ↓  sklearn.model_selection.train_test_split
         ├── train_split（80%，~1600 筆）→ 用於微調 BERT
         └── val_split  （20%，~400  筆）→ 用於評估泛化性能
```

### 為什麼使用 Stratified Split？

使用 `stratify=train_full_df["LABEL"]` 確保切分後 train/val 的**正負樣本比例與原始資料一致**，避免切分偏差導致驗證集不代表真實分布。

```python
train_df, val_df = train_test_split(
    train_full_df,
    test_size=0.2,
    random_state=42,
    stratify=train_full_df["LABEL"],   # ← Stratified
)
```

---

## BERT 如何使用 Training Set 微調

### 1. 模型架構：BertForSequenceClassification

原始 `bert-base-uncased` 只有語言建模能力，**沒有分類頭**。微調時在其上加入一個線性分類層：

```
原始文本（TEXT）
    ↓
BertTokenizer（Tokenize + Padding + Truncation，max_length=128）
    ↓
[CLS] token₁ token₂ ... tokenₙ [SEP]
    ↓
BERT Encoder（12 層 Transformer Block）
    ↓
[CLS] 位置的 hidden state（768 維向量）
    ↓
Dropout（p=0.1）
    ↓
Linear Layer（768 → 2）  ← 新增的分類頭
    ↓
logits [neg_score, pos_score]
    ↓
argmax → label（0 or 1）
```

> [!NOTE]
> `[CLS]` token 的 hidden state 代表整句話的語義聚合表示（sentence-level representation），是 BERT 用於分類任務的標準做法。

### 2. 微調流程（Fine-tuning Pipeline）

```
Training Set（80%）= 1600 筆{TEXT, LABEL}
    ↓ DataLoader（batch_size=16, shuffle=True）
每個 Batch（16 筆）
    ↓
Forward Pass：
  - Tokenize → input_ids, attention_mask, token_type_ids
  - BERT Encoder 計算每個 token 的 hidden state
  - 取 [CLS] hidden state，通過 Dropout + Linear
  - 得到 logits（shape: [16, 2]）
    ↓
計算損失（Loss）：
  - CrossEntropyLoss(logits, labels)
    ↓
Backward Pass（Backpropagation）：
  - 計算梯度 ∂Loss/∂θ（對所有 BERT 層 + 分類頭的參數）
    ↓
參數更新（AdamW Optimizer）：
  - 更新所有 BERT 層參數（Embedding, Attention, FFN）
  - 更新分類頭參數（Linear weight & bias）
    ↓
Scheduler 調整 Learning Rate（Linear Warmup + Linear Decay）
```

### 3. 訓練超參數

| 超參數 | 值 | 說明 |
|--------|-----|------|
| `learning_rate` | 2e-5 | BERT 微調的標準推薦值 |
| `batch_size` | 16 | 顯存/記憶體與效率的平衡 |
| `epochs` | 3 | BERT 通常 3~5 epoch 已足夠收斂 |
| `max_length` | 128 | 短文本截斷，比 512 更有效率 |
| `warmup_ratio` | 0.1 | 前 10% steps 做 linear warmup，防止初始不穩定 |
| `weight_decay` | 0.01 | L2 正則化，防止過擬合 |
| `grad_clip` | 1.0 | 梯度裁剪，防止梯度爆炸 |

### 4. AdamW + Linear Warmup Scheduler

$$\theta_{t+1} = \theta_t - \eta_t \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) - \eta_t \cdot \lambda \cdot \theta_t$$

- **AdamW**：在 Adam 的基礎上將 L2 regularization 從梯度更新中分離，BERT 官方推薦使用
- **Linear Warmup**：前 10% steps 中 learning rate 從 0 線性增加到 2e-5，避免初始大幅更新破壞預訓練權重
- **Linear Decay**：後 90% steps 中 learning rate 從 2e-5 線性衰減到 0

```
Learning Rate 曲線示意：

  2e-5 ─────*─────────────────╲
            ↑                  ╲
         warmup               decay
         10%                  90%
  0 ────*                      *───
       start                  end
```

### 5. 損失函式：Cross-Entropy Loss

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} y_i \log \hat{p}_{i,1} + (1-y_i) \log \hat{p}_{i,0}$$

- $y_i \in \{0, 1\}$：真實標籤
- $\hat{p}_{i,c}$：模型預測類別 $c$ 的機率（softmax 後）
- 直接內建於 `BertForSequenceClassification` 中

### 6. Tokenization 細節

```python
encoding = tokenizer(
    text,
    truncation=True,       # 超過 max_length 截斷
    padding="max_length",  # 不足補 [PAD]
    max_length=128,
    return_tensors="pt",
)
```

輸入範例：

```
原始文本: "this headset is awesome, the sound quality is truly amazing."

Tokenize 後:
[CLS] this headset is awesome , the sound quality is truly amazing . [SEP] [PAD] ... [PAD]
  ↑                                                                     ↑              ↑
分類用的                                                              結束符         填充
聚合 token
```

---

## 與 vBERT 零樣本版本比較

| 比較項目 | vBERT（零樣本） | vBERT_finetune（本版）|
|----------|:--------------:|:--------------------:|
| 使用標注資料 | ❌ | ✅ train 的 80% |
| 微調方式 | 無（Zero-shot）| 全參數微調 |
| 推論機制 | MLM Prompting | 分類頭 Softmax |
| 模型結構 | BertForMaskedLM | BertForSequenceClassification |
| 期望準確率 | ~60–65% | ~85–90% |

> [!TIP]
> 微調（Fine-tuning）之所以顯著優於零樣本，是因為 BERT 預訓練的語義知識可以被**任務特定的標注資料引導和校準**，分類頭更能學到情感決策邊界。

---

## 推論流程（Test Set）

```
test_no_answer_2022.csv
    ↓
TestDataset（Tokenize，無標籤）
    ↓
DataLoader（batch_size=16，不 shuffle）
    ↓
最佳模型（best_bert_finetune.pt）Forward Pass
    ↓
argmax(logits) → label（0 or 1）
    ↓
vBERT_finetune.csv（row_id, label）
```

---

## 驗證策略（Validation Strategy）

每個 Epoch 結束後在 val_split（20%）上評估：

- **Val Loss**：若連續上升表示過擬合
- **Val Accuracy**：分類準確率
- **Best Model Saving**：以 val_acc 最高的 epoch 儲存 `best_bert_finetune.pt`

最終評估指標（最佳模型）：
- Accuracy、Precision、Recall、F1-Score
- 混淆矩陣（TP, FP, FN, TN）

---

## 使用方式

```bash
cd vBERT
pip install transformers torch pandas numpy scikit-learn matplotlib
python train_vBERT_finetune.py
```

執行後產生：
- `vBERT_finetune.csv`：測試集預測結果
- `vBERT_finetune.png`：訓練/驗證 Loss 與 Accuracy 曲線
- `best_bert_finetune.pt`：最佳模型權重

---

## 與其他版本比較

| 版本 | 方法 | 使用標注資料 | BERT 類型 |
|------|------|:-----------:|-----------|
| vBaseline | TF-IDF + Logistic Regression | ✅ train 全量 | — |
| vSBERT | Sentence-BERT + Cosine Sim | ❌ | 他人微調版 |
| vSelfTraining | TF-IDF + Self-Training | ✅ train（半監督）| — |
| vBERT（零樣本）| MLM Prompting | ❌ | Google 原始版 |
| **vBERT_finetune（本版）** | **BERT 微調** | **✅ 80% train** | **Google 原始版 + 任務微調** |

---

## 技術細節

- **預訓練模型**：`bert-base-uncased`，12 層 Transformer，隱藏層 768 維，12 個 attention head
- **全參數微調**：包含 BERT 所有層（Embedding + 12 Transformer Block + 分類頭）的參數均參與梯度更新
- **Stratified Split**：確保 train/val 正負樣本比例一致
- **早停機制**：以 val_acc 最高的 epoch 為最佳模型（非嚴格早停，但取最佳快照）
- **裝置支援**：自動偵測 GPU（`cuda`）/ CPU
- **依賴套件**：`transformers`, `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`
