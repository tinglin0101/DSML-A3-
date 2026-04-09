# vBERT — Zero-Shot 原始 Google BERT 情感分類（MLM Prompting）

## 模型概述

vBERT 使用 Google 原始發布的 `bert-base-uncased`，**完全不做任何微調**，透過 **Masked Language Model Prompting** 進行零樣本情感分類。

| 項目 | 說明 |
|------|------|
| 模型 | `bert-base-uncased`（Google 原始發布） |
| 微調 | ❌ 完全無（Zero-Shot） |
| 推論方法 | MLM Prompting（遮罩語言模型提示） |
| Validation Set | `train_2022.csv`（2001 筆，含 LABEL） |
| 預測目標 | `test_no_answer_2022.csv`（11001 筆） |
| 輸出檔案 | `vBERT.csv` |

---

## 為什麼選用原始 BERT？

原始 `bert-base-uncased` 是 Google 2018 年發布的預訓練語言模型，**僅經過以下兩種預訓練任務**，從未接觸過任何情感分類資料：

1. **Masked Language Model (MLM)**：隨機遮蓋 15% token，預測被遮蓋的詞
2. **Next Sentence Prediction (NSP)**：預測兩句是否相連

這代表模型對詞彙的情感傾向是**透過大量英文語料自然習得**的，而非由人工標注驅動。

---

## 推論方法：MLM Prompting

### 核心思想

既然原始 BERT 沒有分類頭，改為利用其 **MLM 頭（語言建模能力）** 來做推論。

### Prompt 設計

```
{原始文本} Overall, it was [MASK].
```

範例：

| 輸入文本 | Prompt |
|----------|--------|
| `"great quality for the price"` | `"great quality for the price Overall, it was [MASK]."` |
| `"terrible product, broke in 2 days"` | `"terrible product, broke in 2 days Overall, it was [MASK]."` |

### 決策邏輯

BERT 對 `[MASK]` 位置輸出整個詞彙表的機率分佈，我們比較：

$$\text{score}_{\text{pos}} = \sum_{w \in \text{POS\_WORDS}} P(w \mid \text{context})$$

$$\text{score}_{\text{neg}} = \sum_{w \in \text{NEG\_WORDS}} P(w \mid \text{context})$$

$$\text{label} = \begin{cases} 1 & \text{if } \text{score}_\text{pos} \geq \text{score}_\text{neg} \\ 0 & \text{otherwise} \end{cases}$$

### 情感詞彙集

| 類別 | 詞彙 |
|------|------|
| **正面（label=1）** | great, good, excellent, wonderful, fantastic, amazing, superb, brilliant, outstanding, perfect, positive, enjoyable, impressive, beautiful, love |
| **負面（label=0）** | bad, terrible, awful, horrible, poor, disappointing, dreadful, worse, worst, useless, negative, boring, ugly, hate, inferior |

> 只保留在 BERT 詞彙表中為**單一 token** 的詞（確保機率可正確讀取）

---

## 推論流程

```
輸入文本
   ↓
構造 Prompt: "{text} Overall, it was [MASK]."
   ↓
BertTokenizer（tokenize + truncate，max_length=512）
   ↓
BertForMaskedLM（12-layer Transformer + MLM head）
   ↓
取出 [MASK] 位置的 logits → softmax → 機率分佈
   ↓
正面詞機率加總 vs. 負面詞機率加總
   ↓
label = 1（正面較高）或 0（負面較高）
```

---

## 與其他版本比較

| 版本 | 方法 | 使用標注資料 | BERT 類型 |
|------|------|:-----------:|-----------|
| vBaseline | TF-IDF + Logistic Regression | ✅ train | — |
| vSBERT | Sentence-BERT + Cosine Sim | ❌ | 他人微調版 |
| vSelfTraining | TF-IDF + Self-Training | ✅ train | — |
| vBERT（前版） | DistilBERT SST-2 | ❌ | 他人微調版 |
| **vBERT（本版）** | **MLM Prompting** | **❌** | **Google 原始版** |

---

## 技術細節

- **截斷策略**：max_length=512，超過部分截斷（`[MASK]`位置若被截斷則 fallback 為 label=1）
- **裝置**：自動偵測 GPU（`cuda`）/ CPU
- **逐筆推論**：每筆 prompt 長度不同，逐筆處理確保正確性
- **依賴套件**：`transformers`, `torch`, `pandas`, `numpy`

---

## 使用方式

```bash
cd vBERT
pip install transformers torch pandas numpy
python train_vBERT.py
```

執行後產生 `vBERT.csv`：

```
row_id,label
0,1
1,0
...
```

## 實驗結果

1 : 0 = 1993 : 7