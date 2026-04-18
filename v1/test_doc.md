# End-to-End Sentiment Scoring Pipeline — v4

## 概述

本腳本實作一個三階段的情感評分 Pipeline，針對每筆文字資料進行：
1. 以轉折詞切割文本（Stanza）
2. 對每段文字產生 Sentence-BERT 嵌入向量（384 維）
3. 加權聚合後，以迴歸模型預測情感分數並評估

---

## 外部資源 / 依賴套件

| 套件 | 用途 | 備註 |
|------|------|------|
| [Stanza](https://stanfordnlp.github.io/stanza/) | NLP Pipeline：tokenize、POS、lemma、dependency parse | 首次執行自動下載英文模型 |
| [sentence-transformers](https://www.sbert.net/) | Sentence-BERT 嵌入向量 | 使用模型 `all-MiniLM-L6-v2`（384 維） |
| scikit-learn | 迴歸模型、交叉驗證、評估指標 | RandomForest / ElasticNet / GBM |
| NumPy / Pandas | 數值運算與資料讀寫 | — |

### 預訓練模型

- **`sentence-transformers/all-MiniLM-L6-v2`**：輕量 Sentence-BERT 模型，輸出 384 維正規化嵌入向量。若有 GPU，自動啟用。

---

## 系統架構

```
Input CSV (row_id, TEXT, LABEL)
        │
        ▼
┌───────────────────────────────┐
│  Step 1 — Text Splitting      │
│  Stanza dependency parse      │
│  → 以轉折詞 / clause-linking  │
│    "and" 切割句子為數段       │
└───────────────┬───────────────┘
                │ segments[]
                ▼
┌───────────────────────────────┐
│  Step 2 — Sentence-BERT       │
│  每段文字 → 384-dim embedding  │
│  加權聚合（weight scheme）    │
│  agg_emb = Σ emb_i × w(i)    │
└───────────────┬───────────────┘
                │ agg_emb (384-dim per row)
                ▼
┌───────────────────────────────┐
│  Step 3 — Regression Scoring  │
│  5-fold Stratified CV         │
│  模型：RF / ElasticNet / GBM  │
│  輸出 OOF scores + 最終預測   │
└───────────────┬───────────────┘
                │
                ▼
        Output CSV (row_scores_v4.csv)
```

---

## 詳細步驟說明

### Step 1 — 文本切割（`split_text`）

使用 Stanza 對每筆文字進行依存分析，尋找以下切割點：

- **轉折詞**（`TRANSITION_WORDS`）：but、however、although、though、yet、nevertheless、nonetheless、except、while、whereas
  - 詞性需為 `CCONJ`、`SCONJ` 或 `ADV`
- **子句連接用 "and"**：當 "and" 的 head 詞性為 `VERB` 或 `AUX` 時視為子句邊界

切割後回傳若干字串片段（segments）；若無切割點，回傳原始文字作為單一片段。

---

### Step 2 — Sentence-BERT 嵌入 + 加權聚合（`process_row`）

1. 對每個 segment 呼叫 `sbert.encode()`，取得正規化 384 維向量
2. 依 `weight_scheme` 計算各段權重 $w_i$
3. 聚合：$\text{agg\_emb} = \sum_{i=0}^{n-1} \text{emb}_i \times w_i$

#### 可用權重方案（`--weight`）

| 方案 | 公式 | 說明 |
|------|------|------|
| `sqrt`（預設）| $\sqrt{i+1}$ | 溫和遞增，較平衡 |
| `uniform` | $1$ | 所有段落等權 |
| `linear` | $i+1$ | 線性遞增，強調末段 |
| `log` | $\log(i+2)$ | 比 sqrt 更平緩 |
| `decay` | $e^{-0.5i}$ | 指數衰減，強調首段 |
| `last` | $1$（末段為 $3$）| 特別強調結尾 |
| `contrast` | 首末各 $2$，中間 $0.5$ | 強調開頭與結尾對比 |

---

### Step 3 — 迴歸評分（`run_cross_validation`）

1. 以 384 維聚合嵌入作為特徵矩陣 $X$，`LABEL` 為目標
2. 執行 **5-fold Stratified Cross-Validation**
3. 每個 fold 訓練一個 `StandardScaler + Regressor` Pipeline
4. 預測分數 clip 至 $[0, 1]$，以 $\geq 0.5$ 為正類閾值
5. 輸出每個 fold 的 Confusion Matrix 與各項指標
6. 最後在全資料集上 retrain 並輸出最終預測

#### 可選模型（`--model`）

| 名稱 | 模型 |
|------|------|
| `rf`（預設）| RandomForestRegressor（100 棵樹） |
| `elastic` | ElasticNet（α=0.01, l1_ratio=0.5） |
| `gbm` | GradientBoostingRegressor（100 棵樹） |

---

## 使用方式

```bash
# 基本執行（使用預設 RF 模型）
python test.py --input dataset_split/train_2022.csv --output row_scores_v4.csv

# 指定模型
python test.py --input dataset_split/train_2022.csv --model gbm

# 指定權重方案
python test.py --input dataset_split/train_2022.csv --weight contrast

# 完整參數
python test.py \
  --input  dataset_split/train_2022.csv \
  --output row_scores_v4.csv \
  --model  rf \
  --weight sqrt
```

### 輸入格式

CSV 檔案須包含以下欄位：

| 欄位 | 說明 |
|------|------|
| `row_id` | 整數，唯一識別碼 |
| `TEXT` | 待分析的文字 |
| `LABEL` | 情感標籤（0 或 1） |

### 輸出格式

| 欄位 | 說明 |
|------|------|
| `row_id` | 原始識別碼 |
| `row_label` | 原始標籤 |
| `oof_score` | Out-of-fold 預測分數 |
| `oof_pred_label` | OOF 預測標籤 |
| `final_score` | 全資料 retrain 後的分數 |
| `predicted_label` | 最終預測標籤 |

---

## 實驗結果（RF + sqrt weighting，5-fold CV）

### Overall OOF Confusion Matrix

```
[[668  332]
 [296  704]]
```

- 真陰性（TN）= 668，偽陽性（FP）= 332
- 偽陰性（FN）= 296，真陽性（TP）= 704

### 各項指標（mean ± std）

| 指標 | Mean | Std |
|------|------|-----|
| Accuracy  | 0.6860 | ±0.0098 |
| Precision | 0.6796 | ±0.0086 |
| Recall    | 0.7040 | ±0.0271 |
| F1 Score  | 0.6913 | ±0.0141 |
| AUC       | 0.7525 | ±0.0071 |
| MSE       | 0.2100 | ±0.0014 |

### 結果分析

- **AUC = 0.7525**：模型具備合理的區分能力，優於隨機猜測（0.5）
- **Recall（0.704）> Precision（0.680）**：模型傾向偵測正類，適合對漏判代價較高的情境
- **F1 = 0.6913**：精確率與召回率平衡良好
- **Std 值小**（0.007–0.027）：5-fold 結果穩定，模型無明顯過擬合
