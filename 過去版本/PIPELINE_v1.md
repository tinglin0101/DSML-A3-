# Pipeline_v1 文件

## 概覽

`pipeline.py` 是一個端對端的情感評分流程，將原始文字資料轉換為逐行的情感預測分數。

```
原始 CSV (TEXT, LABEL)
    │
    ▼  Step 1: Stanza 文字切割
分段後的文字 (SPLIT_TEXT)
    │
    ▼  Step 2: NRC 情緒分析
各段情緒特徵 (16 個欄位)
    │
    ▼  Step 3: 加權迴歸
row_scores.csv (final_score, predicted_label)
```

---

## 執行方式

```bash
# 基本執行（使用預設路徑與 bayesian 模型）
python pipeline.py

# 指定輸入、輸出、模型
python pipeline.py --input dataset_split/train_2022.csv --output row_scores.csv --model ridge

# 跳過測試集預測
python pipeline.py --test ""
```

### 參數一覽

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--input` | `dataset_split/train_2022.csv` | 訓練集 CSV，需含 `row_id`, `TEXT`, `LABEL` 欄位 |
| `--output` | `row_scores.csv` | 訓練集預測結果輸出路徑 |
| `--model` | `bayesian` | 迴歸模型選擇，見下方模型說明 |
| `--test` | `test_no_answer_2022.csv` | 測試集 CSV，需含 `row_id`, `TEXT`（無 LABEL） |
| `--test-output` | `test_predictions.csv` | 測試集預測結果輸出路徑 |

> **注意：** `--test` 預設非空，因此每次執行都會自動進行測試集預測。若要跳過，傳入 `--test ""`。

---

## 三個步驟詳解

### Step 1：文字切割（`split_text` / `run_split`）

使用 [Stanza](https://stanfordnlp.github.io/stanza/) 的 `tokenize + pos + lemma + depparse` 處理器，對每筆文字在**轉折詞**處進行切割，將一段長評論拆成語義上相對獨立的子段落。

**切割條件：**
- 一般轉折詞（詞性為 `CCONJ` / `SCONJ` / `ADV`）：
  `but`, `however`, `although`, `though`, `yet`, `nevertheless`, `nonetheless`, `except`, `while`, `whereas`
- 特殊處理 `and`：僅當 `and` 的依附詞（head）詞性為動詞（`VERB` / `AUX`）時才切割，避免誤切名詞並列句。

**輸出欄位：**

| 欄位 | 說明 |
|------|------|
| `SPLIT_TEXT` | Python list，每個元素為一個子段落字串 |
| `NUM_SPLITS` | 切割後的段落數量 |

---

### Step 2：NRC 情緒分析（`run_emotion_analysis`）

使用 [NRCLex](https://github.com/metalcorebear/NRCLex) 對每個子段落計算 8 種情緒的特徵，將 `SPLIT_TEXT` 展開成「每段落一列」的格式。

**8 種情緒：**
`Anger`, `Disgust`, `Fear`, `Joy`, `Sadness`, `Surprise`, `Trust`, `Anticipation`

**每種情緒產生 2 個特徵（共 16 欄）：**

| 欄位類型 | 範例 | 說明 |
|----------|------|------|
| `*_count` | `Anger_count` | 該情緒詞的原始出現次數 |
| `*_freq`  | `Anger_freq`  | 情緒詞佔全部情緒詞的頻率（0~1） |

**展開後的 schema：**

```
row_id | sub_id | split_text | LABEL | NUM_SPLITS | Anger_count | ... | Anticipation_freq
```

- `sub_id` 從 0 開始，代表該 `row_id` 的第幾個子段落。
- 測試集無 LABEL，`has_label=False` 時填入 `-1`。

---

### Step 3：加權迴歸評分（`build_weighted_features` + `run_regression`）

#### 3-1 建構加權寬表（`build_weighted_features`）

將同一 `row_id` 的所有 `sub_id` 特徵橫向展開成一列，形成「寬表」格式。

**sub_id 權重計算：**

```
weight(rank) = sqrt(rank + 1)
```

| sub_id（rank） | 權重（範例：4 段） |
|---------------|-------------------|
| 0（rank=0） | 1.00 |
| 1（rank=1） | 1.41 |
| 2（rank=2） | 1.73 |
| 3（rank=3） | 2.00 |

> 越晚出現的段落（結尾語意）賦予較高權重，但採平方根使成長較為平緩。

**輸出欄位格式：**

```
row_id | row_label | sub0_Anger_count | sub0_Anger_freq | ... | sub3_Anticipation_freq
```

- 若某 `row_id` 缺少特定 `sub_id`，對應特徵填 0。
- `row_label` 為該 `row_id` 所有子列 LABEL 的眾數（majority vote）。

#### 3-2 迴歸模型（`run_regression`）

在寬表特徵上訓練迴歸模型，預測連續分數後以 0.5 為閾值二元化。

**可選模型（`--model`）：**

| 選項 | 模型 | 特性 |
|------|------|------|
| `ridge` | Ridge（L2） | 穩定，保留所有特徵 |
| `lasso` | Lasso（L1） | 稀疏解，自動特徵選擇 |
| `elastic` | ElasticNet（L1+L2） | Ridge 與 Lasso 的折衷 |
| `bayesian` | BayesianRidge | 自動調整正則化強度（**預設**） |
| `gbm` | GradientBoosting | 非線性，捕捉特徵交互作用 |

**訓練策略：**
- `n ≥ 10`：80 / 20 train-val split，印出 MSE 與 AUC（若可計算）
- `n < 10`：全量訓練，僅印出 MSE

**預測輸出 schema：**

```
row_id | row_label | final_score | predicted_label
```

| 欄位 | 說明 |
|------|------|
| `final_score` | 迴歸預測值，clip 至 [0, 1] |
| `predicted_label` | `final_score >= 0.5` → 1，否則 0 |

---

## 測試集預測流程（`predict_test`）

當 `--test` 非空且檔案存在時，使用訓練好的 `pipe` 對測試集進行預測：

1. 以相同的 Stanza pipeline 切割測試集文字
2. 執行 NRC 情緒分析（`has_label=False`）
3. 使用**訓練時的 `sorted_sids` 與 `weight_map`** 建構寬表，確保特徵對齊
4. 補齊訓練集有但測試集缺少的特徵欄（填 0）
5. 以訓練好的 `pipe`（含 StandardScaler）預測

**測試集輸出 schema：**

```
row_id | final_score | predicted_label
```

---

## 輸出檔案

| 檔案 | 內容 |
|------|------|
| `row_scores.csv` | 訓練集逐行預測結果 |
| `test_predictions.csv` | 測試集逐行預測結果（無 LABEL） |
