# baseline(原v8) — Sentiment Scoring Pipeline

## 概述

本 pipeline 以兩階段微調的 Sentence-BERT（SBERT）為核心，對文本進行情感二元分類（LABEL 0/1），並使用 Random Forest 迴歸模型輸出最終預測標籤。

---

## 資料

| 檔案 | 說明 |
|------|------|
| `train_2022.csv` | 訓練集，2000 筆，含 `TEXT`、`LABEL`、`row_id` 欄位 |
| `test_no_answer_2022.csv` | 測試集，11000 筆，無 `LABEL` |

---

## Pipeline 架構

```
原始文本
   │
   ├── [Phase 1] MLM Domain Adaptation
   │       ├─ 使用 train + test 全部文本（無標籤）
   │       ├─ 以 Masked Language Modeling 讓 MPNet 適應領域詞彙
   │       └─ 儲存 checkpoint：baseline_da_model.pt
   │
   ├── [Phase 2] Supervised Fine-tuning
   │       ├─ 從 train 中採樣 3000 對文本對
   │       │     same label → similarity = 1.0
   │       │     diff label → similarity = 0.0
   │       ├─ 使用 CosineSimilarityLoss 微調
   │       └─ 儲存 checkpoint：baseline_ft_model.pt
   │
   ├── [Embedding] 將每筆文本編碼為 768-dim 向量
   │
   └── [ML] 5-Fold Stratified CV + Random Forest Regressor
               └─ 輸出最終測試集預測 result_baseline.csv
```

---

## 兩階段微調詳解

### Phase 1：MLM Domain Adaptation

- **目的**：讓預訓練 MPNet 熟悉任務領域的詞彙與語言風格
- **方式**：以 15% 機率隨機遮蔽 token，訓練模型還原（MLM 目標）
- **資料**：train + test 共 13000 筆文本（不使用標籤）
- **設定**：`epochs=1`、`batch=8`、`lr=3e-5`、`mlm_prob=0.15`
- **可跳過**：設 `skip_da=True`；或載入已存檔：設 `load_da=<路徑>`

### Phase 2：Supervised Fine-tuning

- **目的**：讓 SBERT 將相同情感的句子嵌入拉近，不同情感的推遠
- **方式**：CosineSimilarityLoss（相同標籤對 → 1.0，不同 → 0.0）
- **資料**：從 train 隨機採樣 3000 對（正正、負負各半；正負剩餘）
- **設定**：`epochs=3`、`batch=16`、`lr=2e-5`、`warmup_steps=100`

---

## Embedding 策略

每筆文本直接編碼為單一 768-dim L2-normalized 向量。

---

## ML 模型（Step 3）

- **輸入特徵**：768-dim SBERT 向量
- **模型**：`RandomForestRegressor(n_estimators=100)`（可換 `elastic`、`gbm`）
- **評估**：5-Fold Stratified Cross-Validation
- **閾值**：score ≥ 0.5 → LABEL=1

---

## 主要設定（CFG）

```python
CFG = {
    "model":       "rf",          # rf | elastic | gbm
    "skip_da":     False,         # 是否跳過 Phase 1
    "load_da":     None,          # 載入已存 Phase 1 checkpoint 路徑
    "skip_ft":     False,         # 是否只做 Phase 1 後退出
    "da_epochs":   1,
    "da_batch":    8,
    "da_lr":       3e-5,
    "da_mlm_prob": 0.15,
    "ft_epochs":   3,
    "ft_lr":       2e-5,
    "ft_batch":    16,
    "ft_pairs":    3000,
    "ft_warmup":   100,
}
```

---

## 輸出檔案

| 檔案 | 說明 |
|------|------|
| `baseline_da_model.pt` | Phase 1 Domain-Adapted SBERT checkpoint |
| `baseline_ft_model.pt` | Phase 2 Fine-tuned SBERT checkpoint |
| `result_baseline.csv` | 測試集預測結果（`row_id`, `LABEL`） |

---

## 依賴套件

```
sentence-transformers
transformers
torch
scikit-learn
pandas
numpy
vaderSentiment
stanza
```

> `stanza` 與 `vaderSentiment` 在 baseline 中暫時停用（`VADER_FEATURES = []`），保留 import 供未來版本使用。
