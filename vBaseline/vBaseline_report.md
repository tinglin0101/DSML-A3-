# V1 情感分類模型報告

## 任務說明

根據 `train_2022.csv` 訓練一個二元情感分類模型，預測 `test_no_answer_2022.csv` 中文本的正負面標籤（0/1），結果輸出至 `v1.csv`。

## 資料概述

| 項目 | 說明 |
|------|------|
| 訓練集 | `train_2022.csv`，2,000 筆，含 `row_id`、`TEXT`、`LABEL` |
| 測試集 | `test_no_answer_2022.csv`，11,000 筆，含 `row_id`、`TEXT` |
| 標籤分布 | 正面 (1) 與負面 (0) 大致均衡 |

## 模型架構

### 特徵提取 — TF-IDF
- 最大特徵數：10,000
- N-gram 範圍：(1, 2)（unigram + bigram）
- 使用 sublinear TF（對詞頻取 log）

### 分類器 — Logistic Regression
- 正則化參數 C = 1.0
- 最大迭代次數：1,000

## 訓練流程

```
載入資料 → TF-IDF 向量化 → 5-fold 交叉驗證 → 全量訓練 → 預測測試集 → 輸出 v1.csv
```

## 模型表現

| 指標 | 結果 |
|------|------|
| 5-fold CV Accuracy | ≈ 78% |

## 輸出檔案

- **檔案名稱**：`v1.csv`
- **欄位**：`row_id`, `label`
- **筆數**：11,000

## 程式碼

完整程式碼見 [`train_and_predict_v1.py`](file:///c:/Users/tinal/Github_DB/note/A3/train_and_predict_v1.py)。
