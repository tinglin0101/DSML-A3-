# vSBERT 情感分類模型報告

## 任務說明

根據 `train_2022.csv` 訓練一個二元情感分類模型，預測 `test_no_answer_2022.csv` 中文本的正負面標籤（0/1），結果輸出至 `vSBERT.csv`。

本版使用 **Sentence-BERT** 取代 V1 的 TF-IDF 特徵提取，以解決詞袋模型無法捕捉「整體語氣走向」的根本問題。

## 資料概述

| 項目 | 說明 |
|------|------|
| 訓練集 | `train_2022.csv`，2,000 筆，含 `row_id`、`TEXT`、`LABEL` |
| 測試集 | `test_no_answer_2022.csv`，11,000 筆，含 `row_id`、`TEXT` |
| 標籤分布 | 正面 (1) 與負面 (0) 各 1,000 筆（完全平衡） |

## 模型架構

### 特徵提取 — Sentence-BERT

| 參數 | 設定 |
|------|------|
| 模型名稱 | `all-MiniLM-L6-v2` |
| Embedding 維度 | 384 |
| Batch Size | 64 |
| 運算需求 | CPU 即可執行（不需 GPU） |

**為何選擇 Sentence-BERT？**

- **語意理解**：SBERT 將整句文本編碼為固定長度的向量，能捕捉語序、語氣轉折等資訊
- **解決 V1 的根本限制**：本任務的標籤由「整體語氣」決定，例如 "the plastic doesn't feel too comfortable but it's not bad, the light weight makes up for it" 標記為正面。TF-IDF 詞袋模型無法分辨此類前負後正的語氣走向，而 SBERT 天然具備此能力
- **不需 GPU**：`all-MiniLM-L6-v2` 是輕量模型，CPU 可在合理時間內完成編碼

**備選模型比較：**

| 模型名稱 | 速度 | 品質 | 備注 |
|----------|------|------|------|
| `all-MiniLM-L6-v2` | 快 | 中高 | ✅ 本版使用 |
| `all-mpnet-base-v2` | 中 | 高 | 品質更好，速度稍慢 |
| `paraphrase-MiniLM-L3-v2` | 最快 | 中 | 速度優先時使用 |

### 分類器 — Logistic Regression

| 參數 | 設定 |
|------|------|
| 正則化參數 C | 1.0 |
| 最大迭代次數 | 1,000 |

## 訓練流程

```
載入資料 → SBERT 句向量編碼 → 5-fold 交叉驗證 → 全量訓練 → 預測測試集 → 輸出 vSBERT.csv
```

## 與 V1 的差異

| 項目 | V1 (TF-IDF) | vSBERT |
|------|-------------|--------|
| 特徵提取 | TF-IDF (詞袋) | Sentence-BERT (語意) |
| 特徵維度 | 10,000 (稀疏) | 384 (稠密) |
| 語序感知 | ❌ 否 | ✅ 是 |
| 語氣轉折 | ❌ 無法識別 | ✅ 可識別 |
| 預期 CV Accuracy | ~78% | ~83-88% |
| 運算時間 | 數秒 | 數分鐘（CPU） |

## 模型表現

| 指標 | 結果 |
|------|------|
| 5-fold CV Accuracy | `_____`（執行後填入） |

## 輸出檔案

- **檔案名稱**：`vSBERT.csv`
- **欄位**：`row_id`, `label`
- **筆數**：11,000

## 程式碼

完整程式碼見 [`train_vSBERT.py`](./train_vSBERT.py)。

## 依賴套件

```bash
pip install sentence-transformers scikit-learn pandas numpy
```

## 執行方式

```bash
cd <project-root>
python vSBERT/train_vSBERT.py
```

> ⚠️ 首次執行時會自動下載 `all-MiniLM-L6-v2` 模型（~80MB），需要網路連線。
