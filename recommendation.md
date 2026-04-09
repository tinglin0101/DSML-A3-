# 情感分類模型改進紀錄

> 基於 `train_2022.csv`（2,000 筆，平衡標籤）的二元情感分類任務  
> 目標：從 V1 基線（~78% CV Accuracy）逐步提升

---

## ⚠️ 任務關鍵說明（重要，影響所有策略）

根據官方說明，這個任務有幾個非典型特性，**必須重新理解「正面/負面」的定義**：

### 1. 標籤由「整體語氣」決定，不是個別情緒詞

> "the plastic doesn't feel too comfortable but it's not bad, the light weight makes up for it." → **正面**

句子中有負面詞（`doesn't feel comfortable`），但因為**結尾偏正向**，整體判定為正面。  
**這代表傳統 TF-IDF 詞袋模型有根本性的弱點**：它計算的是詞頻，無法感知語氣的走向與結構。

### 2. 標籤邊界模糊（Vague Polarity）

沒有固定的正負面定義，即使人工標注也可能有分歧。這意味著：
- 模型不應追求在訓練集上的極致 overfitting
- 需要能捕捉**語境與語氣**的模型，而非單純詞頻統計

### 3. 訓練集極小、測試集極大（1:5.5）

- 訓練集：2,000 筆（含標籤）
- 測試集：11,000 筆（無標籤）
- 這個設定鼓勵使用 **半監督學習（Semi-supervised Learning）** 或 **Self-training**，利用大量無標籤的測試集來輔助訓練

### 4. 從資料觀察到的現象

| 現象 | 數值 |
|------|------|
| 含轉折詞（but/however/yet 等）的文本 | 265 筆（13.2%） |
| 這些轉折句中正面比例 | 51.3%（幾乎與整體相同）|
| 含 `num_extend` 特殊 token | 30 筆 |

轉折句中正負面幾乎各半，進一步確認：**決定標籤的是結尾語氣，不是轉折詞本身**。

---

## 資料集概況

| 項目 | 數值 |
|------|------|
| 訓練筆數 | 2,000 |
| 測試筆數 | 11,000 |
| 標籤分布 | 正面 (1) 1,000 筆 / 負面 (0) 1,000 筆（完全平衡） |
| 平均詞數 | 15.6 words |
| 中位詞數 | 13 words |
| 最長文本 | 48 words |
| 文本類型 | 電影評論 + 商品評論混合 |

---

## 版本紀錄

### V1 — 基線模型（已完成）

**準確率：~78%（5-fold CV）**

| 元件 | 設定 |
|------|------|
| 特徵提取 | TF-IDF，max_features=10000，ngram=(1,2)，sublinear_tf=True |
| 分類器 | Logistic Regression，C=1.0，max_iter=1000 |
| 驗證 | 5-fold Cross Validation |

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(C=1.0, max_iter=1000))
])
```

**根本限制（在任務說明更新後更加確定）：**
- TF-IDF 是詞袋模型，無法感知語氣方向（前段負面→後段正面 無法分辨）
- 否定詞處理雖有幫助，但不能解決「整體語氣走向」的問題
- 完全沒有利用 11,000 筆無標籤的測試資料

---

### V2 — 結尾語氣加權特徵（待實作）

**預期增益：+1~3%**  
**核心概念：** 針對「結尾語氣決定標籤」的特性，給文本後半段更高的權重

> ⚠️ 這版仍在 TF-IDF 框架內，屬於低成本但有根本上限的改法。建議作為快速實驗，若時間有限可直接跳 V3。

#### 2-1. 清理 `num_extend` token

```python
import re

def clean_text(text):
    text = text.replace('num_extend', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['TEXT'] = df['TEXT'].apply(clean_text)
```

#### 2-2. 拼接結尾句子作為額外特徵

```python
def extract_ending(text, ratio=0.4):
    """取文本後 40% 的詞，複製拼接強化結尾語氣的權重"""
    words = text.split()
    n = max(1, int(len(words) * ratio))
    ending = ' '.join(words[-n:])
    return text + ' ' + ending   # 結尾重複出現，TF-IDF 權重自然提高

df['TEXT_aug'] = df['TEXT'].apply(extract_ending)
```

> **原理：** TF-IDF 以詞頻計算權重，將結尾的詞重複拼接後，結尾詞的 TF 值上升，讓模型更重視文末語氣。這是在不改變模型架構的前提下，最直接對應任務特性的 trick。

#### 2-3. TF-IDF 參數調整

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),   # trigram 能抓到「makes up for it」這類尾句模式
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)
```

**實作結果：**
- CV Accuracy：`_____%`
- 備註：

---

### V3 — Sentence-BERT（強烈推薦優先實作）

**預期增益：+5~10%（相較 V1）**  
**核心概念：** SBERT 能理解整句的語意與語氣走向，天然適合「整體語氣」標籤定義

> ⭐ **根據任務說明，這版的優先級應高於 V2。** TF-IDF 的詞袋假設與本任務的標籤定義從根本上就不吻合，換成語意模型才能真正解決問題。不需 GPU，CPU 可執行。

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 不需 GPU，CPU 可跑
model = SentenceTransformer('all-MiniLM-L6-v2')

X = model.encode(df['TEXT'].tolist(), batch_size=64, show_progress_bar=True)
y = df['LABEL'].values

clf = LogisticRegression(C=1.0, max_iter=1000)
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f'CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}')
```

**備選 Sentence Transformer 模型：**

| 模型名稱 | 速度 | 品質 | 備注 |
|----------|------|------|------|
| `all-MiniLM-L6-v2` | 快 | 中高 | 推薦首選 |
| `all-mpnet-base-v2` | 中 | 高 | 品質更好，速度稍慢 |
| `paraphrase-MiniLM-L3-v2` | 最快 | 中 | 速度優先時使用 |

**實作結果：**
- CV Accuracy：`_____%`
- 使用模型：
- 備註：

---

### V4 — Self-Training（利用 11,000 筆無標籤測試資料）

**預期增益：+2~5%（疊加在 V3 之上）**  
**核心概念：** 用小量標籤資料訓練初始模型，對高信心的測試集預測打上「偽標籤」，再用擴充後的資料重新訓練

> ⭐ **這版直接回應任務說明的第三個挑戰**：訓練集小、測試集大。官方說明幾乎在暗示半監督方法是正確方向。

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

# 載入資料
train_df = pd.read_csv('train_2022.csv')
test_df = pd.read_csv('test_no_answer_2022.csv')

sbert = SentenceTransformer('all-MiniLM-L6-v2')
X_train = sbert.encode(train_df['TEXT'].tolist(), show_progress_bar=True)
X_test  = sbert.encode(test_df['TEXT'].tolist(), show_progress_bar=True)
y_train = train_df['LABEL'].values

CONFIDENCE_THRESHOLD = 0.90   # 只採用信心度 > 90% 的偽標籤
MAX_ITERATIONS = 5

for iteration in range(MAX_ITERATIONS):
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    confidence = proba.max(axis=1)
    pseudo_labels = proba.argmax(axis=1)

    # 只加入高信心樣本
    high_conf_mask = confidence > CONFIDENCE_THRESHOLD
    n_added = high_conf_mask.sum()
    print(f'Iter {iteration+1}: 高信心樣本 {n_added} 筆（threshold={CONFIDENCE_THRESHOLD}）')

    if n_added == 0:
        break

    # 擴充訓練集
    X_train = np.vstack([X_train, X_test[high_conf_mask]])
    y_train = np.concatenate([y_train, pseudo_labels[high_conf_mask]])

    # 將已加入的樣本從候選池移除
    keep_mask = ~high_conf_mask
    X_test  = X_test[keep_mask]
    test_df = test_df[keep_mask].reset_index(drop=True)

print(f'最終訓練集大小: {len(y_train)} 筆')
```

**超參數建議：**

| 參數 | 建議起點 | 可嘗試範圍 | 影響 |
|------|---------|-----------|------|
| `CONFIDENCE_THRESHOLD` | 0.90 | 0.85 ~ 0.95 | 過低會引入錯誤標籤，過高加入太少樣本 |
| `MAX_ITERATIONS` | 5 | 3 ~ 10 | 通常 3~5 輪後信心樣本會耗盡 |

**實作結果：**
- CV Accuracy（原始 train）：`_____%`
- 最終訓練集大小：`_____` 筆
- 使用迭代次數：
- 信心閾值：
- 備註：

---

### V5 — Fine-tune 預訓練語言模型（最高上限）

**預期準確率：88~93%**  
**需求：GPU（建議 ≥ 8GB VRAM）**

> 可在 V5 之上再疊加 Self-Training（V4 的概念），效果更佳。

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

MODEL_NAME = "roberta-base"   # 或 bert-base-uncased / distilbert-base-uncased

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding=True, max_length=128)

dataset = Dataset.from_dict({
    'text': train_df['TEXT'].tolist(),
    'label': train_df['LABEL'].tolist()
})
dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_strategy="epoch",
)
```

**備選預訓練模型：**

| 模型 | 大小 | 速度 | 適用情境 |
|------|------|------|---------|
| `distilbert-base-uncased` | 小 | 快 | 資源有限 |
| `bert-base-uncased` | 中 | 中 | 通用基線 |
| `roberta-base` | 中 | 中 | 通常比 BERT 強 |
| `deberta-v3-base` | 中 | 慢 | 追求最高精度 |

**實作結果：**
- Test Accuracy：`_____%`
- 使用模型：
- 最佳 epoch：
- 備註：

---

## 改進路線總覽（根據任務說明修訂）

```
V1 TF-IDF 基線 (~78%)
  │
  ├─► V2 結尾加權 + 參數調整 (+1~3%)      ← 低成本快速實驗，仍有根本限制
  │
  └─► V3 Sentence-BERT (+5~10%)           ← 優先跳到這裡，根本解決語意問題
        │
        └─► V4 Self-Training (+2~5%)       ← 利用 11,000 筆無標籤資料
              │
              └─► V5 Fine-tune BERT (~88~93%)
```

| 版本 | 核心改動 | 預期增益 | 優先級 | 狀態 |
|------|---------|---------|--------|------|
| V1 | TF-IDF + Logistic Regression | 基線 78% | — | ✅ 完成 |
| V2 | 結尾加權 + TF-IDF 調參 | +1~3% | 低（有根本限制） | ⬜ 待實作 |
| V3 | Sentence-BERT Embedding | +5~10% | ⭐⭐⭐ 高 | ⬜ 待實作 |
| V4 | Self-Training（利用測試集） | +2~5% | ⭐⭐⭐ 高 | ⬜ 待實作 |
| V5 | Fine-tune RoBERTa/BERT | ~88~93% | ⭐⭐⭐⭐ 最高 | ⬜ 待實作 |

---

## 實驗結果彙整

| 版本 | CV Accuracy | 備注 |
|------|------------|------|
| V1 | ~78% | 基線 |
| V2 | | |
| V3 | | |
| V4 | | |
| V5 | | |

---

## 注意事項與坑

- **最核心的坑**：TF-IDF 統計詞頻，無法感知「前負後正」的語氣走向，這是本任務的關鍵特性，TF-IDF 的進一步調優效益有限，應優先換語意模型
- **Self-training 的偽標籤風險**：信心閾值設太低（如 0.7）容易引入錯誤標籤，反而降分，建議從 0.90 開始
- **不需要處理 class imbalance**：資料完全平衡，各 1,000 筆
- **Fine-tune 時 2,000 筆偏少**：務必搭配 early stopping，epoch 不要超過 5
- **`num_extend`** 是數字遮蔽 token，對語氣預測無意義，建議清除
- **模糊邊界的影響**：CV 分數在任何版本下可能有 ±1~2% 的隨機波動，這是任務本身的標注模糊性造成的，非模型問題
