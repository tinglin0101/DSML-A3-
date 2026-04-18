# 文本切割模組文件 — split_text_stanza.py

## 概述

本模組使用 [Stanza](https://stanfordnlp.github.io/stanza/) 對英文文本進行**語義導向的子句切割**，目標是在轉折詞與子句連接用 "and" 處將一句話拆成語義相對獨立的片段，供後續情感分析使用。

模組共實作三個版本的切割函數，最終以 `split_text_by_transition_v3`（v3）作為主要切割邏輯。

---

## 外部資源 / 依賴套件

| 套件 | 用途 |
|------|------|
| [Stanza](https://stanfordnlp.github.io/stanza/) | 英文 NLP Pipeline：tokenize、POS、lemma、dependency parse、NER |
| Pandas | CSV 讀寫 |

### Stanza Processors 說明

| Processor | 功能 | 用途 |
|-----------|------|------|
| `tokenize` | 分詞與句子邊界偵測 | 基礎文字切分 |
| `pos` | 詞性標注（UPOS） | 判斷詞的語法角色 |
| `lemma` | 詞形還原 | 讓 "however" / "However" 等變體都能被正確辨認 |
| `depparse` | 依存句法分析 | 取得每個詞的 `head` 與 `deprel`，用於 "and" 的三層切割判斷 |
| `ner` | 命名實體識別 | v3 用於 Entity Masking，避免實體名稱內的關鍵詞被誤切 |

---

## 版本演進

| 版本 | 核心策略 | 差異 |
|------|----------|------|
| v1（已棄用）| 僅依轉折詞詞性切割 | 無 "and" 特殊處理，無 NER 保護 |
| **v2(表現最好)** | 轉折詞 + "and" 子句判斷 | 新增 "and" 依 head 詞性切割，但無 NER 保護 |
| **v3（現行）** | v2 + Entity Masking + 三層 "and" 判斷 | 最完整，遮罩命名實體後再切割，減少誤切 |

---

## 切割流程（v3）

```
原始文字
    │
    ▼
┌──────────────────────────────────────┐
│  Step 1 — NER Masking                │
│  用 Stanza NER 找出命名實體          │
│  將其替換為 __ENT_0__、__ENT_1__... │
│  避免實體名稱中的轉折詞被誤切        │
└─────────────────┬────────────────────┘
                  │ masked_text + mapping
                  ▼
┌──────────────────────────────────────┐
│  Step 2 — Syntax Parse               │
│  對遮罩後的文字重新執行              │
│  POS + Dependency Parsing            │
│  （若無實體，直接複用原始 doc）      │
└─────────────────┬────────────────────┘
                  │ doc（含 upos / head / deprel）
                  ▼
┌──────────────────────────────────────┐
│  Step 3 — Smart Split                │
│  依切割規則找出切割點（char index）  │
│  → 見下方詳細說明                   │
└─────────────────┬────────────────────┘
                  │ parts（遮罩文字片段）
                  ▼
┌──────────────────────────────────────┐
│  Step 4 — Unmask                     │
│  將各片段中的 __ENT_N__ 佔位符       │
│  還原為原始實體文字                  │
└─────────────────┬────────────────────┘
                  │
                  ▼
        list[str]（切割後的段落）
```

---

## 切割規則詳解（`_smart_split`）

### 規則 A — 強轉折詞（直接切割）

觸發詞彙：

> `but`、`however`、`although`、`though`、`yet`、`nevertheless`、`nonetheless`、`except`、`while`、`whereas`

**判斷條件**：
1. 該詞或其 lemma 屬於上述轉折詞
2. UPOS 詞性為 `CCONJ`（對等連接詞）、`SCONJ`（從屬連接詞）或 `ADV`（副詞）
3. 出現位置不在句首（`start_char > 0`）

**設計原因**：這些詞幾乎在任何語境下都引導一個對立或讓步子句，語義轉折明確，直接以字元位置為切割點。

```
範例：
"The movie was enjoyable but the ending was disappointing."
                           ↑ 切割點
→ ["The movie was enjoyable ", "but the ending was disappointing."]
```

---

### 規則 B — "and" 三層判斷

"and" 語境多樣（連接名詞、動詞、子句皆可），需依句法結構謹慎判斷。

#### 層 1 — `deprel` 必須為 `cc`

只有當 "and" 的依存關係標籤（`deprel`）為 `cc`（協調連接詞）時才進入後續判斷。其他標籤（如 `conj`、`det`）直接保留，不切割。

#### 層 2 — 查看 "and" 的 head（稱為 B）的詞性

| B 的 UPOS | 決策 | 原因 |
|-----------|------|------|
| `NOUN`、`PROPN`、`ADJ`、`ADV`、`NUM` | **不切割** | "and" 連接名詞短語（如 "cats and dogs"） |
| `VERB`、`AUX` | 進入層 3 | 可能是子句並列，需進一步確認 |
| 其他（`PART`、`DET` 等） | **不切割** | 保守策略，避免誤切 |

#### 層 3 — B 為動詞時，確認 B 的 deprel

當 B 的 `deprel == "conj"`（B 本身是另一個動詞的並列節點），需往上看 B 的 head（稱為 A）：

| A 的 UPOS | 決策 | 原因 |
|-----------|------|------|
| 名詞類（NOUN、PROPN 等）| **不切割** | B 是被誤判的動詞形式名詞（如 "swingers and go"），實為名詞短語並列 |
| 動詞類（VERB、AUX）| **切割** | A 和 B 都是動詞，確認為真實的動詞子句並列 |

當 B 的 `deprel != "conj"`（例如 ROOT、ccomp），B 直接是真實動詞子句根節點，直接切割。

```
範例（切割）：
"She laughed and he cried."
              ↑ and.deprel=cc, B=cried(VERB/ROOT) → 切割
→ ["She laughed ", "and he cried."]

範例（不切割）：
"She bought apples and oranges."
                    ↑ and.deprel=cc, B=oranges(NOUN) → 層2 名詞類 → 不切割
→ ["She bought apples and oranges."]

範例（不切割，層3）：
"She likes swingers and go dancing."
                       ↑ and.deprel=cc, B=go(VERB/conj), A=swingers(NOUN) → 層3 名詞 → 不切割
→ ["She likes swingers and go dancing."]
```

---

## Entity Masking 機制（`mask_entities` / `unmask_entities`）

### 問題背景

部分命名實體的名稱中含有轉折詞，例如：
- 公司名：`"Yet Another Framework Inc."`
- 人名：`"But Williams"`

若不保護，Step 3 會將實體名內的 "Yet" 或 "But" 誤判為切割點。

### 遮罩流程

1. 用 Stanza NER 取得所有命名實體的字元範圍 `(start_char, end_char, text)`
2. **由後往前**替換（避免替換後字元偏移量錯亂）
3. 每個實體替換為 `__ENT_0__`、`__ENT_1__`...，並記錄 `mapping`
4. 對遮罩後的文字進行句法分析與切割
5. 切割完成後，將各片段中的佔位符還原為原始實體文字

---

## 使用方式

```bash
# 使用預設輸入/輸出路徑
python split_text_stanza.py

# 指定輸入/輸出檔案
python split_text_stanza.py \
  --input  train_2022.csv \
  --output dataset_split/train_2022_split_stanza_v4.csv
```

### 輸入格式

| 欄位 | 說明 |
|------|------|
| `TEXT` | 原始英文文字 |
| 其他欄位 | 原樣保留 |

### 輸出格式

| 欄位 | 說明 |
|------|------|
| `SPLIT_TEXT` | 切割後的段落清單（Python list，序列化為字串） |
| `NUM_SPLITS` | 切割後的段落數 |
| 其他欄位 | 原樣保留 |

---

## 程式介面

```python
from split_text_stanza import split_text_by_transition_v3
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse,ner')

text = "The product was well-designed but the price was too high."
parts = split_text_by_transition_v3(text, nlp)
# → ["The product was well-designed ", "but the price was too high."]
```

---

## 設計注意事項

- **`start_char > 0` 保護**：避免句首就是轉折詞時產生空片段
- **由後往前 Masking**：確保實體替換不影響其他實體的字元偏移
- **重新解析遮罩後文字**：Masking 改變文字結構，必須重新執行 Stanza 以取得正確的依存關係
- **保守策略**：對於詞性不明確的 "and"，預設不切割，避免破壞語意完整性
