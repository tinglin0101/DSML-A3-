# =============================================================
# vBERT — 使用原始 Google BERT 的零樣本情感分類
#         (MLM Prompting，不依賴任何他人微調的模型)
#
# 策略：
#   使用 Google 原始發布的 bert-base-uncased（純 MLM 預訓練，
#   無任何分類頭、無情感資料微調）進行零樣本情感推論。
#
# 推論方法 — Masked Language Model (MLM) Prompting:
#   將文本轉換為 prompt 格式：
#       "[TEXT] Overall, it was [MASK]."
#   由 BERT 的 MLM 頭預測 [MASK] 位置的詞彙分佈，
#   比較正面詞彙組（good, great, excellent, ...）與
#   負面詞彙組（bad, terrible, awful, ...）的機率總和，
#   機率較高的一方決定 label（1=正面, 0=負面）。
#
# 以 train_2022.csv 作為 validation set 評估準確率，
# 再對 test_no_answer_2022.csv 輸出 vBERT.csv。
# =============================================================

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM

# ------------------------------------------------------------------
# 1. 載入資料
# ------------------------------------------------------------------
print("載入資料...")
train_df = pd.read_csv("train_2022.csv")   # 作為 validation set
test_df  = pd.read_csv("test_no_answer_2022.csv")

print(f"Validation set 大小 (train_2022): {len(train_df)}")
print(f"Test  set 大小 (test_no_answer_2022): {len(test_df)}")
print(f"Validation label 分布:\n{train_df['LABEL'].value_counts()}\n")

# ------------------------------------------------------------------
# 2. 載入原始 Google BERT（bert-base-uncased）
#    - 純 MLM 預訓練，無任何分類微調
#    - 無情感分類頭，完全依賴語言模型機率做推論
# ------------------------------------------------------------------
MODEL_NAME = "bert-base-uncased"
print(f"載入原始 Google BERT: {MODEL_NAME} ...")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model     = BertForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"模型載入完成，使用裝置: {device}\n")

# ------------------------------------------------------------------
# 3. 定義情感詞彙集
#    正面詞彙 → label=1，負面詞彙 → label=0
#    只保留在 BERT 詞彙表中為單一 token 的詞
# ------------------------------------------------------------------
POSITIVE_WORDS = [
    "great", "good", "excellent", "wonderful", "fantastic",
    "amazing", "superb", "brilliant", "outstanding", "perfect",
    "positive", "enjoyable", "impressive", "beautiful", "love",
]

NEGATIVE_WORDS = [
    "bad", "terrible", "awful", "horrible", "poor",
    "disappointing", "dreadful", "worse", "worst", "useless",
    "negative", "boring", "ugly", "hate", "inferior",
]


def get_token_ids(words):
    """取得詞彙在 BERT 詞彙表中的 token id（只保留單 token 詞）。"""
    ids = []
    for w in words:
        tok = tokenizer.tokenize(w)
        if len(tok) == 1:
            ids.append(tokenizer.convert_tokens_to_ids(tok[0]))
    return ids


pos_ids = get_token_ids(POSITIVE_WORDS)
neg_ids = get_token_ids(NEGATIVE_WORDS)

# print(f"正面詞 (共 {len(pos_ids)} 個): {[tokenizer.convert_ids_to_tokens(i) for i in pos_ids]}")
# print(f"負面詞 (共 {len(neg_ids)} 個): {[tokenizer.convert_ids_to_tokens(i) for i in neg_ids]}")
# print()

# ------------------------------------------------------------------
# 4. MLM Prompting 推論函式
# ------------------------------------------------------------------
MASK_TOKEN = tokenizer.mask_token    # "[MASK]"
MAX_LENGTH = 512


def build_prompt(text: str) -> str:
    """建立包含 [MASK] 的 prompt。"""
    return f"{text.strip()} Overall, it was {MASK_TOKEN}."


def predict_single(text: str) -> int:
    """
    對單筆文本做 MLM prompting 情感推論。
    回傳 1（正面）或 0（負面）。
    """
    prompt = build_prompt(text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    input_ids     = inputs["input_ids"][0]
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

    # 若 prompt 因截斷導致 [MASK] 消失，回傳預設值
    if len(mask_positions) == 0:
        return 1

    mask_pos = mask_positions[-1].item()

    with torch.no_grad():
        logits = model(**inputs).logits[0, mask_pos, :]  # [vocab_size]

    probs     = torch.softmax(logits, dim=-1)
    pos_score = probs[pos_ids].sum().item()
    neg_score = probs[neg_ids].sum().item()

    return 1 if pos_score >= neg_score else 0


def predict_labels(texts, label=""):
    """對一系列文本逐筆執行 MLM prompting 情感推論。"""
    results    = []
    texts_list = texts.tolist() if hasattr(texts, "tolist") else list(texts)
    total      = len(texts_list)

    for i, text in enumerate(texts_list):
        results.append(predict_single(text))
        if (i + 1) % 200 == 0 or i + 1 == total:
            print(f"  [{label}] 進度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

    return np.array(results)


# ------------------------------------------------------------------
# 5. Validation — 在 train_2022.csv 上評估準確率
# ------------------------------------------------------------------
print("=" * 60)
print("Validation (train_2022.csv)")
print("=" * 60)

val_preds = predict_labels(train_df["TEXT"], label="Validation")
val_true  = train_df["LABEL"].values

accuracy  = (val_preds == val_true).mean()
print(f"\nValidation Accuracy: {accuracy:.4f}")

# 混淆矩陣
tp = int(((val_preds == 1) & (val_true == 1)).sum())
tn = int(((val_preds == 0) & (val_true == 0)).sum())
fp = int(((val_preds == 1) & (val_true == 0)).sum())
fn = int(((val_preds == 0) & (val_true == 1)).sum())

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n混淆矩陣:")
print(f"  TP={tp}  FP={fp}")
print(f"  FN={fn}  TN={tn}")
print(f"\nPrecision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"\nValidation 預測分布: {pd.Series(val_preds).value_counts().to_dict()}")

# ------------------------------------------------------------------
# 6. 對 test_no_answer_2022.csv 做最終預測
# ------------------------------------------------------------------
# print("\n" + "=" * 60)
# print("推論 test_no_answer_2022.csv")
# print("=" * 60)

# test_preds = predict_labels(test_df["TEXT"], label="Test")

# # ------------------------------------------------------------------
# # 7. 儲存結果
# # ------------------------------------------------------------------
# output_df = pd.DataFrame({
#     "row_id": test_df["row_id"],
#     "label" : test_preds,
# })
# output_df.to_csv("vBERT.csv", index=False)

# print(f"\n預測結果已儲存至 vBERT.csv ({len(output_df)} 筆)")
# print(f"預測分布: {pd.Series(test_preds).value_counts().to_dict()}")

# ------------------------------------------------------------------
# 8. 摘要
# ------------------------------------------------------------------
# print("\n" + "=" * 60)
# print("摘要")
# print("=" * 60)
# print(f"模型      : {MODEL_NAME}  (原始 Google BERT，無任何微調)")
# print(f"推論方法  : MLM Prompting — [MASK] 位置正/負面詞機率加總比較")
# print(f"Prompt    : '<text> Overall, it was [MASK].'")
# print(f"正面詞數  : {len(pos_ids)}  負面詞數: {len(neg_ids)}")
# print(f"Val Acc   : {accuracy:.4f}")
# print(f"Val F1    : {f1:.4f}")
# print(f"Test 筆數 : {len(output_df)}")
