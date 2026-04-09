import pandas as pd
import numpy as np
import torch
import ssl
import nltk

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertForMaskedLM

train_df = pd.read_csv("train_2022.csv")   # 作為 validation set

MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model     = BertForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_sentiwordnet_score(word: str) -> float:
    """
    查詢一個詞的情感分數
    回傳 pos_score - neg_score（正數=正面，負數=負面）
    """
    synsets = list(swn.senti_synsets(word))
    if not synsets:
        return 0.0
    s = synsets[0]
    return s.pos_score() - s.neg_score()

print("建立情感分數向量...")
vocab_size = tokenizer.vocab_size
sentiment_vector = torch.zeros(vocab_size)

for token, idx in tokenizer.get_vocab().items():
    # BERT token 可能有 ## 前綴，先清掉
    clean_token = token.replace("##", "")
    score = get_sentiwordnet_score(clean_token)
    sentiment_vector[idx] = score

sentiment_vector = sentiment_vector.to(device)

# ------------------------------------------------------------------
# 4. MLM Prompting 推論函式
# ------------------------------------------------------------------
MASK_TOKEN = tokenizer.mask_token    # "[MASK]"
MAX_LENGTH = 512


def build_prompt(text: str) -> str:
    """建立包含 [MASK] 的 prompt。"""
    return f"{text.strip()} Overall, it was {MASK_TOKEN}."

def predict_single(text: str) -> float:
    prompt = build_prompt(text)
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_LENGTH
    ).to(device)

    input_ids = inputs["input_ids"][0]
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return 1

    mask_pos = mask_positions[-1].item()

    with torch.no_grad():
        logits = model(**inputs).logits[0, mask_pos, :]

    probs = torch.softmax(logits, dim=-1)  # 每個詞的機率

    # ✅ 核心：機率 × 情感分數 加權加總
    sentiment_score = (probs * sentiment_vector).sum().item()
    # print(sentiment_score)
    return sentiment_score
    # return 1 if sentiment_score >= 0 else 0

def predict_labels(texts, label=""):
    """對一系列文本逐筆執行 MLM prompting 情感推論。"""
    results    = []
    texts_list = texts.tolist() if hasattr(texts, "tolist") else list(texts)
    total      = len(texts_list)
    

    for i, text in enumerate(texts_list):
        # print(i,":",end=' ')
        results.append(predict_single(text))
        # if (i + 1) % 200 == 0 or i + 1 == total:
        #     print(f"  [{label}] 進度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

    # print(results)
    return np.array(results)


# ------------------------------------------------------------------
# 5. Validation — 在 train_2022.csv 上評估準確率
# ------------------------------------------------------------------
# print("=" * 60)
# print("Validation (train_2022.csv)")
# print("=" * 60)

# predict_labels(train_df["TEXT"], label="Validation")

# val_preds = predict_labels(train_df["TEXT"], label="Validation")
# val_true  = train_df["LABEL"].values

# accuracy  = (val_preds == val_true).mean()
# print(f"\nValidation Accuracy: {accuracy:.4f}")

# 混淆矩陣
# tp = int(((val_preds == 1) & (val_true == 1)).sum())
# tn = int(((val_preds == 0) & (val_true == 0)).sum())
# fp = int(((val_preds == 1) & (val_true == 0)).sum())
# fn = int(((val_preds == 0) & (val_true == 1)).sum())

# precision = tp / (tp + fp) if (tp + fp) > 0 else 0
# recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
# f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# print(f"\n混淆矩陣:")
# print(f"  TP={tp}  FP={fp}")
# print(f"  FN={fn}  TN={tn}")
# print(f"\nPrecision : {precision:.4f}")
# print(f"Recall    : {recall:.4f}")
# print(f"F1-Score  : {f1:.4f}")
# print(f"\nValidation 預測分布: {pd.Series(val_preds).value_counts().to_dict()}")

# ------------------------------------------------------------------
# 6. 對 test_no_answer_2022.csv 做最終預測
# ------------------------------------------------------------------
# print("\n" + "=" * 60)
# print("推論 test_no_answer_2022.csv")
# print("=" * 60)

test_preds = predict_labels(train_df["TEXT"], label="Test")

# # ------------------------------------------------------------------
# # 7. 儲存結果
# # ------------------------------------------------------------------
output_df = pd.DataFrame({
    "label": train_df["LABEL"],
    "predict" : test_preds,
})
output_df.to_csv("eMotion_vBERT.csv", index=False)

print(f"\n預測結果已儲存至 eMotion_vBERT.csv ({len(output_df)} 筆)")
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
