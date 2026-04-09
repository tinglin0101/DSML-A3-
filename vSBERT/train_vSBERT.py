import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ============================================================
# vSBERT — Sentence-BERT Embedding + Logistic Regression
# ============================================================
# 與 V1 (TF-IDF) 的差異：
#   - 使用 Sentence-BERT (all-MiniLM-L6-v2) 生成句向量
#   - 能捕捉整句語意與語氣走向，不再受限於詞袋假設
#   - 不需 GPU，CPU 即可執行
# ============================================================

# 1. 載入資料
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

train_df = pd.read_csv(os.path.join(PROJECT_DIR, "train_2022.csv"))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, "test_no_answer_2022.csv"))

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"Label distribution:\n{train_df['LABEL'].value_counts()}")

# 2. 使用 Sentence-BERT 編碼文本
MODEL_NAME = "all-MiniLM-L6-v2"
print(f"\nLoading Sentence-BERT model: {MODEL_NAME} ...")
sbert = SentenceTransformer(MODEL_NAME)

print("Encoding training set ...")
X_train = sbert.encode(train_df["TEXT"].tolist(), batch_size=64, show_progress_bar=True)

print("Encoding test set ...")
X_test = sbert.encode(test_df["TEXT"].tolist(), batch_size=64, show_progress_bar=True)

y_train = train_df["LABEL"].values

print(f"Embedding dimension: {X_train.shape[1]}")

# 3. 訓練 Logistic Regression + 交叉驗證
model = LogisticRegression(C=1.0, max_iter=1000)

print("\nRunning 5-fold cross-validation ...")
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
print(f"Per-fold scores: {[f'{s:.4f}' for s in scores]}")

# 4. 全量訓練 + 預測
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 5. 儲存結果
output_df = pd.DataFrame({
    "row_id": test_df["row_id"],
    "label": predictions
})
output_path = os.path.join(SCRIPT_DIR, "vSBERT.csv")
output_df.to_csv(output_path, index=False)

print(f"\nPredictions saved to {output_path} ({len(output_df)} rows)")
print(f"Prediction distribution:\n{pd.Series(predictions).value_counts()}")
