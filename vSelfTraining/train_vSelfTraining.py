import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# =============================================================
# vSelfTraining — 半監督式自訓練 (Self-Training)
# 基於 V1 (TF-IDF + Logistic Regression)，
# 利用 test set 中高信心度的預測結果作為偽標籤，
# 迭代擴充訓練集以提升模型表現。
# =============================================================

# 1. 載入資料
train_df = pd.read_csv("train_2022.csv")
test_df = pd.read_csv("test_no_answer_2022.csv")

print(f"原始訓練集大小: {len(train_df)}")
print(f"測試集大小: {len(test_df)}")
print(f"標籤分布:\n{train_df['LABEL'].value_counts()}\n")

# 2. 合併所有文本建立 TF-IDF（確保特徵空間一致）
all_texts = pd.concat([train_df["TEXT"], test_df["TEXT"]], ignore_index=True)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
tfidf.fit(all_texts)  # 在所有文本上 fit，利用未標記資料改善特徵表示

X_labeled = tfidf.transform(train_df["TEXT"])
X_unlabeled_all = tfidf.transform(test_df["TEXT"])
y_labeled = train_df["LABEL"].values

# 3. 基線模型（與 V1 對照）
base_model = LogisticRegression(max_iter=1000, C=1.0)
base_scores = cross_val_score(base_model, X_labeled, y_labeled, cv=5, scoring="accuracy")
print(f"[基線] 5-fold CV accuracy: {base_scores.mean():.4f} (+/- {base_scores.std():.4f})")

# 4. Self-Training 迭代
# ── 使用遞減閾值策略 ──
# 初始閾值較高，每輪遞減，讓模型逐步納入偽標籤
INITIAL_THRESHOLD = 0.85     # 初始信心度閾值
MIN_THRESHOLD = 0.75         # 最低信心度閾值
THRESHOLD_DECAY = 0.02       # 每輪遞減量
MAX_ITERATIONS = 15          # 最大迭代次數
BATCH_SIZE = 500             # 每輪最多新增的偽標籤數量（取最高信心度的前 N 個）

# 追蹤用
unlabeled_mask = np.ones(len(test_df), dtype=bool)  # True = 尚未被標記
pseudo_texts = []       # 偽標籤文本
pseudo_labels = []      # 偽標籤

iteration_log = []
threshold = INITIAL_THRESHOLD

for iteration in range(1, MAX_ITERATIONS + 1):
    # 合併已標記資料 + 偽標籤資料
    if len(pseudo_texts) > 0:
        combined_texts = pd.concat([train_df["TEXT"], pd.Series(pseudo_texts)], ignore_index=True)
        combined_labels = np.concatenate([y_labeled, np.array(pseudo_labels)])
    else:
        combined_texts = train_df["TEXT"]
        combined_labels = y_labeled

    X_combined = tfidf.transform(combined_texts)

    # 訓練模型
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_combined, combined_labels)

    # 對剩餘未標記資料預測
    remaining_count = unlabeled_mask.sum()
    if remaining_count == 0:
        print(f"迭代 {iteration}: 所有未標記資料已被納入，提前結束。")
        break

    remaining_indices = np.where(unlabeled_mask)[0]
    X_remaining = tfidf.transform(test_df["TEXT"].iloc[remaining_indices])
    proba = model.predict_proba(X_remaining)
    max_proba = proba.max(axis=1)
    pred_labels = model.classes_[proba.argmax(axis=1)]

    # 篩選高信心度樣本（超過閾值，且限制每輪最多新增 BATCH_SIZE 個）
    high_conf_mask = max_proba >= threshold

    if high_conf_mask.sum() == 0:
        # 嘗試降低閾值
        if threshold > MIN_THRESHOLD:
            threshold = max(threshold - THRESHOLD_DECAY, MIN_THRESHOLD)
            print(f"迭代 {iteration}: 無高信心度樣本，降低閾值至 {threshold:.2f}，重試...")
            continue
        else:
            print(f"迭代 {iteration}: 閾值已降至最低 ({MIN_THRESHOLD})，仍無高信心度樣本，停止。")
            break

    # 取信心度最高的前 BATCH_SIZE 個
    if high_conf_mask.sum() > BATCH_SIZE:
        # 只取 top-k
        conf_values = max_proba.copy()
        conf_values[~high_conf_mask] = 0  # 排除低於閾值的
        top_k_idx = np.argsort(conf_values)[-BATCH_SIZE:]
        selected_mask = np.zeros(len(max_proba), dtype=bool)
        selected_mask[top_k_idx] = True
    else:
        selected_mask = high_conf_mask

    n_new = selected_mask.sum()

    # 將選中的樣本加入偽標籤
    selected_global_indices = remaining_indices[selected_mask]
    new_texts = test_df["TEXT"].iloc[selected_global_indices].tolist()
    new_labels = pred_labels[selected_mask].tolist()

    pseudo_texts.extend(new_texts)
    pseudo_labels.extend(new_labels)

    # 標記為已處理
    unlabeled_mask[selected_global_indices] = False

    # 記錄
    total_train = len(train_df) + len(pseudo_texts)
    remaining = unlabeled_mask.sum()
    avg_conf = float(max_proba[selected_mask].mean())
    min_conf = float(max_proba[selected_mask].min())

    # 偽標籤的類別分布
    new_label_counts = pd.Series(new_labels).value_counts().to_dict()

    log_entry = {
        "iteration": iteration,
        "threshold": threshold,
        "new_pseudo_labels": int(n_new),
        "total_pseudo_labels": len(pseudo_texts),
        "total_training_size": total_train,
        "remaining_unlabeled": int(remaining),
        "avg_confidence": avg_conf,
        "min_confidence": min_conf,
        "label_dist": new_label_counts,
    }
    iteration_log.append(log_entry)

    print(f"迭代 {iteration} (閾值={threshold:.2f}): "
          f"新增 {n_new} 筆偽標籤 "
          f"(信心度: avg={avg_conf:.4f}, min={min_conf:.4f}), "
          f"類別分布={new_label_counts}, "
          f"訓練集總計: {total_train}, 剩餘: {remaining}")

    # 迭代後遞減閾值
    threshold = max(threshold - THRESHOLD_DECAY, MIN_THRESHOLD)

# 5. 最終模型訓練
print(f"\n{'='*50}")
print(f"最終模型")
print(f"{'='*50}")
print(f"偽標籤總數: {len(pseudo_texts)}")
print(f"最終訓練集大小: {len(train_df) + len(pseudo_texts)}")

# 合併最終訓練資料
if len(pseudo_texts) > 0:
    final_texts = pd.concat([train_df["TEXT"], pd.Series(pseudo_texts)], ignore_index=True)
    final_labels = np.concatenate([y_labeled, np.array(pseudo_labels)])
    pseudo_label_dist = pd.Series(pseudo_labels).value_counts()
    print(f"偽標籤分布:\n{pseudo_label_dist}")
else:
    final_texts = train_df["TEXT"]
    final_labels = y_labeled

X_final = tfidf.transform(final_texts)

final_model = LogisticRegression(max_iter=1000, C=1.0)

# 在原始標記資料上做 CV（評估最終模型架構的穩定性）
final_cv_scores = cross_val_score(final_model, X_labeled, y_labeled, cv=5, scoring="accuracy")
print(f"\n[原始標記資料] 5-fold CV accuracy: {final_cv_scores.mean():.4f} (+/- {final_cv_scores.std():.4f})")

# 在全部資料上訓練最終模型
final_model.fit(X_final, final_labels)

# 6. 對整個測試集做最終預測
predictions = final_model.predict(X_unlabeled_all)

# 7. 輸出結果
output_df = pd.DataFrame({
    "row_id": test_df["row_id"],
    "label": predictions
})
output_df.to_csv("vSelfTraining.csv", index=False)

print(f"\n預測結果已儲存至 vSelfTraining.csv ({len(output_df)} 筆)")
print(f"預測分布:\n{pd.Series(predictions).value_counts()}")

# 8. 印出迭代摘要
print(f"\n{'='*50}")
print("Self-Training 迭代摘要")
print(f"{'='*50}")
print(f"{'迭代':>4} | {'閾值':>6} | {'新增':>6} | {'累計':>6} | {'訓練集':>8} | {'平均信心':>8} | {'最低信心':>8}")
print("-" * 70)
for entry in iteration_log:
    print(f"{entry['iteration']:>4} | {entry['threshold']:>6.2f} | "
          f"{entry['new_pseudo_labels']:>6} | {entry['total_pseudo_labels']:>6} | "
          f"{entry['total_training_size']:>8} | {entry['avg_confidence']:>8.4f} | "
          f"{entry['min_confidence']:>8.4f}")
