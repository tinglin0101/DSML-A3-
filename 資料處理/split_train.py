# 切分方式說明：

# 使用 sklearn.model_selection.train_test_split 搭配 stratified sampling（分層抽樣）：

# test_size=0.2：20% 作為 validation
# stratify=df["LABEL"]：依照 LABEL 欄位分層，確保 train 和 val 中正負樣本比例相同（各 50%），避免隨機切分造成標籤分佈不均的問題
# random_state=42：固定隨機種子，保證每次執行結果一致、可重現
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train_2022_with_category.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["LABEL"]
)

train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)

print(f"Total samples : {len(df)}")
print(f"Train samples : {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val   samples : {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"\nLabel distribution in train:\n{train_df['LABEL'].value_counts(normalize=True)}")
print(f"\nLabel distribution in val:\n{val_df['LABEL'].value_counts(normalize=True)}")
