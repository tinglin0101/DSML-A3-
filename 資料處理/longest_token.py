import pandas as pd
from transformers import AutoTokenizer

df = pd.read_csv("../train_2022.csv")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

df["token_len"] = df["TEXT"].astype(str).apply(
    lambda t: len(tokenizer.encode(t, add_special_tokens=True))
)

print(df["token_len"].describe())
print(f"\n最長: {df['token_len'].max()} tokens，第 {df['token_len'].idxmax()} 行")
print(f"超過 128 tokens 的筆數: {(df['token_len'] > 128).sum()}")
# 第二個版本更實用，直接告訴你有幾筆資料會被 max_length=128 截斷。