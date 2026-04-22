import pandas as pd

df = pd.read_csv("train_2022.csv")

df["word_count"] = df["TEXT"].astype(str).apply(lambda t: len(t.split()))

top10 = df.nlargest(10, "word_count")[["row_id", "word_count", "TEXT"]]

print("=== Top 10 最多 words 的資料 ===")
for _, row in top10.iterrows():
    print(f"\nrow_id={row['row_id']}  words={row['word_count']}")
    print(f"  {row['TEXT'][:200]}{'...' if len(row['TEXT']) > 200 else ''}")

print(f"\n=== 統計 ===")
print(df["word_count"].describe().round(1))