import pandas as pd

df = pd.read_csv("train_2022.csv")

longest_word = ""
longest_row = None

for idx, text in df["TEXT"].items():
    if not isinstance(text, str):
        continue
    for word in text.split():
        clean = word.strip(".,!?\"'()[]")
        if len(clean) > len(longest_word):
            longest_word = clean
            longest_row = idx

print(f"最長單字: '{longest_word}'")
print(f"長度: {len(longest_word)} characters")
print(f"來自第 {longest_row} 行")