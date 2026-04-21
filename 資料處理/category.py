import pandas as pd

# 1. 讀取資料集 (請確認檔名與路徑正確)
base_path = "D:\\[課程]DSML\\DSML-A3-\\"
# df = pd.read_csv(base_path + "train_2022.csv") 
df = pd.read_csv(base_path + "test_no_answer_2022.csv") 

# 2. 定義三大領域的專屬關鍵字字典 (全小寫)
movie_kw = ['movie', 'film', 'director', 'comedy', 'drama', 'scene', 'screen', 'cinema', 'actor', 'actress', 'watch', 'story']
game_kw = ['game', 'play', 'graphics', 'player', 'multiplayer', 'level', 'nintendo', 'xbox', 'playstation', 'pc']
product_kw = ['price', 'product', 'quality', 'buy', 'bought', 'use', 'device', 'dish', 'amazon', 'prime', 'item', 'money', 'work', 'battery', 'cable', 'cheap']

# 3. 建立分類函數
def categorize(text):
    # 處理空值，避免程式報錯
    if not isinstance(text, str):
        return 'Unknown'
        
    # 將文字轉為小寫，方便比對
    text_lower = text.lower()
    
    # 計算各個領域關鍵字在句子中出現的次數
    # 使用 .split() 可以避免把 "scenery" 誤認為包含 "scene"
    words = text_lower.split()
    counts = {
        'Movie': sum(1 for kw in movie_kw if kw in words or kw in text_lower),
        'Game': sum(1 for kw in game_kw if kw in words or kw in text_lower),
        'Product': sum(1 for kw in product_kw if kw in words or kw in text_lower)
    }
    
    # 找出最高分的領域
    max_cat = max(counts, key=counts.get)
    
    # 如果所有領域的分數都是 0，則歸類為 'Other'
    if counts[max_cat] == 0:
        return 'Other'
        
    return max_cat

# 4. 將分類函數應用到 TEXT 欄位，並建立新的 category 欄位
df['category'] = df['TEXT'].apply(categorize)
# df.to_csv(base_path + "train_2022_with_category.csv", index=False)  # 儲存帶有 category 欄位的新 CSV

# 5. 印出分析結果 (各類別的數量與百分比)
print("=== 領域數量統計 ===")
print(df['category'].value_counts())

print("\n=== 領域比例分佈 (%) ===")
print(df['category'].value_counts(normalize=True) * 100)

# (選擇性) 篩選出特定領域的資料，例如只看 'Product' 的前 5 筆
# print(df[df['category'] == 'Product'].head())