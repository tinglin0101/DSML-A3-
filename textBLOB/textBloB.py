import pandas as pd
from textblob import TextBlob

# 1. 讀取原始資料集
# # 請確保 'train_2022.csv' 與此程式碼在同一個資料夾下
df = pd.read_csv('dataset/train_2022.csv')

# # 準備一個列表來儲存分析後的所有結果
results = []


# # 2. 逐筆遍歷資料進行分析
for idx, row in df.iterrows():
#     # 將文本轉為小寫，方便關鍵字比對
    text = str(row['TEXT']).lower()
    label = row['LABEL']  # 1 代表正評, 0 代表負評
    
#     # 使用 TextBlob 計算情感極性分數 (範圍：-1.0 極負面 到 1.0 極正面)
    blob = TextBlob(text)
    comp = blob.sentiment.polarity
    
    cat = ""
        
#     # 檢查是否為「中立」 (情感分數極度接近 0)
    if -0.1 <= comp <= 0.1:
        cat = 'central' #中立
        
#     # 檢查是否為「諷刺或混合」
    elif (label == 1 and comp <= -0.2) or (label == 0 and comp >= 0.2):
        cat = 'mixed' #諷刺或混合 
        # 備註 0:1 = 274(85.8%):45(14.2%)

    elif label == 1:
        cat = 'positive' #稱讚
        
    else:
        cat = 'negative' #批評
        
#     # 將這筆資料的分析結果記錄下來
    results.append({
        'TEXT': text, 
        'LABEL': label, 
        'Polarity_Score': round(comp, 4), # 四捨五入到小數點後四位
        'Category': cat

    })

# # 3. 將結果轉換為 DataFrame
res_df = pd.DataFrame(results)

# # 印出這 5 個類別的最終統計數量
# print("===== 各類別數量統計 =====")
print(res_df['Category'].value_counts())

# # 4. 匯出檔案 (可選)
# # 匯出所有結果
# res_df.to_csv('textBLOB_v1.csv', index=False)
# print("\n已成功匯出完整分析結果至 'textBLOB_v1.csv'")

# # 如果你想單獨匯出「中立」的資料，可以這樣寫：
neutral_df = res_df[res_df['Category'] == 'central']
neutral_df.to_csv('neutral_reviews.csv', index=False)
# mixed_df = res_df[res_df['Category'] == 'mixed']
# mixed_df.to_csv('mixed_reviews.csv', index=False)