import pandas as pd

def analyze_text_length(file_path):
    print(f"--- 檔案: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
        if 'TEXT' not in df.columns:
            print(f"錯誤: 在 {file_path} 中找不到 'TEXT' 欄位")
            return
        
        # 計算長度
        lengths = df['TEXT'].astype(str).apply(len)
        
        print(f"資料筆數: {len(df)}")
        print(f"最長長度: {lengths.max()}")
        print(f"最短長度: {lengths.min()}")
        print(f"平均長度: {lengths.mean():.2f}")
        print(f"中位數長度: {lengths.median()}")
        print()
    except Exception as e:
        print(f"處理 {file_path} 時發生錯誤: {e}")

if __name__ == "__main__":
    files = ["train_2022.csv", "test_no_answer_2022.csv"]
    for f in files:
        analyze_text_length(f)
