import stanza
import pandas as pd
import argparse

def get_stanza_pipeline():
    # 下載英文模型（如果尚未下載）
    stanza.download('en')
    # 我們需要 tokenize (分詞) 和 lemma (詞形還原)，也可以選擇加入 pos。
    return stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')

def split_text_by_transition_v1(text, nlp):
    transitions = { "but", "however", "although", "though", "yet", "nevertheless", 
        "nonetheless", "except", "while", "whereas"}
    if not isinstance(text, str):
        return [text]
        
    doc = nlp(text)
    split_indices = []
    
    for sentence in doc.sentences:
        for word in sentence.words:
            # 檢查單字或其原型是否在我們的轉折詞列表中
            word_lower = word.text.lower()
            lemma_lower = word.lemma.lower() if word.lemma else ""
            
            if word_lower in transitions or lemma_lower in transitions:
                # 加上詞性 (POS) 判斷，確保該詞真的是作為轉折連接詞或副詞使用
                # CCONJ = 對等連接詞, SCONJ = 從屬連接詞, ADV = 副詞
                # 這樣可以避免把介係詞 (例如 except this) 或名詞 (例如 for a while) 誤當成轉折切開
                if getattr(word, "upos", "") in ("CCONJ", "SCONJ", "ADV"):
                    # 確保我們不從第一個字開始切割（如果句首就是轉折詞的話）
                    if word.start_char > 0:
                        split_indices.append(word.start_char)
                    
    # 根據取得的索引來切割原始字串
    parts = []
    last_idx = 0
    for idx in split_indices:
        segment = text[last_idx:idx].strip()
        if segment:
            parts.append(segment)
        last_idx = idx
        
    # 加入最後一段
    final_segment = text[last_idx:].strip()
    if final_segment:
        parts.append(final_segment)
    
    # 如果都沒有擷取到，回傳原本的 text
    if not parts:
        parts = [text]
        
    return parts

def split_text_by_transition_v2(text, nlp):
    # 這裡的轉折詞不包含 and，因為 and 需要特殊處理
    transitions = { "but", "however", "although", "though", "yet", "nevertheless", 
                    "nonetheless", "except", "while", "whereas"}
    
    if not isinstance(text, str):
        return [text]
        
    doc = nlp(text)
    split_indices = []
    
    for sentence in doc.sentences:
        words = sentence.words  # 取得該句所有的單字列表
        
        for word in words:
            word_lower = word.text.lower()
            lemma_lower = word.lemma.lower() if word.lemma else ""
            start_char = getattr(word, "start_char", 0)
            
            # --- 情況 1：處理一般的轉折詞 ---
            if word_lower in transitions or lemma_lower in transitions:
                if getattr(word, "upos", "") in ("CCONJ", "SCONJ", "ADV"):
                    if start_char > 0:
                        split_indices.append(start_char)
                continue # 已經判斷完畢，跳過後面的 and 判斷
            
            # --- 情況 2：特殊處理 "and" ---
            if word_lower == "and":
                # 取得 and 依附的 head_id (注意：Stanza 的 head_id 是 1-based，0 代表 ROOT)
                # head_id = getattr(word, "head", 0)
                head_id = word.head if word.head is not None else 0

                if head_id > 0:
                    # 透過 index 找到 and 所依附的目標詞 (轉換為 0-based index)
                    head_word = words[head_id - 1]
                    
                    # 判斷依附的目標詞詞性是否為動詞 (VERB) 或助動詞 (AUX)
                    # 如果是，代表 and 正在連接兩個動作/子句，此時才作為切割點
                    if getattr(head_word, "upos", "") in ("VERB", "AUX"):
                        if start_char > 0:
                            split_indices.append(start_char)
                            
    # 確保切割點由小到大排序 (有時句法分析順序不一定，排序較保險)
    split_indices = sorted(list(set(split_indices)))
    
    # 根據取得的索引來切割原始字串
    parts = []
    last_idx = 0
    for idx in split_indices:
        segment = text[last_idx:idx].strip()
        if segment:
            parts.append(segment)
        last_idx = idx
        
    # 加入最後一段
    final_segment = text[last_idx:].strip()
    if final_segment:
        parts.append(final_segment)
    
    # 如果都沒有擷取到，回傳原本的 text
    if not parts:
        parts = [text]
        
    return parts

def split_text_by_transition(text, nlp):
    # return split_text_by_transition_v1(text, nlp)
    return split_text_by_transition_v2(text, nlp)
    
    
def main():
    parser = argparse.ArgumentParser(description="使用 Stanza 根據轉折詞分割 CSV 中的 TEXT 欄位")
    parser.add_argument('--input', type=str, default='dataset_split/train_2022.csv', help='輸入的 CSV 檔案')
    parser.add_argument('--output', type=str, default='dataset_split/train_2022_split.csv', help='輸出的 CSV 檔案')
    parser.add_argument('--mode', type=str, choices=['csv', 'interactive'], default=None, help='執行模式：csv 或 interactive')
    args = parser.parse_args()

    print("正在初始化 Stanza Pipeline...")
    # nlp = get_stanza_pipeline()
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
    
    # 定義轉折詞/連接詞
    # transition_words = {
    #     "but", "however", "although", "though", "yet", "nevertheless", 
    #     "nonetheless", "except", "while", "whereas"
    # }

    # mode = args.mode
    # if not mode:
    #     print("\n=== 請選擇執行模式 ===")
    #     print("1. 處理整個 CSV 檔案 (預設)")
    #     print("2. 手動輸入句子測試")
    #     choice = input("請輸入 1 或 2 (直接按 Enter 執行 1): ").strip()
    #     mode = "interactive" if choice == "2" else "csv"

    # if mode == "interactive":
    #     print(f"\n--- 進入手動測試模式 ---")
    #     # print(f"目前使用的轉折詞: {transition_words}")
    #     print("請隨時輸入句子來測試切割結果 (輸入 'q', 'exit' 或 'quit' 離開)")
    #     while True:
    #         text = input("\n請輸入測試句子: ").strip()
    #         if text.lower() in ('q', 'exit', 'quit'):
    #             break
    #         if not text:
    #             continue
            
    #         parts = split_text_by_transition(text, nlp, transition_words)
    #         print(f"-> 切割成 {len(parts)} 段:")
    #         for i, part in enumerate(parts):
    #             print(f"   [{i+1}] {part}")
                
# else:
    print(f"\n正在讀取資料: {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"無法讀取檔案 {args.input}: {e}")
        return
        
    # print(f"將依據以下轉折詞對句子進行切割: {transition_words}")
    
    # 套用切割函數
    df['SPLIT_TEXT'] = df['TEXT'].apply(lambda x: split_text_by_transition(x, nlp))
    df['NUM_SPLITS'] = df['SPLIT_TEXT'].apply(len)
        
    print(f"正在將結果儲存至: {args.output}...")
    df.to_csv(args.output, index=False)
    print("完成！")

if __name__ == "__main__":
    main()
