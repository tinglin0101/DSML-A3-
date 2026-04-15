"""
NRC Emotion Analysis on SPLIT_TEXT column
=========================================
針對 train_2022_split.csv 的 SPLIT_TEXT 欄位做情緒分析：
  1. 將 SPLIT_TEXT list 展開成獨立子列
  2. 新增 sub_id（從 0 開始，代表第幾段）
  3. 每列包含：row_id, sub_id, split_text, LABEL
  4. 使用 NRC Word-Emotion Lexicon 計算 8 種情緒的 raw count 與 frequency
     - 憤怒（Anger）、厭惡（Disgust）、恐懼（Fear）、喜悅（Joy）
     - 悲傷（Sadness）、驚訝（Surprise）、信任（Trust）、期待（Anticipation）

輸出：nrc_emotion_result.csv
"""

import ast
import pandas as pd
from nrclex import NRCLex

# ── 設定 ────────────────────────────────────────────────────────────────────────
INPUT_FILE  = "train_2022_split.csv"
OUTPUT_FILE = "nrc_emotion_result.csv"

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "trust", "anticipation"]

# 中文欄位對照（raw count）
EMOTION_ZH_COUNT = {
    "anger":        "Anger_count",
    "disgust":      "Disgust_count",
    "fear":         "Fear_count",
    "joy":          "Joy_count",
    "sadness":      "Sadness_count",
    "surprise":     "Surprise_count",
    "trust":        "Trust_count",
    "anticipation": "Anticipation_count",
}

# 中文欄位對照（frequency）
EMOTION_ZH_FREQ = {
    "anger":        "Anger_freq",
    "disgust":      "Disgust_freq",
    "fear":         "Fear_freq",
    "joy":          "Joy_freq",
    "sadness":      "Sadness_freq",
    "surprise":     "Surprise_freq",
    "trust":        "Trust_freq",
    "anticipation": "Anticipation_freq",
}


def get_emotion_scores(nrc_instance: NRCLex, text: str) -> dict:
    """
    用同一個 NRCLex 實例載入文字，回傳 8 種情緒的 raw count 與 frequency。
    """
    nrc_instance.load_raw_text(text)

    raw   = nrc_instance.raw_emotion_scores    # dict, 僅含有值的情緒
    freq  = nrc_instance.affect_frequencies    # dict, 所有情緒皆有（含 0.0）

    result = {}
    for emo in EMOTIONS:
        # result[EMOTION_ZH_COUNT[emo]] = raw.get(emo, 0)
        result[EMOTION_ZH_FREQ[emo]]  = round(freq.get(emo, 0.0), 6)

    return result


def main():
    print(f"讀取 {INPUT_FILE} …")
    df = pd.read_csv(INPUT_FILE)

    # 將 SPLIT_TEXT 字串轉回 Python list
    df["SPLIT_TEXT"] = df["SPLIT_TEXT"].apply(ast.literal_eval)

    # 建立 NRCLex 實例（載入一次字典，供所有文字重複使用）
    print("載入 NRC 情緒字典 …")
    nrc = NRCLex()

    records = []
    total = len(df)

    print(f"開始分析 {total} 列 …")
    for idx, row in df.iterrows():
        if idx % 200 == 0:
            print(f"  進度：{idx}/{total}")

        split_texts = row["SPLIT_TEXT"]
        for sub_id, segment in enumerate(split_texts):
            emotion_scores = get_emotion_scores(nrc, segment)

            record = {
                "row_id":     row["row_id"],
                "sub_id":     sub_id,
                "split_text": segment,
                "LABEL":      row["LABEL"],
                "NUM_SPLITS": row["NUM_SPLITS"],
            }
            record.update(emotion_scores)
            records.append(record)

    result_df = pd.DataFrame(records)

    # 欄位排序
    fixed_cols   = ["row_id", "sub_id", "split_text", "LABEL", "NUM_SPLITS"]
    # count_cols   = [EMOTION_ZH_COUNT[e] for e in EMOTIONS]
    count_cols   = []  # 不輸出 raw count 欄位
    freq_cols    = [EMOTION_ZH_FREQ[e]  for e in EMOTIONS]
    result_df    = result_df[fixed_cols + count_cols + freq_cols]

    # 輸出
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n完成！共 {len(result_df)} 列，已儲存至 {OUTPUT_FILE}")
    print("\n前 5 列預覽：")
    print(result_df.head().to_string())

    # 簡單統計
    print("\n─── 各情緒 raw count 統計（前 5 列）───")
    print(result_df[count_cols].describe().round(3).to_string())


if __name__ == "__main__":
    main()
