import stanza
import pandas as pd
import json
import argparse

# ──────────────────────────────────────────────
# 簡易情緒詞典 (可依需求擴充)
# ──────────────────────────────────────────────
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "outstanding",
    "superb", "perfect", "brilliant", "nice", "beautiful", "lovely", "delightful",
    "pleased", "happy", "satisfied", "impressive", "reliable", "durable", "affordable",
    "comfortable", "easy", "fast", "quick", "efficient", "effective", "helpful",
    "friendly", "clean", "fresh", "safe", "secure", "convenient", "smooth", "light",
    "quality", "best", "better", "worth", "valuable", "cheap", "reasonable",
    "sturdy", "solid", "strong", "powerful", "clear", "sharp", "bright", "fun",
    "entertaining", "engaging", "interesting", "recommend", "loved", "enjoy",
    "charming", "quirky", "witty", "compelling", "moving", "touching", "heartfelt",
    "original", "creative", "innovative", "refreshing", "inspiring", "thoughtful",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "poor", "worst", "disappointing",
    "disappointed", "useless", "broken", "defective", "expensive", "overpriced",
    "fragile", "weak", "slow", "difficult", "hard", "complicated", "confusing",
    "uncomfortable", "unsafe", "unreliable", "dirty", "rough", "heavy", "bulky",
    "hate", "dislike", "avoid", "waste", "regret", "ugly", "loud", "noisy",
    "flimsy", "faulty", "damaged", "boring", "dull", "tedious", "predictable",
    "cliché", "mediocre", "shallow", "pointless", "ridiculous", "absurd",
    "annoying", "irritating", "frustrating", "pretentious", "overlong", "messy",
    "incoherent", "convoluted", "derivative", "formulaic", "forgettable",
}


def classify_sentiment(word: str) -> str:
    """
    根據詞典判斷情緒詞的極性。
    回傳 'positive'、'negative' 或 'neutral'。
    """
    w = word.lower()
    if w in POSITIVE_WORDS:
        return "positive"
    if w in NEGATIVE_WORDS:
        return "negative"
    return "neutral"


def extract_aspect_opinion_pairs(text: str, nlp) -> dict:
    """
    利用 Stanza 依存句法分析，從文本中抽取
    情緒目標（Aspect）與情緒詞（Opinion）配對。

    支援兩種常見句型：
    1. amod 關係：形容詞直接修飾名詞
       e.g. "great price"  → aspect=price, opinion=great
    2. 謂語形容詞 (cop + nsubj)：主語名詞 + be + 形容詞
       e.g. "the price is reasonable" → aspect=price, opinion=reasonable

    回傳格式：{"aspect": "positive/negative/neutral", ...}
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    doc = nlp(text)
    pairs = {}

    for sentence in doc.sentences:
        words = sentence.words  # 1-based id list

        # 建立 id → word 的快速查詢表 (1-based)
        id_to_word = {w.id: w for w in words}

        for word in words:
            upos = getattr(word, "upos", "")
            deprel = getattr(word, "deprel", "")
            head_id = word.head  # 0 = ROOT

            # ── 情況 1：amod（形容詞直接修飾名詞）────────────────
            # 依存關係：形容詞 --amod--> 名詞
            if upos == "ADJ" and deprel == "amod":
                noun = id_to_word.get(head_id)
                if noun and getattr(noun, "upos", "") in ("NOUN", "PROPN"):
                    aspect = noun.lemma.lower()
                    opinion = word.lemma.lower()
                    sentiment = classify_sentiment(opinion)
                    if sentiment != "neutral":
                        pairs[aspect] = sentiment

            # ── 情況 2：謂語形容詞（主語名詞 + be + 形容詞）────────
            # 依存關係：形容詞 --acomp/xcomp/adj--> 動詞
            #           名詞   --nsubj-----------> 動詞 (同一個 head)
            # 或更常見：形容詞作為 ROOT，名詞透過 nsubj 指向形容詞
            if upos == "ADJ" and deprel in ("ROOT", "acomp", "xcomp", "advcl", "ccomp"):
                adj_id = word.id
                adj_lemma = word.lemma.lower()
                # 找同一句中所有 nsubj 指向此形容詞的詞
                for other in words:
                    if other.head == adj_id and getattr(other, "deprel", "") in ("nsubj", "nsubj:pass"):
                        if getattr(other, "upos", "") in ("NOUN", "PROPN"):
                            aspect = other.lemma.lower()
                            sentiment = classify_sentiment(adj_lemma)
                            if sentiment != "neutral":
                                pairs[aspect] = sentiment

    return pairs


def main():
    parser = argparse.ArgumentParser(description="使用 Stanza 抽取文本中的 Aspect-Opinion 配對")
    parser.add_argument('--input',  type=str, default='dataset_split/train_2022_split_stanza_v1.csv',
                        help='輸入的 CSV 檔案（需含 TEXT 欄位）')
    parser.add_argument('--output', type=str, default='dataset_split/train_2022_aspect_opinion.csv',
                        help='輸出的 CSV 檔案')
    parser.add_argument('--text_col', type=str, default='TEXT',
                        help='要處理的文字欄位名稱（預設 TEXT）')
    args = parser.parse_args()

    print("正在初始化 Stanza Pipeline（tokenize, pos, lemma, depparse）...")
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', verbose=False)

    print(f"正在讀取資料：{args.input} ...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"無法讀取檔案 {args.input}: {e}")
        return

    if args.text_col not in df.columns:
        print(f"找不到欄位 '{args.text_col}'，可用欄位：{list(df.columns)}")
        return

    print("正在抽取 Aspect-Opinion 配對...")
    df['ASPECT_OPINION'] = df[args.text_col].apply(
        lambda x: json.dumps(extract_aspect_opinion_pairs(x, nlp), ensure_ascii=False)
    )

    print(f"正在將結果儲存至：{args.output} ...")
    df.to_csv(args.output, index=False)
    print("完成！")

    # 印出前幾筆範例
    print("\n─── 前 5 筆範例 ───")
    for _, row in df.head(5).iterrows():
        print(f"  TEXT  : {str(row[args.text_col])[:80]}")
        print(f"  結果  : {row['ASPECT_OPINION']}")
        print()


# ──────────────────────────────────────────────
# 單句互動測試（python extract_aspect_opinion_stanza.py --mode interactive）
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "interactive":
        print("正在初始化 Stanza Pipeline...")
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', verbose=False)
        print("進入互動測試模式，輸入 q 離開。\n")
        while True:
            text = input("請輸入句子: ").strip()
            if text.lower() in ("q", "quit", "exit"):
                break
            if not text:
                continue
            result = extract_aspect_opinion_pairs(text, nlp)
            print(f"  → {json.dumps(result, ensure_ascii=False)}\n")
    else:
        main()
