import stanza
import pandas as pd
import argparse

def get_stanza_pipeline():
    # 下載英文模型（如果尚未下載）
    stanza.download('en')
    # 我們需要 tokenize (分詞) 和 lemma (詞形還原)，也可以選擇加入 pos。
    return stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')

# def split_text_by_transition_v1(text, nlp):
#     transitions = { "but", "however", "although", "though", "yet", "nevertheless", 
#         "nonetheless", "except", "while", "whereas"}
#     if not isinstance(text, str):
#         return [text]
        
#     doc = nlp(text)
#     split_indices = []
    
#     for sentence in doc.sentences:
#         for word in sentence.words:
#             # 檢查單字或其原型是否在我們的轉折詞列表中
#             word_lower = word.text.lower()
#             lemma_lower = word.lemma.lower() if word.lemma else ""
            
#             if word_lower in transitions or lemma_lower in transitions:
#                 # 加上詞性 (POS) 判斷，確保該詞真的是作為轉折連接詞或副詞使用
#                 # CCONJ = 對等連接詞, SCONJ = 從屬連接詞, ADV = 副詞
#                 # 這樣可以避免把介係詞 (例如 except this) 或名詞 (例如 for a while) 誤當成轉折切開
#                 if getattr(word, "upos", "") in ("CCONJ", "SCONJ", "ADV"):
#                     # 確保我們不從第一個字開始切割（如果句首就是轉折詞的話）
#                     if word.start_char > 0:
#                         split_indices.append(word.start_char)
                    
#     # 根據取得的索引來切割原始字串
#     parts = []
#     last_idx = 0
#     for idx in split_indices:
#         segment = text[last_idx:idx].strip()
#         if segment:
#             parts.append(segment)
#         last_idx = idx
        
#     # 加入最後一段
#     final_segment = text[last_idx:].strip()
#     if final_segment:
#         parts.append(final_segment)
    
#     # 如果都沒有擷取到，回傳原本的 text
#     if not parts:
#         parts = [text]
        
#     return parts

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
                            print("and: ",text)
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


# ──────────────────────────────────────────────────────────────────
# v3：加入 Entity Masking，對命名實體（NER）進行保護後再切割
# ──────────────────────────────────────────────────────────────────

def mask_entities(text: str, doc) -> tuple[str, dict]:
    """
    將 Stanza NER 偵測到的命名實體替換為佔位符 __ENT_0__、__ENT_1__...
    避免實體名稱中的轉折詞（例如 "Yet Another Company"）被誤切割。

    回傳：
        masked_text : 替換後的字串
        mapping     : { "__ENT_0__": "原始實體文字", ... }
    """
    # 收集所有 entity span（start_char, end_char, text），由長到短排序
    # 避免短實體覆蓋長實體
    entities = []
    for sent in doc.sentences:
        for ent in sent.ents:
            entities.append((ent.start_char, ent.end_char, ent.text))

    # 依 start_char 由後往前替換，保持字串偏移量正確
    entities_sorted = sorted(entities, key=lambda x: x[0], reverse=True)

    mapping = {}
    masked = text
    counter = [0]  # 用 list 方便閉包修改

    for start, end, ent_text in entities_sorted:
        placeholder = f"__ENT_{counter[0]}__"
        counter[0] += 1
        mapping[placeholder] = ent_text
        masked = masked[:start] + placeholder + masked[end:]

    return masked, mapping


def unmask_entities(parts: list[str], mapping: dict) -> list[str]:
    """
    將切割後的各段落中的佔位符還原為原始實體文字。
    """
    restored = []
    for part in parts:
        for placeholder, original in mapping.items():
            part = part.replace(placeholder, original)
        restored.append(part)
    return restored


def split_text_by_transition_v3(text: str, nlp) -> list[str]:
    """
    v3 三步驟流程：
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 1 │ NER Masking   — 遮罩命名實體，防止實體內的關鍵詞被誤切  │
    │ Step 2 │ Syntax Parse  — 對遮罩後文字進行 POS + Dep Parsing    │
    │ Step 3 │ Smart Split   — 依規則決定是否切割（見下方說明）         │
    └─────────────────────────────────────────────────────────────┘

    切割規則（Step 3）：
    • but / however / although / though / yet / nevertheless /
      nonetheless / except / while / whereas
        → 直接切割（強烈轉折，幾乎必然引導對立子句）
    • and
        → 檢查 deprel 與 head 詞性：
          ‣ deprel == "cc" 且 head.upos in (VERB, AUX)  → 切割（連接動詞/子句）
          ‣ deprel == "cc" 且 head.upos in (NOUN, PROPN, ADJ, ADV, NUM) → 不切割（連接名詞短語）
          ‣ 其他情況 → 不切割（保守策略）
    """
    if not isinstance(text, str):
        return [text]

    # ── Step 1：解析原始文字（NER 用來 Masking）────────────────────
    doc = nlp(text)
    masked_text, mapping = mask_entities(text, doc)

    # if mapping:
    #     print(f"[NER Masking] 遮罩 {len(mapping)} 個實體：{list(mapping.values())}")

    # ── Step 2：對遮罩後的文字重新進行句法分析（POS + Dep Parse）────
    # 遮罩改變了文字結構，需重新解析；若無實體則複用原 doc
    analysis_doc = nlp(masked_text) if mapping else doc
    analysis_text = masked_text if mapping else text

    # ── Step 3：智慧切割────────────────────────────────────────────
    parts = _smart_split(analysis_text, analysis_doc)

    # ── Step 4：Unmask，還原命名實體────────────────────────────────
    parts = unmask_entities(parts, mapping)

    return parts


def _smart_split(text: str, doc) -> list[str]:
    """
    智慧切割核心：
    - 強轉折詞（but, however…）→ 直接切
    - and → 三層檢查決定是否切割：
        1. and.deprel 必須為 cc
        2. B（and 的 head）的詞性
        3. 若 B 為動詞但 B.deprel == conj，往上看 A（B 的 head）的詞性；
           A 為名詞代表 B 是被誤判的名詞並列，不切
    """
    STRONG_TRANSITIONS = {
        "but", "however", "although", "though", "yet",
        "nevertheless", "nonetheless", "except", "while", "whereas"
    }

    VERBAL_UPOS  = {"VERB", "AUX"}
    NOMINAL_UPOS = {"NOUN", "PROPN", "ADJ", "ADV", "NUM"}

    split_indices = []

    for sentence in doc.sentences:
        words = sentence.words
        id_to_word = {w.id: w for w in words}  # 1-based 查詢表

        for word in words:
            word_lower = word.text.lower()
            lemma_lower = word.lemma.lower() if word.lemma else ""
            start_char = getattr(word, "start_char", 0)
            upos   = getattr(word, "upos", "")
            deprel = getattr(word, "deprel", "")

            # ── 規則 A：強轉折詞，直接切 ──────────────────────────
            if word_lower in STRONG_TRANSITIONS or lemma_lower in STRONG_TRANSITIONS:
                if upos in ("CCONJ", "SCONJ", "ADV"):
                    if start_char > 0:
                        # print(f"[SPLIT] 強轉折詞 '{word.text}' @ char {start_char}")
                        split_indices.append(start_char)
                continue

            # ── 規則 B：and，三層檢查 ─────────────────────────────
            if word_lower == "and":
                # 層 1：and 必須是 cc（協調連接詞角色）
                if deprel != "cc":
                    # print(f"[KEEP]  'and' deprel={deprel}（非 cc），保守保留")
                    continue

                # 取得 B（and 的直接 head）
                b_id = word.head if word.head is not None else 0
                if b_id <= 0:
                    continue
                b_word = id_to_word.get(b_id)
                if b_word is None:
                    continue
                b_upos   = getattr(b_word, "upos", "")
                b_deprel = getattr(b_word, "deprel", "")

                # 層 2a：B 是名詞類 → 名詞片語並列，不切
                if b_upos in NOMINAL_UPOS:
                    # print(f"[KEEP]  'and' 連接名詞 B='{b_word.text}'({b_upos})，保留")
                    continue

                # 層 2b：B 是動詞類 → 進入層 3 確認
                if b_upos in VERBAL_UPOS:
                    # 層 3：B 若是 conj（並列節點），往上看 A（B 的 head）
                    # "swingers and go" 中 go.deprel=conj, go.head=swingers(NOUN)
                    # → B 是被誤判的動詞形式名詞，不應切割
                    if b_deprel == "conj":
                        a_id   = b_word.head if b_word.head is not None else 0
                        a_word = id_to_word.get(a_id)
                        a_upos = getattr(a_word, "upos", "") if a_word else ""

                        if a_upos in NOMINAL_UPOS:
                            # print(f"[KEEP]  'and' B='{b_word.text}'({b_upos}/conj) "
                                #   f"但 A='{a_word.text}'({a_upos}) 為名詞，推斷為名詞片語，保留")
                            continue
                        else:
                            # A 也是動詞 → 真正的動詞子句並列，切割
                            if start_char > 0:
                                # print(f"[SPLIT] 'and' B='{b_word.text}'({b_upos}/conj) "
                                #       f"A='{a_word.text}'({a_upos})，動詞並列，切割")
                                split_indices.append(start_char)
                    else:
                        # B 是動詞且 deprel 不是 conj（例如 ROOT/ccomp）→ 真實動詞子句
                        if start_char > 0:
                            # print(f"[SPLIT] 'and' 連接動詞子句 B='{b_word.text}'({b_upos}/{b_deprel})，切割")
                            split_indices.append(start_char)
                    continue

                # 層 2c：其他詞性（PART、DET 等）→ 保守不切
                # print(f"[KEEP]  'and' B='{b_word.text}'({b_upos})，詞性不明確，保守保留")

    split_indices = sorted(list(set(split_indices)))

    parts = []
    last_idx = 0
    for idx in split_indices:
        segment = text[last_idx:idx].strip()
        if segment:
            parts.append(segment)
        last_idx = idx

    final_segment = text[last_idx:].strip()
    if final_segment:
        parts.append(final_segment)

    if not parts:
        parts = [text]

    return parts


def main():
    parser = argparse.ArgumentParser(description="使用 Stanza 根據轉折詞分割 CSV 中的 TEXT 欄位")
    parser.add_argument('--input', type=str, default='train_2022.csv', help='輸入的 CSV 檔案')
    parser.add_argument('--output', type=str, default='dataset_split/train_2022_split_stanza_v4.csv', help='輸出的 CSV 檔案')
    args = parser.parse_args()

    print("正在初始化 Stanza Pipeline（含 NER）...")
    # v4 加入 ner processor 以支援 Entity Masking
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse,ner')
                
# else:
    print(f"\n正在讀取資料: {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"無法讀取檔案 {args.input}: {e}")
        return
        
    # print(f"將依據以下轉折詞對句子進行切割: {transition_words}")
    
    # 套用切割函數（v3：含 Entity Masking）
    df['SPLIT_TEXT'] = df['TEXT'].apply(lambda x: split_text_by_transition_v3(x, nlp))
    df['NUM_SPLITS'] = df['SPLIT_TEXT'].apply(len)
        
    print(f"正在將結果儲存至: {args.output}...")
    df.to_csv(args.output, index=False)
    print("完成！")

if __name__ == "__main__":
    main()
