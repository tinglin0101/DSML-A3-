# 紀錄
記錄將文本切割後再情緒分析

## 情緒分析(切後)
- semantics: 使用train_2022_split_v2
- 可以增強
  - and
  - 轉折詞的

## pipeline-v1
from gemini

1. 重新審視評估指標與資料分佈
   1. 可以引入 F1-score、PRAUC (Precision-Recall AUC) 或是 G-mean 來進行評估。
   2. 這對於處理潛在的資料不平衡（Imbalanced Dataset）問題非常有幫助，能更真實反映模型對少數類的判別能力。

2. 寬表（Wide Table）特徵對齊的缺陷 -> **v2完成**
   1. Step 3 中將不同 sub_id 橫向展開成寬表 (sub0_Anger, sub1_Anger...) 在機器學習中容易造成「特徵錯位」與「特徵稀疏」。
   2. 問題點： 假設 A 句的轉折重點在 sub_0，B 句的轉折重點在 sub_3，模型會將它們視為完全不同的特徵，難以泛化。句子切分越多，填補的 0 也越多，導致維度災難。
   3. 優化建議（加權聚合）： 不要橫向展開，而是將同一列（row_id）的子段落特徵直接進行加權加總或平均。
   4. 例如，單一文本的最終 Anger 特徵可以設計為：Sum( Anger_count_i * weight(i) )。這樣不論文本被切成幾段，輸入給模型的永遠是固定的 16 個特徵，不但降低維度，也保留了尾段權重較高的設計邏輯。

3. 詞典方法（Lexicon）的語義局限性 -> **v3: BERT、v4:Sentence-BERT** 
   1. NRC 情緒分析是基於詞頻（Lexicon-based），這在遇到真實語意時非常脆弱。
   2. 問題點： 詞典無法處理否定句（例如「Not happy at all」會被算作 Joy）以及上下文反諷。
   3. 優化建議： 考慮引入更強的語義捕捉工具。既然目標是系統的落地，可以嘗試抽取預訓練語言模型（如 BERT、RoBERTa 或其他輕量級 Transformer）的 Embedding 作為特徵。如果預算或算力有限，哪怕是使用 TF-IDF 搭配 N-gram，在捕捉短語（如 "not good"）上的效果通常也比單純的 NRC 詞頻更好。

4. 模型選擇與任務不對齊 -> **v2完成**
   1. 在 Step 3 中，你使用了迴歸模型（Ridge, Lasso, BayesianRidge）去預測連續分數，然後以 0.5 切割來做分類。
   2. 問題點： 迴歸模型優化的目標是連續數值的 MSE，但分類任務更適合優化 Log-loss（Cross-Entropy）。迴歸模型容易受到極端值的懲罰，導致決策邊界偏移。
   3. 優化建議： 直接將模型替換為原生的分類器。例如 LogisticRegression、**RandomForestClassifier** 或 LGBMClassifier。利用分類器的 predict_proba() 輸出機率值，再依據 PRAUC 的表現來決定最佳的決策閾值，而不是寫死 0.5。
   
### 結果(validation acc)
v1: 約0.6
v2: 0.5630(AUC)
v3: 0.7350(AUC) -> kaggle: 0.6336
v4: 0.7159(AUC) -> kaggle: 0.6752

### 可以做更改的部分
1. bert 微調，用一小部分的微調樣本
   1. 要建立多個句子微調樣本
2. v3 預測些微不平衡 -> **vX.1**
   1. 0->1025
   2. 1->975
3. 加權方式修改 -> **vX.1**
   1. 目前(v2,v3,v4): Sum_i( feature_i * sqrt(rank_i + 1) )
4. 重新檢視切割方式

## v4 測試檔

### 原始: 5-cross validation(random forest)

Confusion Matrix:
   [[668 332]
   [296 704]]
ACC      : mean=0.6860  std=0.0098
PRE      : mean=0.6796  std=0.0086
REC      : mean=0.7040  std=0.0271
F1       : mean=0.6913  std=0.0141
AUC      : mean=0.7525  std=0.0071
MSE      : mean=0.2100  std=0.0014
