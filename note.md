# 紀錄

## 可調整的地方
1. Phase 1 — MLM Domain Adaptation
    在 baseline.ipynb 的 CFG 中：
    "da_epochs":   **1(vbf_b:3)**       # 可增加到 2-3，讓 encoder 更適應 domain
    "da_batch":    **1(vbf_b:16)**       # 顯存夠可調大（8/6/32）
    "da_lr":       3e-5    # 學習率，一般不需大動
    "da_mlm_prob": 0.15    # 遮蔽比例，可試 0.10~0.20
2. Phase 2 — Supervised Fine-tuning
    "ft_epochs":  **3(vbf_c:5)**        # 可增加，但小心 overfitting
    "ft_lr":      2e-5     # 學習率
    "ft_batch":   16
    "ft_pairs":   **3000(vbf_a1:5000,vbf_a2:5000)**     # pair 數量，可增加到 5000-10000
    "ft_warmup":  100      # warmup steps
3. ML 分類模型
    在 baseline.ipynb 的 _make_model 函式中：
    "model": "rf"   # 可換成 "gbm" 或 "elastic"4
    各模型的超參數也可以調：
    RF：n_estimators、max_depth
    GBM：n_estimators、learning_rate、max_depth
    ElasticNet：alpha、l1_ratio
4. 分類門檻
    目前寫死是 >= 0.5，可以根據 validation 結果微調：
    scores >= 0.5   # 可改成 0.4 或 0.6
5. 建議優先試的調整：
   1. ft_pairs 從 3000 → 5000（最低成本，通常有效果） **vbf_a**
   2. ~~"model" 換成 "gbm"（GBM 通常比 RF 準）~~
   3. ft_epochs 從 3 → 5  **vbf_c**