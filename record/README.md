# Record 實驗紀錄整理

## 任務概述

- **任務**：情感分析（正/負向）二分類
- **模型架構**：`sentence-transformers/all-mpnet-base-v2`（768-dim）+ Random Forest
- **訓練資料**：`train_2022.csv`（2000 筆）
- **測試資料**：`test_no_answer_2022.csv`（11000 筆）
- **評估方式**：5-fold 80/20 CV，指標包含 Accuracy、F1、AUC、MSE

---

## 訓練流程說明

採用兩階段 Fine-tuning：

- **Phase 1（Domain Adaptation / MLM）**：以 Masked Language Model 方式對 SBERT 做無監督領域適應，使用 train + test 共 13000 筆文字（不含標籤）
- **Phase 2（Supervised Fine-tuning）**：以 `CosineSimilarityLoss` 在 CV 每個 round 內部進行有監督微調（同標籤 sim=1.0，不同標籤 sim=0.0），再用 RF 分類

---

## 檔案說明與實驗結果

### 一、Baseline 系列（以 baseline MLM checkpoint 為基礎）

#### `baseline_log.txt`
- **作用**：完整兩階段 baseline，Phase 1 從頭訓練 MLM（1 epoch, lr=3e-05），Phase 2 CV 內微調（3 epoch, lr=2e-05, pairs=3000）
- **儲存**：`_da_model.pt`（Phase 1）、`_ft_model.pt`（Phase 2 全量）
- **5-fold CV 結果**：

| ACC | F1 | AUC | MSE |
|---|---|---|---|
| 0.8155 ± 0.0111 | 0.8137 ± 0.0116 | 0.8936 ± 0.0108 | 0.1436 ± 0.0089 |

---

#### `vbf_a_output_log.txt`
- **作用**：載入 `baseline_da_model.pt`，Phase 2 改用更多 pairs（pairs=5000, 3 epoch, lr=2e-05）
- **觀察**：訓練不穩定，部分 round loss 卡在 ~0.27 無法收斂，導致結果大幅下降
- **5-fold CV 結果**：

| ACC | F1 | AUC | MSE |
|---|---|---|---|
| 0.6345 ± 0.1006 | 0.6328 ± 0.0996 | 0.6689 ± 0.1138 | 0.2213 ± 0.0401 |

> 結論：pairs=5000 訓練不穩定，放棄此設定

---

#### `vbf_c_output_log.txt`
- **作用**：載入 `baseline_da_model.pt`，Phase 2 縮減至 1 epoch（pairs=3000, lr=2e-05）
- **5-fold CV 結果**：

| ACC | F1 | AUC | MSE |
|---|---|---|---|
| 0.8245 ± 0.0123 | 0.8221 ± 0.0147 | 0.8946 ± 0.0118 | 0.1303 ± 0.0072 |

> 結論：1 epoch Phase 2 結果與 3 epoch 相近，MSE 略優

---

### 二、MLM 超參數搜尋（僅 Phase 1，無 CV 結果）

以下四個 log 僅記錄 Phase 1 MLM 訓練過程，目的是找到更好的 DA checkpoint，未進行完整 CV 評估。

#### `vbf_d1_output_log.txt`
- **設定**：MLM epochs=3, lr=1e-05, batch=16
- **結果**：訓練完成，avg loss 收斂至約 4.22，儲存 `_da_model.pt`（中途中斷，未完成 CV）

#### `vbf_d2_output_log.txt`
- **設定**：MLM epochs=2, lr=5e-05, batch=16
- **結果**：訓練完成，avg loss 約 3.77（lr 較大，收斂較快）

#### `vbf_d3_output_log.txt`
- **設定**：MLM epochs=2, lr=4e-05, batch=16
- **結果**：訓練完成，avg loss 約 3.78，儲存 `vbf_d3_da_model.pt`
- **後續**：此 checkpoint 作為 `baseline-d3`、`vd3_a/b/c` 的基礎

#### `vbf_d4_output_log.txt`
- **設定**：MLM epochs=2, lr=3e-05, batch=16
- **結果**：訓練完成（MLM Phase 1 only，未完成 CV）

---

### 三、基於 vbf_d3 checkpoint 的 Phase 2 調參

以下實驗均載入 `vbf_d3_da_model.pt` 作為 Phase 1 起點，探索 Phase 2 不同設定。

#### `baseline-d3_output_log.txt`
- **設定**：Phase 2 epochs=3, lr=2e-05, pairs=3000
- **5-fold CV 結果**：

| ACC | F1 | AUC | MSE |
|---|---|---|---|
| 0.8300 ± 0.0112 | 0.8304 ± 0.0103 | 0.8962 ± 0.0079 | 0.1338 ± 0.0081 |

> 結論：使用 vbf_d3 DA 後，F1 與 AUC 均優於原始 baseline

---

#### `vd3_a_output_log.txt`
- **設定**：Phase 2 epochs=3, lr=2e-05, pairs=3000（重跑，不同隨機種子）
- **5-fold CV 結果**：

| ACC | F1 | AUC | MSE |
|---|---|---|---|
| 0.8255 ± 0.0109 | 0.8246 ± 0.0098 | 0.8960 ± 0.0157 | — |

---

#### `vd3_b_output_log.txt`
- **設定**：Phase 2 epochs=3, lr=2e-05, pairs=3000（重跑，驗證穩定性）
- **5-fold CV 結果**：

| ACC | F1 | AUC | MSE |
|---|---|---|---|
| 0.8270 ± 0.0068 | 0.8275 ± 0.0080 | 0.8969 ± 0.0093 | — |

> 結論：vd3_a / vd3_b 結果穩定，std 小

---

#### `vd3_c_output_log.txt`
- **設定**：Phase 2 epochs=3, **lr=1e-05**（降低 learning rate）, pairs=3000
- **5-fold CV 結果**：

| ACC | F1 | AUC | MSE |
|---|---|---|---|
| 0.8325 ± 0.0113 | 0.8306 ± 0.0130 | 0.9019 ± 0.0111 | 0.1320 ± 0.0070 |

> 結論：Phase 2 降低 lr 略有提升，AUC 達 0.9019

---

### 四、Pseudo-Label 擴充訓練資料

#### `mpnet_ab_log.txt`
- **作用**：Pseudo-Label 策略 A+B（one-shot），MLM（2ep, lr=4e-05）→ CV 微調 → 對測試集高信度樣本（prob ≤ 0.15 或 ≥ 0.85）打偽標籤 → 合併訓練集重新訓練
- **5-fold CV 結果（初始輪）**：

| ACC | F1 | AUC | MSE |
|---|---|---|---|
| 0.8320 ± 0.0040 | 0.8320 ± 0.0066 | 0.9012 ± 0.0071 | 0.1361 ± 0.0070 |

> 結論：std 最小（0.0040），結果最穩定；AUC 達 0.9012

---

#### `mpnet_c_log.txt`
- **作用**：Pseudo-Label 策略 C（iterative），MLM（2ep, lr=2e-05）→ 重複 3 次「CV 微調 → 打偽標籤 → 重訓」迭代
- **5-fold CV 結果**：

| ACC | F1 | AUC | MSE |
|---|---|---|---|
| 0.8285 ± 0.0161 | 0.8271 ± 0.0166 | 0.8991 ± 0.0130 | 0.1336 ± 0.0115 |

> 結論：迭代 pseudo-label 效果略遜於一次性 A+B 策略，但仍優於純 baseline

---

## 綜合比較

| 實驗 | Phase 1 | Phase 2 設定 | F1 (mean) | AUC (mean) | 備註 |
|---|---|---|---|---|---|
| baseline | MLM 1ep lr=3e-05 | 3ep lr=2e-05 p=3000 | 0.8137 | 0.8936 | 基準 |
| vbf_a | baseline_da | 3ep lr=2e-05 p=5000 | 0.6328 | 0.6689 | 不穩定，放棄 |
| vbf_c | baseline_da | 1ep lr=2e-05 p=3000 | 0.8221 | 0.8946 | 1ep 即足夠 |
| baseline-d3 | vbf_d3_da | 3ep lr=2e-05 p=3000 | 0.8304 | 0.8962 | DA 升級有效 |
| vd3_a | vbf_d3_da | 3ep lr=2e-05 p=3000 | 0.8246 | 0.8960 | 重複驗證 |
| vd3_b | vbf_d3_da | 3ep lr=2e-05 p=3000 | 0.8275 | 0.8969 | 結果穩定 |
| vd3_c | vbf_d3_da | 3ep **lr=1e-05** p=3000 | 0.8306 | **0.9019** | 降 lr 微幅提升 |
| mpnet_ab | MLM 2ep lr=4e-05 | 3ep lr=2e-05 p=3000 + pseudo-A+B | 0.8320 | 0.9012 | std 最小 |
| mpnet_c | MLM 2ep lr=2e-05 | 3ep lr=2e-05 p=3000 + pseudo-C×3 | 0.8271 | 0.8991 | 迭代效益有限 |

**最佳 AUC**：`vd3_c`（0.9019）  
**最穩定**：`mpnet_ab`（ACC std=0.0040）  
**最差**：`vbf_a`（pairs=5000 導致訓練不穩定）
