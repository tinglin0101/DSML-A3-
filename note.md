# 筆記

## 2026.4.14

**目標**: 使用vBERT_finetune: 將截斷從128改為256或384，使否對於學習有幫助
**預想**: 假設有提升準確度，可以試試small跟roberta，看一下有沒有差異
**結果**:

**128-cpu**
--- Epoch 1/3 ---
  Train Loss: 0.6143 | Train Acc: 0.6600
  Val   Loss: 0.5062  | Val   Acc: 0.7550
  ✓ 儲存最佳模型（Val Acc: 0.7550）

--- Epoch 2/3 ---
  Train Loss: 0.3818 | Train Acc: 0.8450
  Val   Loss: 0.4494  | Val   Acc: 0.7975
  ✓ 儲存最佳模型（Val Acc: 0.7975）

--- Epoch 3/3 ---
  Train Loss: 0.2450 | Train Acc: 0.9106
  Val   Loss: **0.4966**  | Val   Acc: 0.8025
  ✓ 儲存最佳模型（Val Acc: 0.8025）

**384-gpu**
--- Epoch 1/3 ---
  Train Loss: 0.6022 | Train Acc: 0.6744
  Val   Loss: 0.4623  | Val   Acc: 0.7725

--- Epoch 2/3 ---
  Train Loss: 0.3643 | Train Acc: 0.8481
  Val   Loss: 0.4460  | Val   Acc: 0.7975

--- Epoch 3/3 ---
  Train Loss: 0.2360 | Train Acc: 0.9187
  Val   Loss: 0.4745  | Val   Acc: 0.8075


### 2026.4.15

**目標**: 切分多情緒(from 雅涵)
**預想**: 
**結果**: