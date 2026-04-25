# baseline
- phase 1 end loss: 5.1226
- phase 2 validation acc(5-fold cross mean): 0.8155
- kaggle: 0.77410

# 撞牆

## vbf_a: overfitting
- based on: baseline 
- ft_pair: 5000 (bl:3000)
- phase 1: baseline
- phase 2 validation acc(5-fold cross mean): **0.6345**

## vbf_b: better(overfitting)
- based on: baseline 
- da_epochs: 3 (bl:1)
- da_batch: 16(bl:8)
- phase 1 end loss: **3.4458** -> 上上下下
- phase 2 validation acc(5-fold cross mean): **0.8365**
- kaggle: **0.76694**

## vbf_c: better?
- based on: baseline
- ft_epochs:1 (bl:3)
- phase 1: baseline
- phase 2 validation avv(5-fold cross mean): **0.8245**

# 第一階段微調

## vbf_d1
- based on: vbf_b
- da_epochs: 3 (bl:1)
- da_lr: 1e-5(vbf:2e-5)
- da_batch: 16(bl:8)
- phase 1 end loss: **4.2195?** -> 有一個loss上去

## vbf_d2
- based on: vbf_d1
- da_epochs: 2
- da_lr: 5e-5(vbf:2e-5)
- phase 1 end loss: **3.6947** -> 上上下下

## vbf_d3: best
- based on: vbf_d1
- da_epochs: 2
- da_lr: 4e-5(vbf:2e-5) 
- phase 1 end loss: **3.7769**

## vbf_d4
- based on: vbf_d1
- da_epochs: 2
- da_lr: 3e-5(vbf:2e-5)
- phase 1 end loss: **3.8669**

# 第二階段微調

## baseline-d3
- based on: vbf_d3
- phase 2 validation acc(5-fold cross mean): **0.8300, 0.0112**
- full dataset training loss: **0.0513** -> underfitting
- kaggle: **0.77630**

提醒: 
- learning rate: vali 有overfitting，r4 epo3、r5 epo3
- full 無 overfitting

## vd3_a
- based on: baseline-d3
- warn up: 30(bl_d3:100)
- phase 2 validation acc(5-fold cross mean): **0.8255, 0.0109**
- full dataset training loss: **0.0537** -> overfitting(上上下下)

## vd3_b
- based on: vd3_a
- ft_lr: 1e-5(bl_d3:2e-5)
- warn up: 30(bl_d3:100)
- phase 2 validation acc(5-fold cross mean): **0.8270, 0.0068**
- full dataset training loss: **0.0794** -> underfitting

## vd3_c
- based on: vd3_a
- warn up: 40(bl_d3:100)
- phase 2 validation acc(5-fold cross mean): **0.8325, 0.0113**
- full dataset training loss: **0.0527** -> 無上上下下
- kaggle: **0.76308**