# SAM3-UQ

SAM3-UQ 是一个基于 SAM 3 多提示一致性的分割不确定性评估框架。它把 SAM 3 当作外部分割审查器，用 text prompt、box prompt、mask-derived prompt 等多视角输出评估已有分割模型的可靠性。

## 能力

- 模型-SAM 冲突不确定性：`1 - Dice(model_mask, sam_consensus)`
- SAM3 prompt 稳定性：多 prompt 输出之间的平均不一致性
- 边界不确定性：模型 mask 与 SAM consensus 的边界差异
- presence/score 不确定性：基于 SAM3 detection score 的概念存在风险
- 像素级 uncertainty map：多 prompt mask 方差 + 边界差异
- 图像级数据价值分数：不确定性 + 面积复杂度的轻量 proxy

## 安装

```bash
cd /Users/zhanghanwen/Documents/my_project/sam3-uq
python3 -m pip install -e ".[dev]"
```

如果要接入本地 SAM3：

```bash
python3 -m pip install -e ".[sam3]"
export PYTHONPATH=/Users/zhanghanwen/Documents/my_project/sam3-main:$PYTHONPATH
```

## 快速运行

先用 mock 后端验证框架：

```bash
python3 examples/make_sample.py
sam3-uq \
  --image examples/sample_image.npy \
  --mask examples/sample_mask.npy \
  --concept "polyp" \
  --backend mock \
  --output examples/out
```

真实 SAM3 后端：

```bash
sam3-uq \
  --image /path/to/image.png \
  --mask /path/to/model_mask.png \
  --concept "colon polyp" \
  --backend sam3-local \
  --sam3-root /Users/zhanghanwen/Documents/my_project/sam3-main \
  --device cuda \
  --output /path/to/out
```

## Prostate 风格数据批量运行

如果服务器数据类似：

```text
prostate/
├── data_npy/
├── label_npy/
├── val_data_npy/
└── val_label_npy/
```

可以先用 mock 后端跑 5 个样本验证流程：

```bash
DATA_ROOT=/path/to/prostate \
CONCEPT=prostate \
BACKEND=mock \
LIMIT=5 \
bash scripts/run_prostate_uq.sh
```

接入真实 SAM3：

```bash
DATA_ROOT=/path/to/prostate \
MASK_DIR=/path/to/model_predictions_npy \
CONCEPT=prostate \
BACKEND=sam3-local \
SAM3_ROOT=/path/to/sam3-main \
CHECKPOINT_PATH=/path/to/sam3.pt \
DEVICE=cuda \
bash scripts/run_prostate_uq.sh
```

输出会按样本写入 `sam3_uq_out/<sample>/`，并额外生成 `summary.csv`。`MASK_DIR` 应优先指向待评估模型的预测 mask；如果只是调试流程，可以临时使用 `val_label_npy`。

## 输出

```text
scores.json                # 图像级和实例级分数
uncertainty_pixel.npy      # 像素级不确定性图
consensus_mask.npy         # SAM3 多提示融合 mask
sam_masks_<prompt>.npy     # 每类 prompt 的 SAM3 输出
```

## 方法简述

对待评估模型输出 `Y_M` 做连通域，生成 box prompt；同时使用 concept text prompt。SAM3 输出集合记为 `{S_i}`，融合得到 `S*`：

```text
U_model_sam = 1 - Dice(Y_M, S*)
U_prompt    = 1 - mean Dice(S_i, S_j)
U_boundary  = boundary_disagreement(Y_M, S*)
U_presence  = 1 - mean SAM3 score

U_image = w1 U_model_sam + w2 U_prompt + w3 U_boundary + w4 U_presence
```

这个分数不是校准概率，适合做质量控制、人工复核排序和主动学习候选筛选。用于“数据价值”时，需要再结合多样性、类别稀缺性或下游训练增益验证。
