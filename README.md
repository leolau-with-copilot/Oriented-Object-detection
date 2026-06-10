# Oriented Ship | AICC Round 7 — 实验代码

YOLO11s-OBB 有向边界框舰船检测。**从零训练**（无外部数据、无预训练权重），
随后做针对性微调与测试时增强（TTA）。指标：mAP50@R-IoU。

## 四个核心文件（最重要）

| 作用 | 文件 |
|---|---|
| ① 初始模型训练（从零） | `train/train_yolo_obb.py` |
| ② 微调（小目标/暗目标） | `finetune/train_finetune.py`（配合 `finetune/select_hard.py`） |
| ③ 测试时增强（TTA）推理 | `inference/tta_submission.py` |
| ④ 普通推理 | `inference/predict_submission.py` |

- **模型无需重新训练**：训练好的权重已随包提供 —— `weights/best.pt`（从零）、
  `weights/best_finetuned.pt`（微调后）。
- **TTA 已集成**：`inference/tta_submission.py` **默认加载 `best_finetuned.pt`**，
  对每张图做 4 视图 × 2 尺度推理并用旋转 NMS 合并，直接产出最终 `submission_tta.csv`。
  这就是取得 private **0.8759** 的那条推理路径。

### 直接复现最终提交（无需训练）
```bash
bash setup_server.sh                         # 安装依赖（ultralytics + cu124 torch + polars）
# 把 226 张官方测试图放到 data/test_images/
python inference/tta_submission.py           # -> submission_tta.csv（fine-tuned + TTA）
```

### 从头复现完整流程
```bash
python data_prep/prepare_yolo_obb.py         # 原始标注 -> YOLO-OBB 四点格式 + 768/135 划分
python train/train_yolo_obb.py               # 从零训练 -> best.pt
python finetune/select_hard.py               # 标记小/暗目标 -> hard_images.txt
python finetune/train_finetune.py            # 微调 -> best_finetuned.pt
python inference/tta_submission.py           # TTA 推理 -> submission_tta.csv
```

## 数据已随包提供，开箱即用

数据已打包在 `data/` 内，所有脚本的输入路径集中在 `paths.py`（默认指向本包的
`data/` 与 `weights/`），**无需任何配置即可运行**。布局如下：

```
data/
  images/{train,val}      # 训练/本地验证图（768 / 135）
  labels/{train,val}      # YOLO-OBB 四点标注（9 值）
  raw/labels/{train,val}  # 原始 "cx cy w h theta" 标注（5 值，供 quality_audit.ipynb）
  test_images/            # 226 张官方测试图
  data.yaml
weights/best.pt, best_finetuned.pt
```

> ⚠️ **唯一未打包**：外部泛化集 ShipRSImageNet（约 4.6 GB，未用于训练）。`eval/eval_shiprs_val500.py`没有改数据集无法运行


## 目录结构

```
paths.py                       全部输入/权重路径的唯一配置点
setup_server.sh                环境一键配置

data_prep/prepare_yolo_obb.py  原始 (cx,cy,w,h,θ) 标注 -> YOLO-OBB 四点格式，并做 768/135 划分
train/
  train_yolo_obb.py            从零训练入口（数据增强参数内置）
  check_dependencies.py        环境/数据自检
finetune/
  select_hard.py               标记小/暗“困难图” -> hard_images.txt
  train_finetune.py            在 best.pt 上对困难图过采样后续训（imgsz=1024）
inference/
  tta_submission.py            ★ 微调模型 + OBB-TTA -> submission_tta.csv（最终提交）
  predict_submission.py        整图普通推理 -> submission.csv
  tile_infer.py                切片推理实验（报告 §7.3）
eval/                          评测（报告数据来源）
  run_val_ours.py              本地验证集有向框指标
  eval_shiprs_val500.py        ShipRSImageNet 跨域评测（§6.5）
  tta_shiprs_eval.py           跨域 TTA：单视图 vs OBB-TTA（§7.4）
  download_shiprsimagenet.py   拉取外部泛化集
figures/                       报告插图生成（quality / augmentation / missed / recovered）
notebooks/quality_audit.ipynb  数据质量度量与可视化（§3）
weights/best.pt, best_finetuned.pt
```

## 成绩

| 阶段 | private | public |
|---|---|---|
| best.pt（从零） | 0.8021 | 0.7449 |
| + 针对性微调 | 0.8303 | 0.7572 |
| + 3 视图 OBB-TTA | 0.8572 | 0.8038 |
| + 4 视图 × 2 尺度 TTA（最终） | **0.8759** | **0.8145** |
