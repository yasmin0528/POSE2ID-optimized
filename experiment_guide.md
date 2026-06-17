# PQ-IPG 实验指导

## Pose Quality-Aware Identity Preserving Generation

---

## 环境要求

确保所有依赖已安装：

```bash
# IPG 推理依赖
pip install -r IPG/requirements.txt

# TransReID 评估依赖
pip install -r demo/TransReID-main/requirements.txt
```

工作目录为项目根目录 `/mnt/d/homework/计算机视觉/Pose2ID`。

---

## 实验 1: PQ-IPG 权重可视化

验证姿态质量计算逻辑，无需 GPU：

```bash
# 基础可视化（8个标准姿态的 PQ-IPG 权重）
python visualize_pq_ipg.py 
    --pose_dir IPG/standard_poses 
    --output viz_pq_ipg.png 
    --alpha 0.4 --beta 0.3

# 额外分析 VeRi 数据集的真实关键点质量
python visualize_pq_ipg.py 
    --pose_dir IPG/standard_poses 
    --keypoint_file demo/TransReID-main/datasets/keypoint_train.txt
Found 8 standard poses in IPG/standard_poses
  1.jpg: quality = 0.655897
  2.jpg: quality = 0.575533
  3.jpg: quality = 0.573451
  4.jpg: quality = 0.501651
  5.jpg: quality = 0.647325
  6.jpg: quality = 0.626394
  7.jpg: quality = 0.590326
  8.jpg: quality = 0.630734

============================================================
PQ-IPG Weight Summary:
============================================================
  Pose 1 (1.jpg): quality=0.6559, weight=0.1391
  Pose 2 (2.jpg): quality=0.5755, weight=0.1185
  Pose 3 (3.jpg): quality=0.5735, weight=0.1180
  Pose 4 (4.jpg): quality=0.5017, weight=0.1022
  Pose 5 (5.jpg): quality=0.6473, weight=0.1368
  Pose 6 (6.jpg): quality=0.6264, weight=0.1312
  Pose 7 (7.jpg): quality=0.5903, weight=0.1220
  Pose 8 (8.jpg): quality=0.6307, weight=0.1323

  Equal weight baseline: 0.1250 per pose
  Max weight: 0.1391 (Pose 1)
  Min weight: 0.1022 (Pose 4)
  Weight range: 0.0369

Visualization saved to: viz_pq_ipg.png

Analyzing keypoint quality from: demo/TransReID-main/datasets/keypoint_train.txt
Total samples: 38632
============================================================
  Mean quality: 0.7550 +/- 0.1277
  Min quality: 0.0000, Max quality: 1.0236
  Mean visible keypoints: 11.5 / 20
  Samples with < 5 keypoints: 208 (0.5%)
  Samples with >= 15 keypoints: 607 (1.6%)

python visualize_pq_ipg.py 
    --pose_dir IPG/standard_poses 
    --keypoint_file demo/TransReID-main/datasets/keypoint_test.txt
```
Found 8 standard poses in IPG/standard_poses
  1.jpg: quality = 0.655897
  2.jpg: quality = 0.575533
  3.jpg: quality = 0.573451
  4.jpg: quality = 0.501651
  5.jpg: quality = 0.647325
  6.jpg: quality = 0.626394
  7.jpg: quality = 0.590326
  8.jpg: quality = 0.630734

============================================================
PQ-IPG Weight Summary:
============================================================
  Pose 1 (1.jpg): quality=0.6559, weight=0.1391
  Pose 2 (2.jpg): quality=0.5755, weight=0.1185
  Pose 3 (3.jpg): quality=0.5735, weight=0.1180
  Pose 4 (4.jpg): quality=0.5017, weight=0.1022
  Pose 5 (5.jpg): quality=0.6473, weight=0.1368
  Pose 6 (6.jpg): quality=0.6264, weight=0.1312
  Pose 7 (7.jpg): quality=0.5903, weight=0.1220
  Pose 8 (8.jpg): quality=0.6307, weight=0.1323

  Equal weight baseline: 0.1250 per pose
  Max weight: 0.1391 (Pose 1)
  Min weight: 0.1022 (Pose 4)
  Weight range: 0.0369

Visualization saved to: viz_pq_ipg.png

Analyzing keypoint quality from: demo/TransReID-main/datasets/keypoint_test.txt
Total samples: 12080
============================================================
  Mean quality: 0.7563 +/- 0.1319
  Min quality: 0.0000, Max quality: 0.9886
  Mean visible keypoints: 11.6 / 20
  Samples with < 5 keypoints: 30 (0.2%)
  Samples with >= 15 keypoints: 193 (1.6%)

输出：`viz_pq_ipg.png`，展示每个标准姿态的骨骼图、权重值、质量条。

---

## 实验 2: IPG 生成图可视化（含 PQ-IPG 权重）

运行 IPG 推理，生成图中会显示每个姿态的质量权重：

**必要条件**：已下载预训练权重到 `pretrained/` 目录。

```bash
cd IPG

python inference.py 
    --ckpt_dir /root/autodl-tmp/Pose2ID/pretrained 
    --pose_dir standard_poses 
    --ref_dir demo 
    --out_dir output_pqipg 
    --config ./configs/inference.yaml
```

输出的图片在 `IPG/output_pqipg/` 目录下，每个生成结果包含：
- 每个姿态的 PQ-IPG 权重数值（`w=0.xxx`）
- 彩色质量条（绿色=高质量，红色=低质量）
- 参考线（蓝色虚线）表示等权重基线

---

## 实验 3: TransReID 评估（消融实验）

### 3.1 修改配置文件

编辑 `demo/TransReID-main/configs/Market/vit_transreid_stride.yml` 中的 TEST 部分：

#### 基线：标准 IPG（等权重平均）
```yaml
TEST:
  IPG: True
  PQ_IPG: False          # 关闭 PQ-IPG
  NFC: True
  POSE_DIR: 'IPG/standard_poses'
```

#### PQ-IPG：姿态质量加权融合
```yaml
TEST:
  IPG: True
  PQ_IPG: True           # 开启 PQ-IPG
  NFC: True
  POSE_DIR: 'IPG/standard_poses'
```

### 3.2 运行评估

**必要条件**：
1. 已生成 IPG 图像到 `bounding_box_test_gen/pose{N}/` 和 `query_gen/pose{N}/`
2. 已下载 TransReID 预训练权重

```bash
cd demo/TransReID-main

# 基线：标准 IPG
cd demo/TransReID-main
python test.py \
    --config_file configs/Market/vit_transreid_stride.yml \
    TEST.PQ_IPG False

# PQ-IPG：姿态质量加权
python test.py \
    --config_file configs/Market/vit_transreid_stride.yml \
    TEST.PQ_IPG True
```

### 3.3 结果记录

| Method           | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) | mAP (%) | ΔmAP  |
| ---------------- | ---------- | ---------- | ----------- | ------- | ----- |
| Pose2ID Baseline | 94.77      | 97.06      | 98.16       | 90.40   | -     |
| Pose2ID + PQ-IPG | 94.80      | 97.12      | 98.19       | 90.37   | -0.03 |


---

## 实验 4: 消融：不同权重策略对比

通过修改 `utils/pose_quality.py` 中的参数，对比不同权重策略：

### 策略 A: 等权重（基线）
已在 baseline 中：`feat_ipg = feat_ipg / len(imgs_ipg)`

### 策略 B: 仅骨骼密度
```bash
python test.py \
    --config_file configs/Market/vit_transreid_stride.yml \
    TEST.PQ_IPG True \
    TEST.POSE_DIR 'IPG/standard_poses'
```
对应代码中 `alpha=0.7, beta=0.3, gamma=0.0`

### 策略 C: 仅对称性（正面优先）
对应代码中 `alpha=0.0, beta=0.0, gamma=1.0`

### 策略 D: 完整 PQ-IPG（推荐）
对应代码中 `alpha=0.4, beta=0.3, gamma=0.3`（默认值）

如需修改权重系数，编辑 `utils/pose_quality.py` 中的默认参数。

### 结果记录表

| Weight Strategy                               | Rank-1 (%) | mAP (%) | ΔmAP  |
| --------------------------------------------- | ---------- | ------- | ----- |
| Equal Weight (Baseline)                       | 94.77      | 90.40   | -     |
| Density + Confidence Only (α=0.7, β=0.3, γ=0) | 94.89      | 90.39   | -0.01 |
| Symmetry Only (α=0, β=0, γ=1)                 | 94.77      | 90.32   | -0.08 |
| PQ-IPG (Full)                                 | 94.80      | 90.37   | -0.03 |


---

## 文件清单

| 文件 | 说明 | 修改类型 |
|------|------|----------|
| `demo/TransReID-main/utils/pose_quality.py` | PQ-IPG 核心模块：质量计算 + 权重归一化 | **新增** |
| `demo/TransReID-main/processor/processor.py` | 推理流程：加载权重 → 加权融合特征 | **修改** |
| `demo/TransReID-main/config/defaults.py` | 新增 `TEST.PQ_IPG` 和 `TEST.POSE_DIR` 配置项 | **修改** |
| `demo/TransReID-main/configs/Market/vit_transreid_stride.yml` | 启用 PQ-IPG 配置文件 | **修改** |
| `IPG/inference.py` | 生成时可视化 PQ-IPG 权重 | **修改** |
| `visualize_pq_ipg.py` | 独立可视化脚本 | **新增** |

---


## 论文公式对照

| 公式 | 代码位置 | 说明 |
|------|----------|------|
| Q_i = α · (1/K) Σ c_k + β · V_i/K | `pose_quality.py:compute_pose_quality_from_image()` | 姿态质量分数 |
| w_i = exp(Q_i) / Σ exp(Q_j) | `pose_quality.py:compute_pq_ipg_weights()` | Softmax 归一化 |
| f_PQ-IPG = Σ w_i · f_i | `processor.py:do_inference()` | 加权特征融合 |

