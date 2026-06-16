# ISAF 实验指导

## Identity Similarity-Aware Fusion

---

### 消融实验 1：整体对比

```bash
cd /root/autodl-tmp/POSE2ID-optimized

# Baseline: Mean Pooling (等权平均)
python demo/TransReID-main/test.py \
    --config_file demo/TransReID-main/configs/Market/vit_transreid_stride.yml \
    TEST.IPG True \
    TEST.PQ_IPG False \
    TEST.ISAF False

# ISAF: 身份相似度加权融合 (τ=0.07)
python demo/TransReID-main/test.py \
    --config_file demo/TransReID-main/configs/Market/vit_transreid_stride.yml \
    TEST.IPG True \
    TEST.PQ_IPG False \
    TEST.ISAF True \
    TEST.ISAF_TAU 0.07
```

---

### 消融实验 2：温度系数 τ 消融

```bash
# τ = 0.03
python demo/TransReID-main/test.py \
    --config_file demo/TransReID-main/configs/Market/vit_transreid_stride.yml \
    TEST.IPG True \
    TEST.PQ_IPG False \
    TEST.ISAF True \
    TEST.ISAF_TAU 0.03

# τ = 0.05
python demo/TransReID-main/test.py \
    --config_file demo/TransReID-main/configs/Market/vit_transreid_stride.yml \
    TEST.IPG True \
    TEST.PQ_IPG False \
    TEST.ISAF True \
    TEST.ISAF_TAU 0.05

# τ = 0.07（默认）
python demo/TransReID-main/test.py \
    --config_file demo/TransReID-main/configs/Market/vit_transreid_stride.yml \
    TEST.IPG True \
    TEST.PQ_IPG False \
    TEST.ISAF True \
    TEST.ISAF_TAU 0.07

# τ = 0.10
python demo/TransReID-main/test.py \
    --config_file demo/TransReID-main/configs/Market/vit_transreid_stride.yml \
    TEST.IPG True \
    TEST.PQ_IPG False \
    TEST.ISAF True \
    TEST.ISAF_TAU 0.10
```

---

### 消融实验 3：ISAF 与 PQ-IPG 联合

```bash
# ISAF + PQ-IPG 联合
python demo/TransReID-main/test.py \
    --config_file demo/TransReID-main/configs/Market/vit_transreid_stride.yml \
    TEST.IPG True \
    TEST.PQ_IPG True \
    TEST.ISAF True \
    TEST.ISAF_TAU 0.07
```

---

### 结果记录表

| Method | Rank-1 | Rank-5 | mAP | ΔmAP |
|--------|--------|--------|-----|------|
| Mean Pooling (baseline) | | | | - |
| PQ-IPG | | | | |
| ISAF (τ=0.03) | | | | |
| ISAF (τ=0.05) | | | | |
| ISAF (τ=0.07) | | | | |
| ISAF (τ=0.10) | | | | |
| ISAF + PQ-IPG | | | | |
