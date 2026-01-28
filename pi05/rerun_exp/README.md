# 优化实验配置

重新运行已保存的优化实验配置（两阶段Guided策略）。

## 目录结构

```
optimization_exp/
├── rerun_saved_experiments.py       # 主执行脚本
├── rerun_saved_experiments.sh       # Shell启动脚本（后台运行）
├── deficiency_annotations.json      # 缺陷标注（46个成功场景）
├── scripts/
│   ├── run_experiment.py           # 实验执行引擎
│   └── scene_modifier.py           # 场景修改器
├── saved_configs/                   # 80个场景配置
│   └── scene_*/
│       ├── config_guided.yaml       # Guided组配置
│       ├── mutated_stacked.bddl     # 阶段1：移除遮挡物
│       └── mutated_only.bddl        # 阶段2：移动目标物
└── new_results/                     # 输出目录
```

## 快速开始

### 环境准备
```bash
conda activate env_isaaclab
```

### 运行实验

```bash
cd /home/decom/Downloads/IsaacLab/optimization_exp

# 方式1：后台运行所有场景
./rerun_saved_experiments.sh

# 方式2：运行指定场景
python rerun_saved_experiments.py --scenes "0,4,7,9"

# 方式3：运行场景范围
python rerun_saved_experiments.py --scenes "0-10"
```

### 查看日志
```bash
tail -f logs/rerun_experiment_*.log
```

## 参数说明

```bash
python rerun_saved_experiments.py \
    --scenes "0,4,7,9"                                      # 指定场景（或 --all 运行全部）
    --config-dir ./saved_configs                            # 配置目录
    --output-dir ./new_results                              # 输出目录
    --checkpoint-dir /home/decom/Downloads/IsaacLab/pi05_libero  # 模型路径
```

## 结果输出

```
new_results/
├── summary.json                    # 总结报告
└── scene_*/
    └── guided/
        └── group1_guided/
            ├── episode_0/
            │   ├── images/         # 运行图像
            │   └── stage_info.json # 阶段信息
            └── summary.json
```

## 实验说明

- **总场景数**: 80个（原实验baseline失败的场景）
- **每场景配置**: episodes=1, max_steps=300
- **两阶段策略**: 先移除遮挡物 → 再移动目标物
- **原实验成功率**: 57.5% (46/80)

## 缺陷标注

`deficiency_annotations.json` 包含46个成功场景的缺陷类型：

| 类型 | 说明 | 场景数 | 任务类型 |
|------|------|--------|----------|
| **1-IR** | Incomplete Reasoning（不完整推理） | 15 | pick_up |
| **2-IA** | Incorrect Alignment（错误对齐） | 13 | move_to |
| **3-OPD** | Out-of-Distribution（分布外性能下降） | 10 | put_on |
| **4-IPU** | Inaccurate Perception（感知不准确） | 8 | put_in |
