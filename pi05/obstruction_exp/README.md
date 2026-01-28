# Obstruction Experiments

遮挡物实验框架：用于测试机器人在有遮挡情况下的物体抓取能力。

## 📁 目录结构

```
obstruction/
├── configs/              # 实验配置文件（27个场景配置）
├── custom_bddl_files/    # 自定义场景定义文件
├── scripts/              # 核心脚本
│   ├── generate_configs.py                           # 配置生成器（主入口）
│   ├── generate_pair_experiments.py                   # Pair实验配置生成
│   ├── generate_single_obstruction_experiments.py     # Single实验配置生成
│   ├── run_experiment.py                             # 实验执行引擎
│   └── scene_modifier.py                             # 场景修改工具
├── results/              # 实验结果输出
├── logs/                 # 运行日志
├── run_single_experiment.sh      # 单实验启动脚本
└── run_all_pair_experiments.sh   # 批量实验启动脚本（27个场景）
```

## 🎯 实验类型

### Pair 实验（6组）
两个相同的遮挡物堆叠在目标物体上方：
- `butter` + 2个 `cream cheese`
- `butter` + 2个 `chocolate pudding`
- `cheese` + 2个 `butter`
- `cheese` + 2个 `chocolate pudding`
- `pudding` + 2个 `butter`
- `pudding` + 2个 `cream cheese`

### Single 实验（21组）
单个遮挡物在目标物体上方，目标物体为 `butter`/`cheese`/`pudding`（3种），遮挡物为 `plate`/`bowl`/`tomato_sauce`/`ketchup`/`alphabet_soup`/`orange_juice`/`milk`（7种），共 3×7=21 组。

## ⚙️ 配置文件说明

配置文件定义了实验的场景、任务和执行参数。以 `exp_single_butter_bowl.yaml` 为例：

```yaml
experiment:
  name: "single_butter_bowl"
  description: "单遮挡物实验：butter box(底) + 1个black bowl(顶)"

task:
  suite: "custom"
  task_name: "single_butter_bowl"

groups:
  # Group 1: Guided - 分步策略
  - name: "guided"
    description: "分两步：先移除遮挡物，再移动目标物体"
    stages:
      - stage_name: "remove_top_bowl"
        bddl_file: "experiments/obstruction/custom_bddl_files/single/butter_bowl_stacked.bddl"
        instruction: "put the black bowl in the basket"
        target_object: "akita_black_bowl_1"
      
      - stage_name: "move_bottom_butter"
        bddl_file: "experiments/obstruction/custom_bddl_files/single/butter_only.bddl"
        instruction: "put the butter box in the basket"
        target_object: "butter_1"
  
  # Group 2: Baseline - 直接策略
  - name: "baseline"
    description: "直接移动底层的butter box（有遮挡物）"
    bddl_file: "experiments/obstruction/custom_bddl_files/single/butter_bowl_stacked_baseline.bddl"
    instruction: "put the butter box in the basket"
    use_bddl_stacking: true

execution:
  episodes_per_group: 3           # 每组运行3次
  max_steps_per_episode: 300      # 每次最多300步
  seed_start: 43                  # 随机种子起始值
  checkpoint_dir: "./pi05_libero" # 模型检查点目录

output:
  results_dir: "./experiments/obstruction/results/single/butter_bowl"
  save_images: true               # 保存图像
  save_videos: false              # 不保存视频
```

### 配置生成流程

1. **定义场景组合** → 在 `generate_configs.py` 中定义物体对
2. **自动生成配置** → 运行生成器脚本创建 `.yaml` 文件
3. **生成 BDDL 文件** → 自动创建对应的场景定义文件（位于 `custom_bddl_files/`）

**生成命令：**
```bash
# 生成所有27个场景配置
python experiments/obstruction/scripts/generate_configs.py
```

## 🚀 快速启动

### 1. 运行单个实验

```bash
./run_single_experiment.sh configs/exp_single_butter_bowl.yaml
```

**功能：**
- 后台运行实验
- 自动记录日志到 `logs/` 目录
- 可随时查看实时日志

**监控命令：**
```bash
# 查看实时日志
tail -f logs/exp_single_butter_bowl_*.log

# 查看进程状态
ps aux | grep run_experiment
```

### 2. 批量运行所有实验

```bash
./run_all_pair_experiments.sh
```

**功能：**
- 自动运行所有 27 个场景（6个Pair + 21个Single）
- 顺序执行，自动跳过已完成实验
- 统一日志输出到 `logs/batch_pair_experiments_*.log`
- 支持中断后继续（检查 `report.json` 判断完成状态）

**监控命令：**
```bash
# 查看批量实验实时日志
tail -f logs/batch_pair_experiments_*.log

# 停止批量实验
kill $(cat logs/batch_pair_experiments.pid)
```

## 📊 结果输出

每个实验会在 `results/` 目录下生成：

```
results/
├── pair/
│   └── butter_cheese/
│       ├── group1_guided/
│       │   └── episode_0/
│       │       ├── images/          # 图像序列
│       │       └── metadata.json    # 执行元数据
│       ├── group2_baseline/
│       └── report.json              # 实验报告（成功/失败统计）
└── single/
    └── butter_bowl/
        ├── group1_guided/
        ├── group2_baseline/
        └── report.json
```

## 🛠️ 核心脚本说明

| 脚本 | 功能 |
|------|------|
| `generate_configs.py` | 自动生成所有实验配置文件和 BDDL 场景定义 |
| `run_experiment.py` | 实验执行引擎，加载配置、初始化环境、运行评估 |
| `scene_modifier.py` | 动态修改场景（用于 Guided 模式中的分阶段执行） |
| `run_single_experiment.sh` | 后台启动单个实验，自动日志记录 |
| `run_all_pair_experiments.sh` | 批量运行所有实验，支持断点续传 |

## 💡 使用技巧

1. **修改实验参数**：直接编辑 `configs/` 目录下的 `.yaml` 文件
2. **添加新场景**：修改 `generate_configs.py`，重新生成配置
3. **调试单个场景**：使用 `run_single_experiment.sh` 运行特定配置
4. **大规模实验**：使用 `run_all_pair_experiments.sh` 批量运行
5. **中断恢复**：批量脚本会自动跳过已有 `report.json` 的实验

## 🔍 实验监控

```bash
# 查看当前运行的实验
ps aux | grep run_experiment

# 查看某个实验的结果
cat results/single/butter_bowl/report.json

# 统计成功率
find results/ -name "report.json" | xargs grep -l "success"
```
