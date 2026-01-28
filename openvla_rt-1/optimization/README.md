# Optimization Framework - 穷举式优化系统

自动对失败案例进行穷举式优化的框架，直接扫描目录，无需 CSV 或配置文件。

## 🎯 核心特性

- **穷举所有策略**: 自动尝试所有可能的策略组合
- **无需配置**: 直接扫描 base_dir，从 log.json 判断失败案例
- **智能替换**: 根据物体类型自动选择替换对象
- **差异化尝试**: 不同策略有不同的尝试次数
- **详细记录**: 保存每次尝试的完整历史

## 📁 目录结构

```
optimization/
├── optimizer.py                      # 优化器（主脚本）
├── start_optimization.sh             # 启动脚本
├── fix_strategy_*.py                 # 各种优化策略脚本
├── replay_vla_actions.py             # VLA动作重放脚本
├── README.md                         # 本文档
├── {model}/{task}/                   # 各模型任务目录
│   ├── config.json                   # 任务配置（可选）
│   ├── working/                      # 工作目录
│   ├── success/                      # 成功案例
│   └── history.json                  # 优化历史记录
└── config.json.example               # 配置示例
```

## 🔧 可用策略

### 1. optimize_grasp（8次尝试）
优化抓取位置，多方向微调：
- right, left, up, down（距离 0.02m）
- right-up, right-down, left-up, left-down（距离 0.03m）

### 2. rotate_object（6次尝试）
旋转物体，改变朝向：
- 45°, 90°, 180°, 270°
- 随机旋转
- 侧躺模式

### 3. move_closer（3次尝试）
将物体移近机械臂：
- 20%, 30%, 40% 的距离

### 4. replace_object（0-1次尝试）
根据物体类型智能替换：
- **球状物体** → apple
- **罐装物体（开口）** → opened_coke_can
- **罐装物体（闭口）** → coke_can
- **其他物体** → 跳过此策略

**总计**: 最多 18 次尝试/案例

## 🚀 快速开始

### 基本用法

```bash
# 优化 move 任务的所有失败案例
bash optimization/start_optimization.sh --task move --model openvla-7b

# 优化 grasp 任务
bash optimization/start_optimization.sh --task grasp --model rt_1_x

# 或直接调用 Python 脚本
python3 optimization/optimizer.py --task move --model openvla-7b
```

### 测试单个案例

```bash
# 测试 move 任务的 episode 0
bash optimization/start_optimization.sh \
    --task move \
    --model openvla-7b \
    --episode 0

# 限制尝试次数（快速测试）
bash optimization/start_optimization.sh \
    --task move \
    --model openvla-7b \
    --episode 0 \
    --max-trials 5
```

### 后台运行

```bash
# 后台运行，处理所有失败案例
bash optimization/start_optimization.sh \
    --task move \
    --model openvla-7b \
    --background

# 查看日志
tail -f optimization/logs/optimization_*.log
```

## 📊 参数说明

### optimizer.py

```bash
python3 optimization/optimizer.py --task TASK --model MODEL [options]

必需参数:
  --task TASK           任务名称（如: move, grasp, pick_coke_can）
  --model MODEL         模型名称（如: openvla-7b, rt_1_x）

可选参数:
  --episode ID          仅处理指定的 episode（调试用）
  --max-trials N        限制每个案例的最大尝试次数
```

### start_optimization.sh

```bash
bash optimization/start_optimization.sh --task TASK --model MODEL [options]

必需参数:
  --task TASK           任务名称
  --model MODEL         模型名称

可选参数:
  --episode ID          仅处理指定的 episode
  --max-trials N        限制最大尝试次数
  --background          后台运行
  -h, --help            显示帮助
```

## 📈 工作流程

```
1. 自动查找结果目录
   ├─ 扫描 results/ 目录
   ├─ 查找 t-{task}_* 目录
   └─ 查找 {model}* 子目录
   ↓
2. 扫描失败案例
   ├─ 遍历所有 episode 目录
   ├─ 从 log.json 读取成功/失败状态
   └─ 从 options.json 提取物体信息
   ↓
3. 对每个失败案例:
   ├─ 备份原始配置
   ├─ 生成所有策略组合（基于物体类型）
   └─ 逐个尝试策略:
      ├─ 重置到原始配置
      ├─ 应用策略（修改 options.json）
      ├─ 运行推理（调用 run_fuzzer.py）
      ├─ 检查结果
      └─ 成功 → 保存配置，继续下一案例
         失败 → 尝试下一策略
   ↓
4. 生成优化报告
```

## 📝 输出文件

### 成功案例
```
optimization/results/{model}/{task}/success/{episode_id}/
├── options.json           # 成功的配置
├── origin.json            # 原始配置
└── strategy_info.json     # 策略信息
```

### 优化历史
```
optimization/history/{task}/{episode_id}_history.json
{
  "episode_id": "0",
  "timestamp": "2026-01-09T...",
  "trials": [
    {
      "strategy": "optimize_grasp",
      "params": {"direction": "right", "distance": 0.02},
      "description": "优化抓取 #1: right 方向 0.02m",
      "applied": true,
      "inference_success": false
    },
    ...
  ],
  "final_success": true,
  "total_trials": 5
}
```

### 优化报告
```
optimization/reports/{task}/report_20260109_HHMMSS.json
{
  "timestamp": "2026-01-09T...",
  "statistics": {
    "total_cases": 75,
    "success": 30,
    "failed": 45,
    "total_trials": 380,
    "strategy_success_count": {
      "optimize_grasp": 12,
      "rotate_object": 8,
      "move_closer": 7,
      "replace_object": 3
    }
  }
}
```

## 🔍 查看结果

```bash
# 查看优化报告
cat optimization/reports/move/report_*.json | jq .

# 查看成功案例列表
ls optimization/results/openvla-7b/move/success/

# 查看某个 episode 的优化历史
cat optimization/history/move/0_history.json | jq .

# 统计成功率
python3 -c "
import json
from pathlib import Path
report = sorted(Path('optimization/reports/move').glob('report_*.json'))[-1]
data = json.loads(report.read_text())
stats = data['statistics']
rate = stats['success'] / stats['total_cases'] * 100
print(f'成功率: {rate:.1f}% ({stats[\"success\"]}/{stats[\"total_cases\"]})')
"
```

## ⚙️ 策略脚本说明

所有策略脚本位于 `optimization/` 目录：

- `fix_strategy_move_closer.py` - 将物体移近机械臂
- `fix_strategy_optimize_grasp.py` - 优化抓取位置
- `fix_strategy_rotate_object.py` - 旋转物体
- `fix_strategy_replace_object.py` - 替换物体

每个脚本接受：
```bash
python3 fix_strategy_*.py <base_dir> <episode_id> <output_dir> [--param value]
```

## 🎨 自定义策略

要添加新策略：

1. 创建策略脚本 `fix_strategy_new_strategy.py`
2. 在 `optimizer.py` 中添加：
   ```python
   STRATEGY_SCRIPTS["new_strategy"] = PROJECT_ROOT / "optimization/fix_strategy_new_strategy.py"
   ```
3. 在 `StrategyMatrix` 中定义尝试参数：
   ```python
   NEW_STRATEGY_TRIALS = [
       {"param1": value1},
       {"param2": value2},
   ]
   ```

## 💡 Tips

1. **先小规模测试**: 使用 `--episode 0 --max-trials 5` 快速验证
2. **后台运行**: 使用 `--background` 避免占用终端
3. **查看日志**: `tail -f optimization/logs/optimization_*.log` 实时监控
4. **断点续传**: 已成功的案例会自动跳过，可以安全中断重启
5. **结果按任务分类**: 不同任务的结果保存在不同目录，如 `results/{model}/move/`

## 🐛 故障排除

### 问题：策略应用失败
- 检查策略脚本是否存在
- 检查参数是否正确
- 查看详细错误信息

### 问题：推理超时
- 默认超时 600 秒
- 检查环境资源是否充足
- 考虑减少 `--max-trials`

### 问题：找不到原始配置
- 确认 results 目录路径正确
- 检查 `{episode_id}/options.json` 是否存在

### 问题：找不到结果目录
- 确认 `results/t-{task}_*/` 目录存在
- 确认 `{model}*` 子目录存在
- 使用完整路径测试

## 📞 支持

如有问题，请查看：
- 优化历史: `optimization/history/{task}/`
- 日志文件: `optimization/logs/`
- 报告文件: `optimization/reports/{task}/`
