# Optimization Framework

Automated optimization framework for failed test cases using exhaustive strategy exploration.

## Overview

Automatically optimizes failed manipulation episodes by trying different fix strategies:
- Optimize grasp positions (8 directions)
- Rotate object orientations (6 angles)
- Move object closer to robot (3 distances)
- Replace with similar objects (type-based)

## Structure

```
optimization/
├── optimizer.py                   # Main optimizer
├── start_optimization.sh          # Launch script
├── fix_strategy_*.py              # Fix strategies
├── replay_vla_actions.py          # Action replay
├── {model}/{task}/                # Results by model/task
│   ├── working/                   # Working directory
│   ├── success/                   # Successful fixes
│   └── history.json               # Optimization history
```

## Usage

### Basic
```bash
bash start_optimization.sh --task grasp --model openvla-7b
```

### Single Episode
```bash
bash start_optimization.sh --task move --model rt_1_x --episode 0
```

### Background
```bash
bash start_optimization.sh --task grasp --model openvla-7b --background
tail -f logs/optimization_*.log
```

## Strategies

| Strategy | Trials | Description |
|----------|--------|-------------|
| `optimize_grasp` | 8 | Adjust grasp position (right, left, up, down, diagonals) |
| `rotate_object` | 6 | Rotate object (45°, 90°, 180°, 270°, random, sideways) |
| `move_closer` | 3 | Move object closer (20%, 30%, 40%) |
| `replace_object` | 0-1 | Replace with similar object (type-dependent) |

**Total**: Up to 18 trials per failed case

## Output

### Success Cases
```
optimization/{model}/{task}/success/{episode_id}/
├── options.json        # Fixed configuration
├── origin.json         # Original configuration
└── strategy_info.json  # Strategy details
```

### History
```
optimization/history/{task}/{episode_id}_history.json
```

### Reports
```
optimization/reports/{task}/report_*.json
```

## Parameters

```bash
python optimizer.py --task TASK --model MODEL [options]

Required:
  --task TASK       Task name (grasp, move, put-in, put-on)
  --model MODEL     Model name (openvla-7b, rt_1_x, etc.)

Optional:
  --episode ID      Process specific episode only
  --max-trials N    Limit maximum trials per case
```

## Workflow

1. Scan results directory for failed cases
2. For each failure:
   - Backup original configuration
   - Generate strategy combinations
   - Try each strategy sequentially
   - If success → save and continue to next case
   - If all fail → record and continue
3. Generate optimization report
