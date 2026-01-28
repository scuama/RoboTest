# Obstruction Experiments

Testing framework for robot manipulation under occlusion scenarios using LIBERO benchmark.

## Overview

Tests robot's ability to manipulate objects with obstructions:
- **Pair experiments** (6 scenes): Two identical obstructions stacked on target
- **Single experiments** (21 scenes): Single obstruction on target

## Structure

```
obstruction_exp/
├── scripts/                   # Core scripts
│   ├── generate_configs.py            # Config generator
│   ├── generate_pair_experiments.py   # Pair experiment generator
│   ├── generate_single_obstruction_experiments.py  # Single experiment generator
│   ├── run_experiment.py              # Experiment executor
│   └── scene_modifier.py              # Scene modifier
├── configs/                   # Experiment configs (27 scenes)
├── custom_bddl_files/         # Scene definitions
├── results/                   # Results
├── logs/                      # Logs
├── run_optimization.py        # Optimization script
├── run_mcts_fuzzer.py         # MCTS fuzzer
└── run_*.sh                   # Launch scripts
```

## Experiment Types

### Pair (6 scenes)
Target object with 2 identical obstructions stacked on top:
- butter + 2×cream_cheese
- butter + 2×chocolate_pudding
- cheese + 2×butter
- cheese + 2×chocolate_pudding
- pudding + 2×butter
- pudding + 2×cream_cheese

### Single (21 scenes)
Target object with 1 obstruction on top:
- **Targets**: butter, cheese, pudding (3)
- **Obstructions**: plate, bowl, tomato_sauce, ketchup, alphabet_soup, orange_juice, milk (7)
- **Total**: 3 × 7 = 21

## Strategies

### Guided (Two-stage)
1. Remove obstruction
2. Move target object

### Baseline (Direct)
Directly move target object (with obstruction)

## Usage

### Generate Configs
```bash
python scripts/generate_configs.py
```

### Run Single Experiment
```bash
./run_single_experiment.sh configs/exp_single_butter_bowl.yaml
```

### Run All Experiments
```bash
./run_all_pair_experiments.sh
```

### Monitor
```bash
tail -f logs/exp_single_butter_bowl_*.log
```

## Configuration Example

```yaml
experiment:
  name: "single_butter_bowl"

task:
  task_name: "single_butter_bowl"

groups:
  - name: "guided"
    stages:
      - stage_name: "remove_obstruction"
        bddl_file: "butter_bowl_stacked.bddl"
        instruction: "put the black bowl in the basket"
      - stage_name: "move_target"
        bddl_file: "butter_only.bddl"
        instruction: "put the butter box in the basket"
  
  - name: "baseline"
    bddl_file: "butter_bowl_stacked_baseline.bddl"
    instruction: "put the butter box in the basket"

execution:
  episodes_per_group: 3
  max_steps_per_episode: 300
  checkpoint_dir: "./pi05_libero"

output:
  results_dir: "./results/single/butter_bowl"
  save_images: true
```

## Output

```
results/
├── pair/butter_cheese/
│   ├── group1_guided/
│   │   └── episode_0/
│   │       ├── images/
│   │       └── metadata.json
│   ├── group2_baseline/
│   └── report.json
└── single/butter_bowl/
    ├── group1_guided/
    ├── group2_baseline/
    └── report.json
```

## Scripts

| Script | Function |
|--------|----------|
| `generate_configs.py` | Generate all 27 scene configs |
| `run_experiment.py` | Execute experiment from config |
| `scene_modifier.py` | Dynamically modify BDDL scenes |
| `run_optimization.py` | Run optimization on failures |
| `run_mcts_fuzzer.py` | MCTS-guided fuzzing |
