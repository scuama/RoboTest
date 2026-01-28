# Experiments

Fuzzing experiments for VLA models with scene variation and MCTS-guided exploration.

## Scripts

- **run_fuzzer.py**: Basic fuzzing with predefined test datasets
- **run_mcts_vla_fuzzer.py**: MCTS-guided fuzzing for adaptive exploration
- **model_interface.py**: Unified interface for VLA models
- **variation.py**: Scene variation generator (camera, lighting, objects)
- **random_camera.py**: Random camera pose variations
- **random_lighting.py**: Random lighting variations

## Usage

### Basic Fuzzing
```bash
python run_fuzzer.py \
    --data ../data/t-grasp_n-100_o-0_s-*.json \
    --model openvla-7b \
    --output ./results/grasp \
    --seed 42
```

### MCTS Fuzzing
```bash
python run_mcts_vla_fuzzer.py \
    --data ../data/t-move_n-100_o-0_s-*.json \
    --model rt_1_x \
    --output ./results/move_mcts
```

## Parameters

- `--data`: Path to test dataset (JSON)
- `--model`: Model name (rt_1_x, octo-base, openvla-7b, etc.)
- `--output`: Output directory for results
- `--seed`: Random seed for reproducibility
- `--lora_path`: LoRA adapter path for fine-tuned OpenVLA models
- `--resume`: Resume from previous run

## Output

Results are saved to the output directory:
```
results/
└── {task}/
    ├── episode_0/
    │   ├── images/       # Image sequence
    │   ├── options.json  # Scene configuration
    │   └── log.json      # Execution log
    └── report.json       # Summary report
```
