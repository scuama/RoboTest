# OpenVLA & RT-1 Testing Framework

Fuzzing and optimization framework for testing Vision-Language-Action models (OpenVLA and RT-1) on robotic manipulation tasks.

## Structure

```
openvla_rt-1/
├── data/              # Test datasets (grasp, move, put-in, put-on)
├── experiments/       # Fuzzing experiments
│   ├── run_fuzzer.py           # Basic fuzzing
│   ├── run_mcts_vla_fuzzer.py  # MCTS-guided fuzzing
│   ├── model_interface.py      # VLA model interface
│   └── variation.py            # Scene variation generator
└── optimization/      # Automated optimization framework
    ├── optimizer.py            # Main optimizer
    ├── start_optimization.sh   # Launch script
    └── fix_strategy_*.py       # Optimization strategies
```

## Quick Start

### Fuzzing
```bash
cd experiments
python run_fuzzer.py --data ../data/t-grasp_n-100_o-0_s-*.json --model openvla-7b
```

### Optimization
```bash
cd optimization
bash start_optimization.sh --task grasp --model openvla-7b
```

## Supported Tasks

- **grasp**: Object grasping
- **move**: Object movement
- **put-in**: Placing into containers
- **put-on**: Placing on surfaces

## Supported Models

- `rt_1_x`, `rt_1_400k`, `rt_1_58k`, `rt_1_1k`
- `octo-base`, `octo-small`
- `openvla-7b`

See [optimization/README.md](optimization/README.md) for detailed optimization documentation.
