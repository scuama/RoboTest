# RoboTest

A testing and optimization framework for Vision-Language-Action (VLA) models in robotic manipulation tasks.

## Overview

This repository contains:
- **openvla_rt-1**: Fuzzing and optimization framework for OpenVLA and RT-1 models
- **pi05**: Obstruction experiments and scene replay framework for Pi0.5 model

## Directory Structure

```
RoboTest/
├── openvla_rt-1/          # VLA model testing framework
│   ├── data/              # Test datasets
│   ├── experiments/       # Fuzzing experiments
│   └── optimization/      # Automated optimization
└── pi05/                  # Pi0.5 experiments
    ├── obstruction_exp/   # Obstruction experiments
    └── rerun_exp/         # Scene replay experiments
```

See individual README files in each directory for detailed documentation.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (recommended)
- LIBERO (for pi05)
- OpenVLA/RT-1 models (for openvla_rt-1)

## Quick Start

### OpenVLA/RT-1 Fuzzing
```bash
cd openvla_rt-1/experiments
python run_fuzzer.py --data ../data/t-grasp_n-100_o-0_s-*.json --model openvla-7b
```

### Optimization
```bash
cd openvla_rt-1/optimization
bash start_optimization.sh --task grasp --model openvla-7b
```

### Pi0.5 Obstruction Experiments
```bash
cd pi05/obstruction_exp
python scripts/generate_configs.py
./run_all_pair_experiments.sh
```

## Citation

If you use this code in your research, please cite our paper.

## License

This project is for research purposes only.
