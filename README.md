# RoboTest

A comprehensive testing and optimization framework for Vision-Language-Action (VLA) models in robotic manipulation tasks.

## Overview

This repository contains the implementation and experimental code for our paper on systematic testing and automated repair of VLA models. The framework evaluates three state-of-the-art models: **OpenVLA**, **RT-1**, and **π0.5**.

## Repository Structure

```
RoboTest/
├── RQ1_Deficiency_Detection/      # Experiment 1: Deficiency Detection
│   ├── openvla_rt1_fuzzing/       # Fuzzing for OpenVLA/RT-1
│   └── pi05_fuzzing/              # Fuzzing for π0.5
│
├── RQ2_Deficiency_Repair/         # Experiment 2: Deficiency Repair
│   ├── openvla_rt1_optimization/  # Optimization for OpenVLA/RT-1
│   └── pi05_optimization/         # Optimization for π0.5
│
└── RQ3_Real_World_Transfer/       # Experiment 3: Real-world Transfer
    ├── scene_configs/             # 100 real-world scene configurations
    └── replay_framework/          # Scene replay framework
```

## Experiments

### RQ1: Deficiency Detection

**Objective**: Evaluate RoboTest's effectiveness in detecting VLA model deficiencies and compare with baseline tool (VLATest).

**Models Tested**: OpenVLA, RT-1, π0.5

**Key Components**:
- **OpenVLA/RT-1 Fuzzing**: MCTS-guided fuzzing with scene variation (camera, lighting, objects)
- **π0.5 Fuzzing**: MCTS-guided fuzzing with obstruction scenarios (27 scene configurations)

**Usage**:
```bash
# OpenVLA/RT-1 fuzzing
cd RQ1_Deficiency_Detection/openvla_rt1_fuzzing
python run_mcts_fuzzer.py --data data/t-grasp_n-100_o-0_s-*.json --model openvla-7b

# π0.5 fuzzing
cd RQ1_Deficiency_Detection/pi05_fuzzing
python run_mcts_fuzzer.py --config configs/exp_single_butter_bowl.yaml
```

**Deficiency Types**:
- **IR** (Invalid Reasoning): Incorrect task understanding
- **IA** (Invalid Action): Wrong manipulation actions
- **OPD** (Object Placement Deficiency): Incorrect object positioning
- **IPU** (Instruction-Plan Unalignment): Misalignment between instruction and execution

### RQ2: Deficiency Repair

**Objective**: Evaluate semantic abstraction-based repair mechanisms (Planning & Action Repair).

**Key Components**:
- **OpenVLA/RT-1 Optimization**: Exhaustive strategy exploration
  - Grasp optimization (8 directions)
  - Object rotation (6 angles)
  - Distance adjustment (3 levels)
  - Object replacement (type-based)
- **π0.5 Optimization**: Guided repair with obstruction handling

**Usage**:
```bash
# OpenVLA/RT-1 optimization
cd RQ2_Deficiency_Repair/openvla_rt1_optimization
bash start_optimization.sh --task grasp --model openvla-7b

# π0.5 optimization
cd RQ2_Deficiency_Repair/pi05_optimization
bash run_optimization.sh
```

**Output**:
- Repaired trajectories
- Memory Set dataset
- Fine-tuned models (RT-1-X-Memory, π0.5-Memory)

### RQ3: Real-world Transfer

**Objective**: Validate sim-to-real transfer performance on Franka robotic arm.

**Key Components**:
- 100 real-world scene configurations
- Comparison: Original π0.5 vs. π0.5-Memory
- Real robot experiments with deficiency annotations

**Usage**:
```bash
cd RQ3_Real_World_Transfer/replay_framework
python rerun_experiments.py --scene 0

# Batch execution
bash rerun_experiments.sh
```

**Scenes**: 100 pre-configured scenes with varying complexity and obstruction patterns.

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (recommended)

### Core Dependencies

```bash
pip install torch numpy opencv-python pyyaml tqdm transforms3d pillow
```

### Model-Specific Dependencies

**For OpenVLA/RT-1**:
- OpenVLA (install from source)
- RT-1 models (from Google Research)
- Simpler Environment

**For π0.5**:
- LIBERO benchmark: `https://github.com/Lifelong-Robot-Learning/LIBERO`
- OpenPI framework
- π0.5 model checkpoint

### Setup

```bash
# Clone repository
git clone https://github.com/scuama/RoboTest.git
cd RoboTest

# Install dependencies
pip install -r requirements.txt

# Set up LIBERO (for π0.5 experiments)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e .
```

## Quick Start

### Run All Experiments

```bash
# RQ1: Deficiency Detection
cd RQ1_Deficiency_Detection/openvla_rt1_fuzzing
python run_fuzzer.py --data data/t-grasp_n-100_o-0_s-*.json --model openvla-7b

# RQ2: Deficiency Repair
cd RQ2_Deficiency_Repair/openvla_rt1_optimization
bash start_optimization.sh --task grasp --model openvla-7b

# RQ3: Real-world Transfer
cd RQ3_Real_World_Transfer/replay_framework
bash rerun_experiments.sh
```

## Results

Each experiment generates results in its respective directory:

- **RQ1**: `RQ1_Deficiency_Detection/results/`
  - Defect detection logs
  - Coverage statistics
  - Comparison with VLATest

- **RQ2**: `RQ2_Deficiency_Repair/results/`
  - Repair success rates
  - Memory Set data
  - Before/after comparison

- **RQ3**: `RQ3_Real_World_Transfer/results/`
  - Real robot execution logs
  - Success rate comparison
  - Video recordings (if enabled)

## Key Features

- **MCTS-Guided Fuzzing**: Adaptive test case generation using Monte Carlo Tree Search
- **Scene Variation Engine**: Camera, lighting, and object position randomization
- **Semantic Repair**: Abstraction-based deficiency repair mechanism
- **Exhaustive Optimization**: Systematic exploration of fix strategies
- **Sim-to-Real Transfer**: Validation on real Franka robotic arm

## File Organization

### RQ1: Deficiency Detection
```
openvla_rt1_fuzzing/
├── run_fuzzer.py              # Basic fuzzing
├── run_mcts_fuzzer.py         # MCTS-guided fuzzing
├── model_interface.py         # VLA model interface
├── variation.py               # Scene variation engine
└── data/                      # Test datasets

pi05_fuzzing/
├── run_mcts_fuzzer.py         # MCTS fuzzing for π0.5
├── scripts/
│   ├── run_experiment.py      # Experiment executor
│   └── scene_modifier.py      # Scene modification
├── configs/                   # 27 obstruction scenarios
└── custom_bddl_files/         # BDDL scene definitions
```

### RQ2: Deficiency Repair
```
openvla_rt1_optimization/
├── optimizer.py               # Main optimizer
├── start_optimization.sh      # Launch script
└── fix_strategy_*.py          # Repair strategies

pi05_optimization/
├── run_optimization.py        # Optimization executor
└── scripts/                   # Supporting scripts
```

### RQ3: Real-world Transfer
```
scene_configs/                 # 100 scene configurations
└── scene_XXXX/
    ├── config_guided.yaml
    ├── mutated_only.bddl
    └── mutated_stacked.bddl

replay_framework/
├── rerun_experiments.py       # Scene replay script
├── deficiency_annotations.json # Defect annotations
└── scripts/                   # Execution utilities
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{robotest2026,
  title={RoboTest: Systematic Testing and Automated Repair of Vision-Language-Action Models},
  author={Your Name et al.},
  booktitle={Proceedings of ISSTA 2026},
  year={2026}
}
```

## License

This project is for research purposes only. See LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

**Paper**: ISSTA 2026 - Robot SLM Testing  
**Last Updated**: January 2026
