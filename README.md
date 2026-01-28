# RoboTest

A comprehensive testing and optimization framework for Vision-Language-Action (VLA) models in robotic manipulation tasks.

## Overview

This repository contains the implementation and experimental code for our paper on systematic testing and automated repair of VLA models. The framework evaluates three state-of-the-art models: **OpenVLA**, **RT-1**, and **π0.5**.

## Repository Structure

```
RoboTest/
├── RQ1_Deficiency_Detection/      # Section 5.2: Deficiency Detection
│   ├── openvla_rt1_fuzzing/       # Fuzzing for OpenVLA/RT-1
│   └── pi05_fuzzing/              # Fuzzing for π0.5
│
├── RQ2_Deficiency_Repair/         # Section 5.3: Deficiency Repair
│   ├── openvla_rt1_optimization/  # Optimization for OpenVLA/RT-1
│   └── pi05_optimization/         # Optimization for π0.5
│
└── RQ3_Real_World_Validation/     # Section 5.4: Real-world Validation
    ├── scene_configs/             # 80 real-world scene configurations
    └── replay_framework/          # Scene replay framework
```

## Experiments

### RQ1: Deficiency Detection (Section 5.2)

**Objective**: Evaluate RoboTest's effectiveness in detecting VLA deficiencies.

**Setup**: 
- 3 VLA models (OpenVLA, RT-1, π0.5)
- 100 test cases per task × 4 tasks × 5 runs = 6,000 evaluations
- Comparison with baseline (VLATest)

**Usage**:
```bash
# OpenVLA/RT-1 fuzzing
cd RQ1_Deficiency_Detection/openvla_rt1_fuzzing
python run_mcts_vla_fuzzer.py --data data/t-grasp_n-100_o-0_s-*.json --model openvla-7b

# π0.5 fuzzing
cd RQ1_Deficiency_Detection/pi05_fuzzing
python run_mcts_fuzzer.py --config configs/exp_single_butter_bowl.yaml
```

### RQ2: Deficiency Repair (Section 5.3)

**Objective**: Evaluate the effectiveness of deficiency repair mechanisms.

**Setup**:
- Repair models: RT-1-X-Memory, OpenVLA-Memory, π0.5-Memory
- Re-execute all test cases from RQ1 on repaired models

**Usage**:
```bash
# OpenVLA/RT-1 optimization
cd RQ2_Deficiency_Repair/openvla_rt1_optimization
bash start_optimization.sh --task grasp --model openvla-7b

# π0.5 optimization
cd RQ2_Deficiency_Repair/pi05_optimization
bash run_optimization.sh
```

### RQ3: Real-World Validation (Section 5.4)

**Objective**: Validate sim-to-real transfer performance on physical Franka robot.

**Setup**:
- 80 real-world test cases (20% of RQ1 budget)
- Compare π0.5 vs. π0.5-Memory on physical robot

**Usage**:
```bash
cd RQ3_Real_World_Validation/replay_framework
python rerun_experiments.py --scene 0

# Batch execution
bash rerun_experiments.sh
```

## Installation

### Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+ (recommended for GPU acceleration)

### Framework Dependencies

**OpenVLA/RT-1 Experiments** (RQ1 & RQ2 for OpenVLA/RT-1):
- **VLATest Framework**: Our experiments for OpenVLA/RT-1 are built on top of VLATest
- **Simpler-Env**: Simulation environment providing Google Robot and WidowX manipulation tasks
- **SAPIEN**: Physics engine for realistic rendering
- **Model Checkpoints**: RT-1-X, OpenVLA-7b pre-trained weights

**π0.5 Experiments** (RQ1 & RQ2 for π0.5, RQ3 Real-world Validation):
- **LIBERO**: Benchmark suite for lifelong robot learning ([GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO))
- **OpenPI**: Vision-language-action model framework for π0.5
- **Robosuite**: Simulation framework built on MuJoCo
- **MuJoCo**: Physics simulation backend (EGL rendering mode)

### Installation Steps

```bash
# Clone repository
git clone https://github.com/scuama/RoboTest.git
cd RoboTest

# Basic dependencies
pip install torch numpy opencv-python pyyaml tqdm pillow

# For OpenVLA/RT-1 (VLATest-based experiments)
pip install simpler-env gym sapien transforms3d

# For π0.5 (LIBERO-based experiments)
pip install robosuite mujoco
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e .

# Set environment variables for π0.5 experiments
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Quick Start

```bash
# RQ1: Deficiency Detection
cd RQ1_Deficiency_Detection/openvla_rt1_fuzzing
python run_mcts_vla_fuzzer.py --data data/t-grasp_n-100_o-0_s-*.json --model openvla-7b

# RQ2: Deficiency Repair
cd RQ2_Deficiency_Repair/openvla_rt1_optimization
bash start_optimization.sh --task grasp --model openvla-7b

# RQ3: Real-World Validation
cd RQ3_Real_World_Validation/replay_framework
bash rerun_experiments.sh
```
