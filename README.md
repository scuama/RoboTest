# RoboTest

This repository provides the artifacts for the ISSTA 2026 paper (Submission #10):

**RoboTest: A Generic Testing Analysis Approach for Detecting and Repairing Deficiencies in Vision-Language-Action Model**

## Overview

**RoboTest** is a generic testing framework for detecting and repairing deficiencies in Vision-Language-Action (VLA) models through automated evaluation and augmentation.

## Installation

### Prerequisites

- Ubuntu 22.04 (recommended)
- Python 3.10 or 3.11
- CUDA 11.8+ or 12.0+
- NVIDIA GPU (8GB+ memory)

### For OpenVLA/RT-1 Experiments

Built on [SimplerEnv](https://github.com/simpler-env/SimplerEnv) and [ManiSkill2_real2sim](https://github.com/simpler-env/ManiSkill2_real2sim).

```bash
# Install dependencies
pip install -r requirements_openvla_rt1.txt

# Download model checkpoints
# RT-1-X: gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip
# OpenVLA-7b: https://github.com/openvla/openvla
```

### For π0.5 Experiments

Built on [OpenPI](https://github.com/Physical-Intelligence/openpi) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install OpenPI
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && cd ..

# Install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..

# Install additional requirements
pip install -r requirements_pi05.txt

# Set environment variables
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Quick Start

### Deficiency Detection (RQ1)

```bash
# OpenVLA/RT-1 fuzzing
cd scripts/fuzzing
python run_mcts_vla_fuzzer.py --data ../../data/t-grasp_n-100_o-0_s-170912623.json --model openvla-7b

# π0.5 fuzzing
python run_mcts_fuzzer.py --config ../../configs/fuzzing/exp_single_butter_bowl.yaml
```

### Deficiency Repair (RQ2)

```bash
# OpenVLA/RT-1 optimization
cd scripts/optimization
bash start_optimization.sh --task grasp --model openvla-7b

# π0.5 optimization
bash run_optimization.sh
```

### Real-World Validation (RQ3)

```bash
# Run validation on physical robot scenes
cd scripts/validation
bash rerun_experiments.sh
```
