# Environment Setup Guide

This guide provides detailed, step-by-step instructions for setting up the RoboTest environment. It covers the installation of system dependencies, Python packages, and specific simulation frameworks.

## Prerequisites

Before starting, ensure your system meets the following requirements:
- **OS**: Ubuntu 22.04 (Recommended)
- **Python**: Version 3.10 or 3.11
- **CUDA**: Version 11.8+ or 12.0+
- **GPU**: NVIDIA GPU with at least 8GB memory

## 1. Core Installation

Start by setting up the Python environment and installing the base dependencies required for the project.

```bash
# It is recommended to create a new virtual environment
conda create -n robotest python=3.10
conda activate robotest

# Install all shared and framework-specific dependencies
pip install -r requirements.txt
```

## 2. Framework-Specific Setup

Depending on the experiments you intend to run, you may need to set up one or both of the following environments.

### Option A: SimplerEnv / ManiSkill2 Setup (For RT-1 & OpenVLA)

This environment is used for experiments involving **RT-1-X** and **OpenVLA-7b** models. It relies on `SimplerEnv` and `ManiSkill2` (installed via requirements.txt).

#### Model Checkpoints
You need to download the model checkpoints to a local directory.

1.  **RT-1-X (TensorFlow)**:
    -   Location: `gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip`
    -   Description: The specific RT-1-X checkpoint used for evaluation.

2.  **OpenVLA-7b (PyTorch)**:
    -   Location: [OpenVLA Hugging Face / GitHub](https://github.com/openvla/openvla)
    -   Description: Follow the official instructions to download the 7B model weights.

### Option B: OpenPI / LIBERO Setup (For π0)

This environment is typically used for experiments involving policies trained with **OpenPI**, such as the **π0** (pi-zero) related models.

#### Step 1: Install `uv` Package Manager
OpenPI uses `uv` for fast dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Step 2: Install OpenPI
Clone the repository and sync dependencies.

```bash
# Clone recursively to include submodules
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git

# Sync dependencies using uv, skipping LFS download to save bandwidth if not needed immediately
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
cd ..
```

#### Step 3: Install LIBERO
LIBERO is the benchmark environment used for evaluation.

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd ..
```

#### Step 4: Environment Variables
Configure the rendering backend for MuJoCo.

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```
