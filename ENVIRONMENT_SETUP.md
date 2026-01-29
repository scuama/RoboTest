# Environment Setup Guide

This document provides detailed instructions for setting up the specific simulation and framework environments required for RoboTest.

## Install Dependencies

First, ensure you have installed the common Python requirements:

```bash
pip install -r requirements.txt
```

## Detailed Framework Setup

### SimplerEnv / ManiSkill2 Setup

Built on [SimplerEnv](https://github.com/simpler-env/SimplerEnv) and [ManiSkill2_real2sim](https://github.com/simpler-env/ManiSkill2_real2sim).

```bash
# Download model checkpoints (if needed)
# Model A Checkpoint: gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip
# Model B Checkpoint: https://github.com/openvla/openvla
```

### OpenPI / LIBERO Setup

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

# Set environment variables
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```
