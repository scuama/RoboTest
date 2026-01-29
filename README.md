# RoboTest

This repository provides the artifacts for the ISSTA 2026 paper (Submission #10):

**RoboTest: A Generic Testing Analysis Approach for Detecting and Repairing Deficiencies in Vision-Language-Action Model**

## Overview

**RoboTest** is a generic testing framework for detecting and repairing deficiencies in Vision-Language-Action (VLA) models through automated evaluation and augmentation.

## Quick Start

### Deficiency Detection

```bash
cd scripts/fuzzing
python run_mcts_vla_fuzzer.py --data ../../data/t-grasp_n-100_o-0_s-170912623.json --model openvla-7b
python run_mcts_fuzzer.py --config ../../configs/fuzzing/exp_single_butter_bowl.yaml
```

### Deficiency Repair

```bash
cd scripts/optimization
bash start_optimization.sh --task grasp --model openvla-7b
bash run_optimization.sh
```

## Environment Setup

Please refer to [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed installation instructions and framework-specific configurations.

```bash
# Quick install for common dependencies
pip install -r requirements.txt
```


