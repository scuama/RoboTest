# Scene Replay Experiments

Replay saved experimental scenes for Pi0.5 model evaluation.

## Overview

This module contains 100 saved scene configurations with deficiency annotations for systematic evaluation.

## Structure

```
rerun_exp/
├── saved_configs/              # 100 scene configurations
│   └── scene_XXXX/
│       ├── config_guided.yaml  # Guided strategy config
│       ├── mutated_only.bddl   # Scene with target only
│       └── mutated_stacked.bddl # Scene with obstruction
├── rerun_experiments.py        # Main replay script
├── rerun_experiments.sh        # Batch execution
├── deficiency_annotations.json # Deficiency labels
└── requirements.txt
```

## Usage

### Single Scene
```bash
python rerun_experiments.py --scene 0
```

### All Scenes
```bash
bash rerun_experiments.sh
```

### Custom Configuration
```bash
python rerun_experiments.py \
    --scene 0 \
    --config saved_configs/scene_0000/config_guided.yaml
```

## Requirements

```bash
conda activate env_isaaclab
pip install -r requirements.txt
```

Ensure the following are installed:
- LIBERO
- OpenPI
- Pi0.5 checkpoint
