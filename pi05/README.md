# Pi0.5 Experiments

Obstruction experiments and scene replay framework for testing Pi0.5 model on LIBERO benchmark.

## Structure

```
pi05/
├── obstruction_exp/       # Obstruction experiments (27 scenes)
│   ├── scripts/           # Core scripts
│   ├── configs/           # Experiment configurations
│   ├── custom_bddl_files/ # Scene definitions
│   └── results/           # Experiment results
└── rerun_exp/            # Scene replay experiments (100 scenes)
    ├── scripts/           # Execution scripts
    ├── saved_configs/     # Saved scene configurations
    └── rerun_experiments.py
```

## Obstruction Experiments

Test robot manipulation under occlusion scenarios.

### Experiment Types

- **Pair experiments** (6 scenes): Two identical obstructions stacked
- **Single experiments** (21 scenes): Single obstruction on target object
  - Targets: butter, cheese, pudding (3)
  - Obstructions: plate, bowl, tomato_sauce, ketchup, alphabet_soup, orange_juice, milk (7)

### Usage

```bash
cd obstruction_exp

# Generate configurations
python scripts/generate_configs.py

# Run all experiments
./run_all_pair_experiments.sh
```

## Scene Replay

Replay saved scenes to reproduce results.

```bash
cd rerun_exp
python rerun_experiments.py --scene 0
bash rerun_experiments.sh  # Run all
```

## Requirements

- LIBERO environment
- OpenPI model
- Pi0.5 checkpoint

See [obstruction_exp/README.md](obstruction_exp/README.md) for detailed documentation.
