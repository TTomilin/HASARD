# Download Scripts

This directory contains scripts for downloading experimental data from Weights & Biases (WandB) projects. The scripts are designed to download metrics from reinforcement learning experiments and organize them into structured directories for further analysis and plotting.

## Overview

There are three main download scripts, each specialized for different types of experiments:

1. **`download.py`** - General purpose downloader for tag-based experiments
2. **`download_cost_scale.py`** - Specialized for cost scaling experiments  
3. **`download_safety_bound.py`** - Specialized for safety bound experiments

All scripts share common functionality from `download_common.py` and connect to WandB to download JSON files containing metric data.

## Prerequisites

- Python environment with required dependencies (wandb, etc.)
- WandB account and API access
- Access to the target WandB project

## Common Arguments

All download scripts share these common arguments:

- `--project` (required): Name of the WandB project to download from
- `--levels`: Level(s) of runs to download (default: [1, 2, 3])
- `--seeds`: Seed(s) of runs to download (default: [1, 2, 3])
- `--envs`: Environments to download (default: all 6 environments)
- `--metrics`: Metrics to download (default: ['reward/reward', 'policy_stats/avg_cost'])
- `--overwrite`: Overwrite existing files (default: False)
- `--include_runs`: List of specific runs to include regardless of filters

## Script-Specific Usage

### 1. General Download (`download.py`)

Downloads experiments based on WandB tags and organizes them using tag-based folder structure.

```bash
python download.py --project "your-wandb-project" [options]
```

**Additional Arguments:**
- `--algos`: Algorithms to download (default: all 9 algorithms)
- `--output`: Base output directory (default: 'data')
- `--hard_constraint`: Enable hard safety constraint mode
- `--wandb_tags`: WandB tags to filter runs (default: [])

**Output Structure:**
```
data/
├── main/
│   ├── armament_burden/
│   │   ├── PPO/
│   │   │   ├── level_1/
│   │   │   │   ├── seed_1/
│   │   │   │   │   ├── reward.json
│   │   │   │   │   └── cost.json
```

### 2. Cost Scale Download (`download_cost_scale.py`)

Downloads cost scaling experiments, typically using PPOCost algorithm with different penalty scaling values.

```bash
python download_cost_scale.py --project "your-wandb-project" [options]
```

**Additional Arguments:**
- `--algos`: Algorithms to download (default: ["PPOCost"])
- `--output`: Base output directory (default: 'data/cost_scale')
- `--wandb_tags`: WandB tags to filter runs (default: ['COST_SCALING'])

**Output Structure:**
```
data/cost_scale/
├── armament_burden/
│   ├── PPOCost/
│   │   ├── level_1/
│   │   │   ├── scale_0.1/
│   │   │   │   ├── seed_1/
│   │   │   │   │   ├── reward.json
│   │   │   │   │   └── cost.json
```

### 3. Safety Bound Download (`download_safety_bound.py`)

Downloads safety bound experiments with different safety threshold values.

```bash
python download_safety_bound.py --project "your-wandb-project" [options]
```

**Additional Arguments:**
- `--algos`: Algorithms to download (default: all 9 algorithms)
- `--output`: Base output directory (default: 'data/main')
- `--hard_constraint`: Enable hard safety constraint mode
- `--wandb_tags`: WandB tags to filter runs (default: [])

**Output Structure:**
```
data/main/
├── armament_burden/
│   ├── PPOLag/
│   │   ├── level_1/
│   │   │   ├── bound_50/
│   │   │   │   ├── seed_1/
│   │   │   │   │   ├── reward.json
│   │   │   │   │   └── cost.json
```

## Supported Environments

- `armament_burden`
- `volcanic_venture`
- `remedy_rush`
- `collateral_damage`
- `precipice_plunge`
- `detonators_dilemma`

## Supported Algorithms

- `PPO`
- `PPOCost`
- `PPOLag`
- `PPOSaute`
- `PPOPID`
- `P3O`
- `TRPO`
- `TRPOLag`
- `TRPOPID`

## Example Usage

### Download all main experiments:
```bash
python download.py --project "safety-doom/experiments" --wandb_tags NIPS
```

### Download cost scaling experiments for specific environments:
```bash
python download_cost_scale.py --project "safety-doom/experiments" --envs armament_burden volcanic_venture
```

### Download safety bound experiments with hard constraints:
```bash
python download_safety_bound.py --project "safety-doom/experiments" --hard_constraint --levels 1 2
```

### Download specific metrics only:
```bash
python download.py --project "safety-doom/experiments" --metrics reward/reward --overwrite
```

## Output Format

All scripts download metrics as JSON files containing lists of values. The file naming convention is:
- `{metric_name}.json` for regular metrics
- `{metric_name}_hard.json` when `--hard_constraint` is used

## Filtering and Organization

The scripts automatically filter runs based on:
- Algorithm type
- Environment
- Seed values
- Level values
- WandB tags
- Run state (finished, crashed, or running)

Forbidden tags (like 'TEST') are automatically excluded unless explicitly included in the filter.

## Troubleshooting

- Ensure you have proper WandB authentication set up
- Check that the project name is correct and accessible
- Verify that the target runs have the expected configuration parameters
- Use `--overwrite` flag to replace existing files
- Check the console output for specific error messages about failed downloads