# Results Folder

This folder contains scripts and data for reproducing the results from the Safety-DOOM paper. It provides functionality to download experimental data from Weights & Biases (wandb), generate figures, and create metric tables.

## Folder Structure

```
results/
├── README.md               # This file
├── commons.py              # Shared utilities and constants
├── data/                   # Downloaded experimental data
│   ├── main/               # Main paper results
│   ├── segment/            # Segmentation experiments
│   ├── depth/              # Depth buffer experiments
│   ├── curriculum/         # Curriculum learning experiments
│   ├── full_actions/       # Full action space experiments
│   ├── cost_scale/         # Cost scaling experiments
│   ├── fps/                # FPS experiments
│   └── safety_bound/       # Safety bound experiments
├── download/               # Scripts for downloading data from wandb
│   └── download.py         # Main download script
├── figures/                # Generated figure outputs
├── metrics/                # Metric data files
├── plotting/               # General plotting utilities
├── plotting_figures/       # Scripts for generating paper figures
│   └── main_results.py     # Main results figure generation
└── plotting_tables/        # Scripts for generating LaTeX tables
    ├── main_results.py     # Main results table generation
    ├── curriculum.py       # Curriculum learning tables
    ├── action_spaces.py    # Action space analysis tables
    ├── cost_scale.py       # Cost scaling tables
    ├── fps.py              # FPS analysis tables
    └── single_env.py       # Single environment tables
```

## Prerequisites

Before using these scripts, ensure you have:

1. **Weights & Biases account**: You need access to the wandb project containing the experimental data
2. **Python dependencies**: Install required packages (matplotlib, numpy, wandb, etc.)
3. **Wandb authentication**: Run `wandb login` to authenticate your account

## Downloading Results from Wandb

### Basic Usage

To download experimental data from wandb, use the `download.py` script:

```bash
cd results/download
python download.py --project "your-wandb-project-name"
```

### Advanced Options

The download script supports various filtering options:

```bash
python download.py \
    --project "your-wandb-project-name" \
    --wandb_tags NIPS \
    --envs armament_burden volcanic_venture \
    --algos PPO PPOCost PPOLag \
    --levels 1 2 3 \
    --seeds 1 2 3 \
    --metrics reward/reward policy_stats/avg_cost \
    --output ../data \
    --overwrite
```

### Key Parameters

- `--project`: Wandb project name (required)
- `--wandb_tags`: Filter runs by wandb tags (NIPS, SEGMENT_BETTER, DEPTH_OBS, CURRICULUM, FULL_ACTIONS)
- `--envs`: Environment names to download
- `--algos`: Algorithm names to download
- `--levels`: Environment difficulty levels (1, 2, 3)
- `--seeds`: Random seeds used in experiments
- `--metrics`: Specific metrics to download
- `--output`: Output directory for downloaded data
- `--overwrite`: Overwrite existing files
- `--hard_constraint`: Download hard constraint results
- `--include_runs`: Include specific runs by name

### Data Organization

Downloaded data is organized as:
```
data/
└── {experiment_type}/     # main, segment, depth, etc.
    └── {environment}/     # armament_burden, volcanic_venture, etc.
        └── {algorithm}/   # PPO, PPOCost, etc.
            └── level_{level}/
                └── seed_{seed}/
                    ├── reward.json
                    ├── cost.json
                    ├── reward_hard.json  # if hard constraints
                    └── cost_hard.json    # if hard constraints
```

## Generating Paper Figures

### Main Results Figure

To generate the main results figure from the paper:

```bash
cd results/plotting_figures
python main_results.py --input ../data/main
```

This creates a 3×4 subplot grid showing learning curves for all 6 environments and 2 metrics (reward and cost).

### Customization Options

```bash
python main_results.py \
    --input ../data/main \
    --level 1 \
    --envs armament_burden volcanic_venture \
    --algos PPO PPOCost PPOLag PPOSaute \
    --metrics reward cost \
    --total_iterations 500000000 \
    --hard_constraint
```

### Key Parameters

- `--input`: Path to data directory
- `--level`: Environment difficulty level to plot
- `--envs`: Environments to include in the figure
- `--algos`: Algorithms to plot
- `--metrics`: Metrics to plot (reward, cost)
- `--total_iterations`: Total training iterations for x-axis scaling
- `--hard_constraint`: Plot hard constraint results
- `--seeds`: Seeds to include in averaging

### Output

Figures are saved as PDF files in the `figures/` directory:
- `level_1.pdf`, `level_2.pdf`, `level_3.pdf` for soft constraints
- `hard.pdf` for hard constraints

## Generating Metric Tables

### Main Results Table

To generate LaTeX tables with final performance metrics:

```bash
cd results/plotting_tables
python main_results.py --input ../data/main
```

This outputs a LaTeX table to stdout that you can copy into your paper.

### Table Features

The generated tables include:
- **Bold formatting** for best performance (highest reward, lowest cost)
- **Green highlighting** for costs meeting safety thresholds
- **Purple highlighting** for best rewards among safe methods
- Confidence intervals calculated from multiple seeds
- Support for both soft and hard constraint results

### Customization Options

```bash
python main_results.py \
    --input ../data/main \
    --levels 1 2 3 \
    --envs armament_burden volcanic_venture \
    --algos PPO PPOCost PPOLag \
    --constraints Soft Hard \
    --n_data_points 10 \
    --seeds 1 2 3
```

### Key Parameters

- `--input`: Path to data directory
- `--levels`: Environment levels to include
- `--envs`: Environments to include
- `--algos`: Algorithms to include
- `--constraints`: Constraint types (Soft, Hard)
- `--n_data_points`: Number of final data points to average
- `--seeds`: Seeds to include in statistics

### Other Table Types

The `plotting_tables/` directory contains scripts for various specialized tables:

- `curriculum.py`: Curriculum learning analysis
- `action_spaces.py`: Action space comparison
- `cost_scale.py`: Cost scaling experiments
- `fps.py`: FPS analysis
- `single_env.py`: Single environment detailed results

## Environment Information

The experiments cover 6 safety-critical environments:

1. **Armament Burden** (AB): Safety threshold = 50
2. **Volcanic Venture** (VV): Safety threshold = 50  
3. **Remedy Rush** (RR): Safety threshold = 5
4. **Collateral Damage** (CD): Safety threshold = 5
5. **Precipice Plunge** (PP): Safety threshold = 50
6. **Detonator's Dilemma** (DD): Safety threshold = 5

Each environment has 3 difficulty levels and experiments are run with multiple random seeds.

## Algorithms

The following safe reinforcement learning algorithms are supported:

- **PPO**: Proximal Policy Optimization (baseline)
- **PPOCost**: PPO with cost in reward function
- **PPOLag**: PPO with Lagrangian constraints
- **PPOSaute**: PPO with Saute safety wrapper
- **PPOPID**: PPO with PID-based safety
- **P3O**: Primal-Dual Policy Optimization
- **TRPO**: Trust Region Policy Optimization
- **TRPOLag**: TRPO with Lagrangian constraints
- **TRPOPID**: TRPO with PID-based safety

## Common Issues and Solutions

### Download Issues

1. **Authentication Error**: Run `wandb login` and enter your API key
2. **Project Not Found**: Verify the project name and your access permissions
3. **Missing Data**: Check if runs have the expected tags and completed successfully

### Plotting Issues

1. **Missing Data**: Ensure data has been downloaded to the correct directory
2. **Empty Plots**: Check that the specified algorithms/environments have data
3. **LaTeX Errors**: Ensure you have the required LaTeX packages for table generation

### Data Issues

1. **Inconsistent Run Lengths**: The scripts handle runs with different lengths by truncating to the minimum length
2. **Missing Seeds**: Scripts skip missing data and continue with available seeds
3. **Special Cases**: PPOCost reward adjustment is handled automatically

## Example Workflow

Here's a complete workflow to reproduce paper results:

```bash
# 1. Download main results data
cd results/download
python download.py --project "your-project" --wandb_tags NIPS --output ../data

# 2. Generate main results figure
cd ../plotting_figures
python main_results.py --input ../data/main --level 1

# 3. Generate results table
cd ../plotting_tables
python main_results.py --input ../data/main > main_results_table.tex

# 4. Generate other experiment figures
cd ../plotting_figures
python main_results.py --input ../data/curriculum --level 1
python main_results.py --input ../data/depth --level 1
```

This will download the data, generate the main results figure, create a LaTeX table, and produce additional experimental figures.