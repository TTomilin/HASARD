# Plotting Scripts

This directory contains various Python scripts for generating plots and visualizations from experimental data. Each script serves a specific purpose in analyzing and presenting results from safety-constrained reinforcement learning experiments.

## Overview

The scripts are organized into several categories:
- **Main Results**: Scripts for plotting comprehensive results across algorithms and environments
- **Specialized Analysis**: Scripts for specific analyses like cost scaling, curriculum learning, etc.
- **Performance Comparisons**: Scripts comparing different frameworks or configurations
- **Single Environment Analysis**: Scripts focused on individual environment analysis

## Scripts Description

### Main Results Plotting

#### `main_results.py`
Creates comprehensive line plots showing training progress across all environments and algorithms. Displays reward and cost metrics over training steps with confidence intervals and safety thresholds.

**Key Features:**
- Plots multiple algorithms across 6 environments
- Shows training curves with confidence intervals
- Includes safety threshold lines for cost metrics
- Handles PPOCost reward adjustment for fair comparison
- Saves plots to `results/figures/`

**Usage:** `python main_results.py [--input PATH] [--algos ALGO1 ALGO2 ...] [--envs ENV1 ENV2 ...]`

#### `main_results_bar.py`
Generates bar charts showing final performance metrics across algorithms and environments. Provides a summary view of algorithm performance.

**Key Features:**
- Bar charts with error bars
- Compares final performance across algorithms
- Shows both reward and cost metrics
- Includes safety threshold indicators

#### `main_results_bar_minimal.py`
A minimal version of the main results bar chart, focusing on essential comparisons with simplified visualization.

### Single Environment Analysis

#### `single_env.py`
Creates detailed plots for individual environments, allowing focused analysis of algorithm performance on specific tasks.

**Key Features:**
- Single environment focus
- Multiple algorithms comparison
- Customizable metrics selection
- Detailed performance analysis

#### `single_env_gif.py`
Generates animated GIFs showing the evolution of metrics over training time for single environments.

**Key Features:**
- Animated visualization of training progress
- Configurable frame rate and duration
- Optional x-axis shifting
- Saves animations to `results/figures/animated/`

**Usage:** `python single_env_gif.py --env armament_burden --algo PPO --metrics reward cost`

### Specialized Analysis

#### `cost_scale.py`
Analyzes the effect of different cost scaling factors on PPOCost algorithm performance.

**Key Features:**
- Compares multiple cost scaling values (0.1, 0.5, 1, 2)
- Shows how cost scaling affects both reward and cost metrics
- Grid layout showing results across environments
- Specific to PPOCost algorithm

#### `curriculum.py`
Compares regular training versus curriculum learning approaches, specifically for PPOPID algorithm.

**Key Features:**
- Side-by-side comparison of training approaches
- Bar chart visualization
- Focuses on environments that benefit from curriculum learning
- Includes safety threshold indicators

#### `safety_bounds.py`
Analyzes the effect of different safety bounds on algorithm performance, particularly for PPOLag.

**Key Features:**
- Multiple safety bound values comparison
- Shows performance across all environments
- Includes actual safety threshold lines
- Color-coded bound visualization

#### `safety_bounds_minimal.py`
A minimal version of safety bounds analysis, focusing on key environments (armament_burden, detonators_dilemma).

#### `actions_spaces_bar.py`
Compares performance across different action space configurations.

**Key Features:**
- Bar chart comparison of action spaces
- Performance metrics across environments
- Statistical significance indicators

### Performance Comparisons

#### `fps.py`
Compares frames-per-second (FPS) performance between HASARD and Safety-Gymnasium frameworks.

**Key Features:**
- Cumulative frames over time comparison
- Log-scale visualization
- Performance benchmarking
- Reads from CSV data files

**Usage:** `python fps.py --safety_gym_csv PATH --hasard_csv PATH`

#### `policy_updates.py`
Compares the rate of policy updates between HASARD and Safety-Gymnasium frameworks.

**Key Features:**
- Cumulative policy updates over time
- Log-scale visualization
- Parses log files for update timestamps
- Framework performance comparison

### Utility Scripts

#### `partial_plot.py`
Creates selective visualizations by allowing users to specify which algorithms to plot while loading data for all algorithms.

**Key Features:**
- Selective algorithm plotting
- Full data loading with filtered visualization
- Useful for creating focused comparisons
- Customizable algorithm selection

## Common Arguments

Most scripts share common command-line arguments:

- `--input`: Base input directory containing experimental data
- `--level`: Difficulty level to analyze (default: 1)
- `--seeds`: Random seeds to include in analysis (default: [1, 2, 3])
- `--envs`: Environments to analyze
- `--metrics`: Metrics to plot (typically 'reward' and 'cost')
- `--algos`: Algorithms to compare

## Data Structure

Scripts expect data to be organized in the following structure:
```
data/
├── environment_name/
│   ├── algorithm_name/
│   │   ├── level_X/
│   │   │   ├── seed_Y/
│   │   │   │   ├── reward.json
│   │   │   │   └── cost.json
```

## Output

All scripts save their plots to the `results/figures/` directory (not `results/plotting/figures/`). File formats include:
- PDF for publication-quality plots
- PNG for web/presentation use
- GIF for animated visualizations

## Dependencies

- matplotlib
- numpy
- pandas (for some scripts)
- argparse
- json
- Custom modules: `results.commons`, `sample_factory.doom.doom_utils`

## Usage Examples

```bash
# Generate main results plot
python main_results.py --algos PPO PPOLag PPOCost --level 1

# Create animated GIF for single environment
python single_env_gif.py --env volcanic_venture --algo PPOLag --fps 10

# Compare cost scaling effects
python cost_scale.py --scales 0.1 0.5 1.0 2.0

# Analyze curriculum learning
python curriculum.py --envs remedy_rush collateral_damage

# Compare safety bounds
python safety_bounds.py --algo PPOLag --envs armament_burden detonators_dilemma
```

## Notes

- Scripts use seaborn styling for consistent, publication-ready plots
- Confidence intervals are calculated using 95% confidence (1.96 * standard error)
- Safety thresholds are environment-specific and defined in `results.commons`
- PPOCost reward values are adjusted by adding back the cost penalty for fair comparison