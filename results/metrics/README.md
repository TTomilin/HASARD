# Results/Metrics Scripts Documentation

This directory contains scripts for analyzing and generating metrics from reinforcement learning experiments in the Safety-Doom project. Each script processes experimental data and generates either LaTeX tables or markdown output for research publications and analysis.

## Scripts Overview

### 1. `action_spaces.py`
**Purpose**: Compares the performance between simplified and full action spaces in RL environments.

**Key Functionality**:
- Processes data from two different action space configurations (simplified vs. full)
- Calculates percentage differences in performance metrics
- Generates LaTeX tables comparing reward and cost metrics
- Supports multiple environments and seeds for statistical analysis

**Usage**: Analyzes how action space complexity affects agent performance, helping to understand the trade-offs between action space simplification and performance.

**Default Configuration**: Compares `data/main` (simplified) vs `data/full_actions` (full) using PPOLag algorithm.

---

### 2. `curriculum.py`
**Purpose**: Compares the effectiveness of curriculum learning versus regular training approaches.

**Key Functionality**:
- Processes data from regular training and curriculum learning experiments
- Calculates percentage improvements from curriculum learning
- Generates LaTeX tables with performance comparisons
- Focuses on environments that benefit from curriculum learning (remedy_rush, collateral_damage)

**Usage**: Evaluates whether curriculum learning provides performance benefits over standard training methods.

**Default Configuration**: Compares `data/main` (regular) vs `data/curriculum` using PPOPID algorithm at level 3.

---

### 3. `main_results.py`
**Purpose**: Comprehensive analysis and comparison of multiple RL algorithms across all environments and difficulty levels.

**Key Functionality**:
- Processes data from multiple algorithms (PPO, PPOCost, PPOLag, PPOSaute, PPOPID, P3O)
- Analyzes performance across all 6 environments and 3 difficulty levels
- Handles both soft and hard constraint scenarios
- Generates formatted LaTeX tables with:
  - Bold formatting for best performers
  - Color coding for safety constraint compliance
  - Special highlighting for best rewards meeting cost constraints
- Supports PPOCost's combined reward+cost metric calculation

**Usage**: Primary script for generating comprehensive results tables for research publications.

**Default Configuration**: Analyzes all algorithms across all environments using soft constraints.

---

### 4. `single_env.py`
**Purpose**: Detailed analysis of multiple RL algorithms on a single environment with flexible iteration handling.

**Key Functionality**:
- Computes final performance metrics for multiple algorithms on one environment
- Handles different total iteration counts across input directories
- Normalizes data to ensure fair comparisons
- Outputs results in markdown table format
- Special handling for PPOCost algorithm (combines reward and cost with environment-specific scaling)

**Usage**: Focused analysis when you need detailed metrics for one specific environment, useful for debugging or detailed comparisons.

**Default Configuration**: Analyzes PPO, PPOLag, and PPOPID on armament_burden environment.

---

### 5. `cost_scale.py`
**Purpose**: Analyzes how different cost scaling factors affect algorithm performance.

**Key Functionality**:
- Processes data across multiple cost scaling values (0.1, 0.5, 1, 2)
- Combines reward and cost metrics for comprehensive evaluation
- Generates LaTeX tables showing performance across different scales
- Helps understand sensitivity to cost weighting in safety-constrained RL

**Usage**: Evaluates how sensitive algorithms are to different cost penalty scales, crucial for tuning safety-performance trade-offs.

**Default Configuration**: Analyzes PPOCost algorithm across 4 different cost scales.

---

### 6. `fps.py`
**Purpose**: Analyzes frames-per-second (FPS) and timing performance data from training runs.

**Key Functionality**:
- Loads timing data from CSV files
- Calculates total training duration
- Computes update frequency (updates per second)
- Provides performance metrics for computational efficiency analysis

**Usage**: Evaluates computational performance and training efficiency, useful for understanding resource requirements and optimization.

**Default Configuration**: Analyzes `SafetyPointGoal.csv` from the fps data directory.

## Common Features

All scripts share several common features:
- **Flexible input paths**: Support for custom data directories
- **Seed aggregation**: Statistical analysis across multiple random seeds
- **Environment support**: Work with all 6 Safety-Doom environments
- **Metric flexibility**: Support for reward, cost, and other custom metrics
- **Error handling**: Graceful handling of missing data files
- **Command-line interface**: Full argument parsing for customization

## Data Structure Requirements

Scripts expect data to be organized in the following structure:
```
data/
├── {experiment_type}/
│   ├── {environment}/
│   │   ├── {algorithm}/
│   │   │   ├── level_{level}/
│   │   │   │   ├── seed_{seed}/
│   │   │   │   │   ├── reward.json
│   │   │   │   │   ├── cost.json
│   │   │   │   │   └── ...
```

## Usage Examples

```bash
# Generate main results table
python main_results.py --envs armament_burden volcanic_venture --algos PPO PPOLag

# Compare action spaces
python action_spaces.py --inputs data/main data/full_actions

# Analyze single environment
python single_env.py --env remedy_rush --algos PPO PPOLag PPOPID

# Evaluate cost scaling effects
python cost_scale.py --scales 0.1 0.5 1.0 2.0

# Check curriculum learning benefits
python curriculum.py --envs remedy_rush collateral_damage

# Analyze FPS performance
python fps.py --csv_path data/fps/timing_data.csv
```

Each script includes comprehensive help documentation accessible via the `--help` flag.