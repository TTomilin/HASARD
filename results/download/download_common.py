import argparse
import json
import os
from typing import Dict, List, Optional, Callable

import wandb
from wandb.apis.public import Run

FORBIDDEN_TAGS = ['TEST']
TAG_TO_FOLDER = {
    'NIPS': 'main',
    'SEGMENT_BETTER': 'segment',
    'DEPTH_OBS': 'depth',
    'CURRICULUM': 'curriculum',
    'FULL_ACTIONS': 'full_actions',
}

SAFETY_THRESHOLDS = {
    "armament_burden": 50,
    "volcanic_venture": 50,
    "remedy_rush": 5,
    "collateral_damage": 5,
    "precipice_plunge": 50,
    "detonators_dilemma": 5,
}


def main(args: argparse.Namespace, folder_path_builder: Optional[Callable] = None) -> None:
    """Main function to download data from WandB runs."""
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if suitable_run(run, args):
            store_data(run, args, folder_path_builder)


def suitable_run(run, args: argparse.Namespace) -> bool:
    """Check if a run meets the filtering criteria."""
    try:
        # Check whether the run is in the list of runs to include by exception
        if any(logs in run.name for logs in args.include_runs):
            return True
        # Check whether the provided method corresponds to the run
        config = run.config
        # Check whether the wandb tags are suitable
        if args.wandb_tags:
            if 'wandb_tags' not in config:
                return False
            tags = config['wandb_tags']
            # Check whether the run includes one of the provided tags
            if args.wandb_tags and not any(tag in tags for tag in args.wandb_tags):
                return False
            # Check whether the run includes one of the forbidden tags which is not in the provided tags
            if any(tag in tags for tag in FORBIDDEN_TAGS) and not any(tag in tags for tag in args.wandb_tags):
                return False
        if args.algos and config['algo'] not in args.algos:
            return False
        # Check whether the provided environment corresponds to the run
        if args.envs and config['env'] not in args.envs:
            return False
        # Check whether the run corresponds to one of the provided seeds
        if args.seeds and config['seed'] not in args.seeds:
            return False
        # Check whether the run corresponds to one of the provided levels
        if args.levels and config['level'] not in args.levels:
            return False
        if run.state not in ["finished", "crashed", 'running']:
            return False
        # All filters have been passed
        return True
    except Exception as e:
        print(f"Failed to check suitability for run: {run.id}", e)
        return False


def store_data(run: Run, args: argparse.Namespace, 
               folder_path_builder: Optional[Callable] = None) -> None:
    """Store data from a WandB run with customizable folder structure."""
    metrics = args.metrics
    config = run.config
    run_id = run.id
    level = config['level']
    seed = config['seed']
    env = config['env']
    algo = config['algo']

    # Use custom folder path builder if provided, otherwise use default
    if folder_path_builder:
        folder_path = folder_path_builder(config, args)
    else:
        # Default folder structure (original download.py style)
        tag = config['wandb_tags'][0]
        folder_path = os.path.join(args.output, f"{TAG_TO_FOLDER[tag]}", env, algo, f"level_{level}", f"seed_{seed}")

    os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists

    for metric in metrics:
        # Apply metric corrections
        metric = apply_metric_corrections(env, algo, metric)

        # Filename based on metric
        metric_name = metric.split('/')[-1].split('_')[-1]

        # Handle hard constraint suffix if applicable
        if hasattr(args, 'hard_constraint') and args.hard_constraint:
            file_name = f"{metric_name}_hard.json"
        else:
            file_name = f"{metric_name}.json"

        file_path = os.path.join(folder_path, file_name)

        # If the file already exists and we don't want to overwrite, skip
        if not args.overwrite and os.path.exists(file_path):
            print(f"Skipping: {run_id}: {metric_name}")
            continue

        # Attempt to retrieve and save the data
        try:
            values = run.history(keys=[metric])
            metric_data = values[metric].dropna().tolist()

            # Write the metric data to file
            with open(file_path, 'w') as f:
                json.dump(metric_data, f)
                print(f"Successfully stored: {run_id}: {metric_name}")
        except Exception as e:
            print(f"Error downloading data for run: {run_id}, {metric_name}", e)
            continue


def apply_metric_corrections(env: str, algo: str, metric: str) -> str:
    """Apply metric name corrections based on environment and algorithm."""
    # Wrong metric has been stored for Armament Burden
    if env == 'armament_burden' and metric == 'policy_stats/avg_cost':
        return 'policy_stats/avg_total_cost'
    elif algo == 'PPOSaute' and metric == 'reward/reward':
        return 'policy_stats/avg_episode_reward'
    return metric


def build_cost_scale_folder_path(config: Dict, args: argparse.Namespace) -> str:
    """Build folder path for cost scale experiments."""
    level = config['level']
    seed = config['seed']
    env = config['env']
    algo = config['algo']
    cost_scale = config['penalty_scaling']

    return os.path.join(args.output, env, algo, f"level_{level}", f"scale_{cost_scale}", f"seed_{seed}")


def build_safety_bound_folder_path(config: Dict, args: argparse.Namespace) -> str:
    """Build folder path for safety bound experiments."""
    level = config['level']
    seed = config['seed']
    env = config['env']
    algo = config['algo']
    bound = config['safety_bound'] if 'safety_bound' in config and config['safety_bound'] else SAFETY_THRESHOLDS[env]

    return os.path.join(args.output, env, algo, f"level_{level}", f"bound_{bound}", f"seed_{seed}")


def build_tag_based_folder_path(config: Dict, args: argparse.Namespace) -> str:
    """Build folder path for tag-based experiments (original download.py style)."""
    level = config['level']
    seed = config['seed']
    env = config['env']
    algo = config['algo']
    tag = config['wandb_tags'][0]

    return os.path.join(args.output, f"{TAG_TO_FOLDER[tag]}", env, algo, f"level_{level}", f"seed_{seed}")


def create_common_parser() -> argparse.ArgumentParser:
    """Create base argument parser with common arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs='+', default=[1, 2, 3], help="Level(s) of the run(s) to download")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3], help="Seed(s) of the run(s) to download")
    parser.add_argument("--envs", type=str, nargs='+',
                        default=["armament_burden", "volcanic_venture", "remedy_rush",
                                 "collateral_damage", "precipice_plunge", "detonators_dilemma"],
                        help="Environments to download/plot")
    parser.add_argument("--metrics", type=str, nargs='+',
                        default=['reward/reward', 'policy_stats/avg_cost'],
                        help="Name of the metrics to download/plot")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Overwrite existing files")
    parser.add_argument("--include_runs", type=str, nargs="+", default=[],
                        help="List of runs that shouldn't be filtered out")
    return parser
