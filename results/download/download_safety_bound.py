import argparse
import json
import os

import wandb
from wandb.apis.public import Run

FORBIDDEN_TAGS = ['TEST']

SAFETY_THRESHOLDS = {
    "armament_burden": 50,
    "volcanic_venture": 50,
    "remedy_rush": 5,
    "collateral_damage": 5,
    "precipice_plunge": 50,
    "detonators_dilemma": 5,
}


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if suitable_run(run, args):
            store_data(run, args)


def suitable_run(run, args: argparse.Namespace) -> bool:
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


def store_data(run: Run, args: argparse.Namespace) -> None:
    metrics = args.metrics
    config = run.config
    run_id = run.id
    level = config['level']
    seed = config['seed']
    env = config['env']
    algo = config['algo']
    bound = config['safety_bound'] if 'safety_bound' in config and config['safety_bound'] else SAFETY_THRESHOLDS[env]

    base_path = args.output  # Directory to store the data

    for metric in metrics:

        # Wrong metric has been stored for Armament Burden
        if env == 'armament_burden' and metric == 'policy_stats/avg_cost':
            metric = 'policy_stats/avg_total_cost'
        elif algo == 'PPOSaute' and metric == 'reward/reward':
            metric = 'policy_stats/avg_episode_reward'

        # Construct folder path for each configuration
        folder_path = os.path.join(base_path, env, algo, f"level_{level}", f"bound_{bound}", f"seed_{seed}")
        os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists

        # Filename based on metric
        metric_name = metric.split('/')[-1].split('_')[-1]
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


def common_dl_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs='+', default=[1, 2, 3], help="Level(s) of the run(s) to download")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3], help="Seed(s) of the run(s) to download")
    parser.add_argument("--algos", type=str, nargs='+', default=["PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O", "TRPO", "TRPOLag", "TRPOPID"],
                        help="Algorithms to download/plot")
    parser.add_argument("--envs", type=str, nargs='+',
                        default=["armament_burden", "volcanic_venture", "remedy_rush",
                                 "collateral_damage", "precipice_plunge", "detonators_dilemma"],
                        help="Environments to download/plot")
    parser.add_argument("--output", type=str, default='data/main', help="Base output directory to store the data")
    parser.add_argument("--metrics", type=str, nargs='+',
                        default=['reward/reward', 'policy_stats/avg_cost'],
                        help="Name of the metrics to download/plot")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument('--hard_constraint', default=False, action='store_true', help='Soft/Hard safety constraint')
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=[], help="WandB tags to filter runs")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Overwrite existing files")
    parser.add_argument("--include_runs", type=str, nargs="+", default=[],
                        help="List of runs that shouldn't be filtered out")
    return parser


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args())
