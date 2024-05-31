import argparse

import wandb
from wandb.apis.public import Run


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if suitable_run(run, args):
            store_data(run, args)


def store_data(run: Run, args: argparse.Namespace) -> None:
    sequence, metric, data_type, tags = args.sequence, args.metric, args.type, args.wandb_tags
    config = json.loads(run.json_config)
    seq_len = 1 if data_type == 'train' else 4 if sequence in ['CD4', 'CO4'] else 8
    for env_idx in range(seq_len):
        task = SEQUENCES[sequence][env_idx]
        if metric == 'env':
            metric = METRICS[task]
        env = f'run_and_gun-{task}' if sequence in ['CD4', 'CD8', 'CD16'] else f'{task}-{ENVS[sequence]}'
        log_key = f'test/stochastic/{env_idx}/{env}/{metric}' if data_type == 'test' else f'train/{metric}'
        history = list(iter(run.scan_history(keys=[log_key])))

        values = [item[log_key] for item in history]
        method = get_cl_method(run)
        seed = max(run.config["seed"], 1)
        wandb_tags = config['wandb_tags']['value']
        tag = f'{next((tag for tag in tags if tag in wandb_tags), None).lower()}/' if tags and any(tag in tags for tag in SEPARATE_STORAGE_TAGS) else ''
        path = f'{tag}{sequence}/{method}/seed_{seed}'
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created new directory {path}")

        file_name = f'{task}_{metric}.json' if data_type == 'test' else f'train_{metric}.json'
        file_path = f'{path}/{file_name}'
        if args.overwrite or not os.path.exists(file_path):
            print(f'Saving {run.id} --- {path}/{file_name}')
            with open(f'{path}/{file_name}', 'w') as f:
                json.dump(values, f)

def common_dl_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3, 4, 5], help="Seed(s) of the run(s) to plot")
    parser.add_argument("--metric", type=str, default='success', help="Name of the metric to store/plot")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    parser.add_argument("--test_envs", type=int, nargs='+', help="Test environment ID of the actions to download/plot")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--method", type=str, help="Optional filter by CL method")
    parser.add_argument("--type", type=str, default='test', choices=['train', 'test'], help="Type of data to download")
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=[], help="WandB tags to filter runs")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Overwrite existing files")
    parser.add_argument("--include_runs", type=str, nargs="+", default=[],
                        help="List of runs that shouldn't be filtered out")
    return parser

if __name__ == "__main__":
    parser = common_args()
    main(parser.parse_args())
