import argparse
import json
import os
import numpy as np

from results.commons import TRANSLATIONS
from sample_factory.doom.env.doom_utils import DOOM_ENVS


def main(args):
    if len(args.total_iterations) != len(args.inputs):
        raise ValueError("The number of total_iterations must match the number of inputs.")
    data = load_data(args.inputs, args.env, args.algos, args.seeds, args.metrics, args.level)
    compute_and_print_metrics(data, args)

def load_data(base_paths, environment, methods, seeds, metrics, level):
    """Load data from structured directory."""
    data = {}
    for metric in metrics:
        data[metric] = {}
        for method in methods:
            data[metric][method] = {}
            for base_path in base_paths:
                runs = []
                for seed in seeds:
                    file_path = os.path.join(base_path, environment, method, f"level_{level}", f"seed_{seed}", f"{metric}.json")
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as file:
                            runs.append(json.load(file))
                    else:
                        print(f"File not found: {file_path}")
                data[metric][method][base_path] = runs
    return data

def compute_and_print_metrics(data, args):
    doom_env_lookup = {spec.name: spec for spec in DOOM_ENVS}
    total_iterations_dict = dict(zip(args.inputs, args.total_iterations))

    # Compute the minimum total_iterations across inputs
    min_total_iterations = min(total_iterations_dict.values())

    headers = ['Algorithm', 'Input'] + [TRANSLATIONS.get(metric, metric) for metric in args.metrics]
    table_rows = []

    # Initialize results dictionary
    results = {}

    for algo in args.algos:
        for base_path in args.inputs:
            total_iterations = total_iterations_dict[base_path]
            # Initialize the nested dictionary
            if algo not in results:
                results[algo] = {}
            if base_path not in results[algo]:
                results[algo][base_path] = {}
            for metric in args.metrics:
                runs = data.get(metric, {}).get(algo, {}).get(base_path, [])
                if not runs:
                    results[algo][base_path][metric] = 'N/A'
                    continue

                trimmed_runs = []
                if algo == "PPOCost" and metric == "reward":
                    cost_runs = data.get('cost', {}).get(algo, {}).get(base_path, [])
                    if cost_runs:
                        cost_scalar = doom_env_lookup[args.env].penalty_scaling
                        for run, cost_run in zip(runs, cost_runs):
                            num_data_points = len(run)
                            iterations_per_data_point = total_iterations / num_data_points
                            num_points_to_keep = int(min_total_iterations / iterations_per_data_point)
                            num_points_to_keep = min(num_data_points, num_points_to_keep)
                            trimmed_run = np.array(run[:num_points_to_keep])
                            trimmed_cost_run = np.array(cost_run[:num_points_to_keep])
                            adjusted_run = trimmed_run + trimmed_cost_run * cost_scalar
                            trimmed_runs.append(adjusted_run)
                    else:
                        # If cost_runs are missing, proceed without adjustment
                        for run in runs:
                            num_data_points = len(run)
                            iterations_per_data_point = total_iterations / num_data_points
                            num_points_to_keep = int(min_total_iterations / iterations_per_data_point)
                            num_points_to_keep = min(num_data_points, num_points_to_keep)
                            trimmed_run = run[:num_points_to_keep]
                            trimmed_runs.append(trimmed_run)
                else:
                    # For other algorithms or metrics
                    for run in runs:
                        num_data_points = len(run)
                        iterations_per_data_point = total_iterations / num_data_points
                        num_points_to_keep = int(min_total_iterations / iterations_per_data_point)
                        num_points_to_keep = min(num_data_points, num_points_to_keep)
                        trimmed_run = run[:num_points_to_keep]
                        trimmed_runs.append(trimmed_run)

                # Compute the average of the last 10 data points
                averages = []
                for run in trimmed_runs:
                    if len(run) >= 10:
                        last_10 = run[-10:]
                    else:
                        last_10 = run
                    average = np.mean(last_10)
                    averages.append(average)
                overall_average = np.mean(averages)
                results[algo][base_path][metric] = f"{overall_average:.2f}"

    # Build table rows
    for algo in args.algos:
        for base_path in args.inputs:
            row = [algo, TRANSLATIONS.get(base_path, base_path)]
            for metric in args.metrics:
                avg = results.get(algo, {}).get(base_path, {}).get(metric, 'N/A')
                row.append(avg)
            table_rows.append(row)

    # Print the table in markdown format
    print_markdown_table(headers, table_rows)

def print_markdown_table(headers, rows):
    # Determine the width of each column
    col_widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(cell)))

    # Create the header row
    header_row = '| ' + ' | '.join(f"{header.ljust(col_widths[idx])}" for idx, header in enumerate(headers)) + ' |'
    separator_row = '|-' + '-|-'.join('-' * col_widths[idx] for idx in range(len(headers))) + '-|'

    # Create the data rows
    data_rows = []
    for row in rows:
        data_row = '| ' + ' | '.join(f"{str(cell).ljust(col_widths[idx])}" for idx, cell in enumerate(row)) + ' |'
        data_rows.append(data_row)

    # Combine all rows
    table = '\n'.join([header_row, separator_row] + data_rows)
    print(table)

def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute metrics from structured data directory.")
    parser.add_argument("--inputs", type=str, nargs='+', default=['data/main'], help="Base input directories containing the data")
    parser.add_argument("--level", type=int, default=1, help="Level of the run(s) to compute")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3], help="Seed(s) of the run(s) to compute")
    parser.add_argument("--algos", type=str, nargs='+', default=['PPO'],
                        choices=["PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O", "TRPO", "TRPOLag", "TRPOPID"],
                        help="Algorithms to compute")
    parser.add_argument("--env", type=str, default="armament_burden",
                        choices=["armament_burden", "volcanic_venture", "remedy_rush", "collateral_damage",
                                 "precipice_plunge", "detonators_dilemma"],
                        help="Environment to compute")
    parser.add_argument("--metrics", type=str, nargs='+', default=['reward', 'cost'], help="Name of the metrics to compute")
    parser.add_argument("--total_iterations", type=float, nargs='+', default=[5e8, 2e8],
                        help="Total number of environment iterations for each input directory")
    parser.add_argument("--hard_constraint", action='store_true', help="Whether to use hard constraints")
    return parser

if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
