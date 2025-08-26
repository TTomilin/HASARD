import argparse
import json
import os

import numpy as np

from results.commons import TRANSLATIONS, load_data


# Data Loading
# def load_data(base_path, algo, environment, scale, seed, level, metric_key):
#     file_path = os.path.join(base_path, environment, "PPOCost", f"level_{level}", f"scale_{scale}", f"seed_{seed}",
#                              f"{metric_key}.json")
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as file:
#             data = json.load(file)
#             return data
#     return None


# Data Processing
def process_data(base_path, algo, environments, scales, seeds, metrics, n_data_points):
    results = {}
    for env in environments:
        results[env] = {}
        for scale in scales:
            results[env][scale] = {}
            for metric in metrics:
                process_metric(results, base_path, algo, env, scale, seeds, metric, n_data_points)
    return results


def process_metric(results, base_path, algo, env, scale, seeds, metric, n_data_points):
    level = 1
    metric_values = []
    cost_data_combined = []

    for seed in seeds:
        # Load the metric data
        data = load_data(base_path, algo, env, seed, level, metric, "scale", scale)

        # Load cost data if the metric is 'reward' and the method is 'PPOCost'
        if metric == 'reward':
            cost_data = load_data(base_path, algo, env, seed, level, "cost", "scale", scale)
            if data and cost_data and len(data) >= n_data_points and len(cost_data) >= n_data_points:
                # Combine reward and cost data if both are sufficiently long
                last_reward_data = data[-n_data_points:]
                last_cost_data = cost_data[-n_data_points:]
                combined_data = [reward + cost for reward, cost in zip(last_reward_data, last_cost_data)]
                cost_data_combined.extend(combined_data)
        elif data and len(data) >= n_data_points:
            last_data_points = data[-n_data_points:]
            cost_data_combined.extend(last_data_points)

    # Calculate statistics for combined data or regular data
    if cost_data_combined:
        mean = np.mean(cost_data_combined)
        ci = 1.96 * np.std(cost_data_combined) / np.sqrt(len(cost_data_combined))
        metric_values.append((mean, ci))

    # Store calculated values
    if metric_values:
        mean_values, ci_values = zip(*metric_values)
        results[env][scale][f"{metric}_scale_{scale}"] = {
            'mean': np.mean(mean_values),
            'ci': np.mean(ci_values)
        }
    else:
        results[env][scale][f"{metric}_scale_{scale}"] = {'mean': None, 'ci': None}


def generate_latex_table(data, scales, caption=''):
    environments = list(data.keys())

    # Prepare headers with corrected approach for backslashes
    headers = []
    for env in environments:
        translated_env = TRANSLATIONS[env].replace(' ', '\\\\ ')
        headers.append(f"\\multicolumn{{2}}{{c}}{{\\makecell{{{translated_env}}}}}")

    # Start the LaTeX table construction
    latex_str = "\\begin{table}[h!]\n\\centering\n\\small{\n\\begin{tabularx}{\\textwidth}{c " + "X@{\\hspace{0.5cm}}X" * len(
        environments) + "}\n"
    latex_str += "\\toprule\n"
    latex_str += "\\multirow{2}{*}{Cost Scale} & " + " & ".join(headers) + " \\\\\n"
    subheader_row = "& " + "\\textbf{R $\\uparrow$} & \\textbf{C $\\downarrow$} & " * len(environments)
    latex_str += subheader_row.rstrip(' & ') + "\\\\\n\\midrule\n"

    # Iterate over each constraint type, method, and environment to fill the table
    for j, scale in enumerate(scales):
        latex_str += f"{scale} & "
        for env in environments:
            for metric_type in ['reward', 'cost']:
                key = f"{metric_type}_scale_{scale}"
                metric = data[env][scale].get(key, {'mean': None, 'ci': None})
                mean = max(0, metric['mean'])
                mean_str = f"{mean:.2f}" if mean is not None else "N/A"
                latex_str += mean_str + " & "
        latex_str = latex_str.rstrip(' & ') + " \\\\\n"

    latex_str += "\\bottomrule\n\\end{tabularx}\n}"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += "\\end{table}\n"
    return latex_str


def main(args):
    data = process_data(args.input, args.algo, args.envs, args.scales, args.seeds, args.metrics, args.n_data_points)
    table = generate_latex_table(data, args.scales)
    print(table)


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from RL data.")
    parser.add_argument("--input", type=str, default='data/cost_scale', help="Base input directory containing the data")
    parser.add_argument("--algo", type=str, default='PPOCost')
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2], help="Seed(s) of the run(s) to compute")
    parser.add_argument("--scales", type=float, nargs='+', default=[0.1, 0.5, 1, 2],
                        help="Seed(s) of the run(s) to compute")
    parser.add_argument("--n_data_points", type=int, default=10, help="How many final data points to select")
    parser.add_argument("--envs", type=str, nargs='+',
                        default=["armament_burden", "volcanic_venture", "remedy_rush", "collateral_damage",
                                 "precipice_plunge", "detonators_dilemma"],
                        help="Environments to analyze")
    parser.add_argument("--metrics", type=str, nargs='+', default=['reward', 'cost'], help="Metrics to aggregate")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
