import argparse
import json
import os

import numpy as np

SAFETY_THRESHOLDS = {
    "armament_burden": 0.13,
    "volcanic_venture": 50,
    "remedy_rush": 5,
    "collateral_damage": 5,
    "precipice_plunge": 10,
    "detonators_dilemma": 5,
}

TRANSLATIONS = {
    'armament_burden': 'Armament Burden',
    'volcanic_venture': 'Volcanic Venture',
    'remedy_rush': 'Remedy Rush',
    'collateral_damage': 'Collateral Damage',
    'precipice_plunge': 'Precipice Plunge',
    'detonators_dilemma': 'Detonator\'s Dilemma',
    'reward': 'Reward',
    'cost': 'Cost',
}


# Data Loading
def load_data(base_path, environment, method, seed, metric_key):
    file_path = os.path.join(base_path, environment, method, f"seed_{seed}", f"{metric_key}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    return None


# Data Processing
def process_data(base_path, environments, methods, seeds, metrics):
    results = {}
    for env in environments:
        results[env] = {}
        for method in methods:
            results[env][method] = {}
            for metric in metrics:
                process_metric(results, base_path, env, method, seeds, metric)
    return results


def process_metric(results, base_path, env, method, seeds, metric):
    # Process both standard and hard metrics
    for suffix in ['', '_hard']:
        metric_key = f"{metric}{suffix}"
        metric_values = []
        for seed in seeds:
            data = load_data(base_path, env, method, seed, metric_key)
            if data and len(data) >= 10:
                last_10_data = data[-10:]
                mean = np.mean(last_10_data)
                ci = 1.96 * np.std(last_10_data) / np.sqrt(len(last_10_data))
                metric_values.append((mean, ci))
        if metric_values:
            mean_values, ci_values = zip(*metric_values)
            results[env][method][metric_key] = {
                'mean': np.mean(mean_values),
                'ci': np.mean(ci_values)
            }
        else:
            results[env][method][metric_key] = {'mean': None, 'ci': None}

def generate_latex_table(data, caption=''):
    environments = list(data.keys())
    methods = list(next(iter(data.values())).keys())

    # Determine max rewards and min costs per environment for bold formatting
    max_rewards = {}
    min_costs = {}
    for env in environments:
        env_data = data[env]
        max_rewards[env] = {}
        min_costs[env] = {}
        for suffix in ['', '_hard']:
            all_rewards = [env_data[m][f'reward{suffix}']['mean'] for m in methods if env_data[m][f'reward{suffix}']['mean'] is not None]
            all_costs = [env_data[m][f'cost{suffix}']['mean'] for m in methods if env_data[m][f'cost{suffix}']['mean'] is not None]
            max_rewards[env][suffix] = max(all_rewards, default=None)
            min_costs[env][suffix] = min(all_costs, default=None)

    # Prepare headers with corrected approach for backslashes
    headers = []
    for env in environments:
        translated_env = TRANSLATIONS[env].replace(' ', '\\\\ ')
        headers.append(f"\\multicolumn{{2}}{{c}}{{\\makecell{{{translated_env}}}}}")

    # Start the LaTeX table construction
    latex_str = "\\begin{table}[h!]\n\\centering\n\\small{\n\\begin{tabularx}{\\textwidth}{c l " + "X@{\\hspace{0.5cm}}X" * len(environments) + "}\n"
    latex_str += "\\toprule\n\\rule{0pt}{4ex}\n\\multirow{-2}{*}[0.15em]{\\rotatebox[origin=c]{90}{Constraint}} & "
    latex_str += "\\multirow{2}{*}{Method} & " + " & ".join(headers) + " \\\\\n"
    subheader_row = "& & " + "\\textbf{R $\\uparrow$} & \\textbf{C $\\downarrow$} & " * len(environments)
    latex_str += subheader_row.rstrip(' & ') + "\\\\\n\\midrule\n"

    # Iterate over each constraint type, method, and environment to fill the table
    for j, constraint_type in enumerate(['Soft', 'Hard']):
        suffix = '_hard' if constraint_type == 'Hard' else ''
        for i, method in enumerate(methods):
            method_escaped = method.replace('_', '\\_')
            row_lead = "\\multirow{3}{*}{\\rotatebox[origin=c]{90}{" + constraint_type + "}} & " if i == 0 else "& "
            latex_str += row_lead + f"{method_escaped} & "
            for env in environments:
                for metric_type in ['reward', 'cost']:
                    key = f"{metric_type}{suffix}"
                    metric = data[env][method].get(key, {'mean': None, 'ci': None})
                    mean = metric['mean']
                    mean_str = f"{mean:.2f}" if mean is not None else "N/A"
                    # Apply bold formatting for max reward and min cost
                    if mean is not None:
                        if (metric_type == 'reward' and mean == max_rewards[env][suffix]) or (metric_type == 'cost' and mean == min_costs[env][suffix]):
                            mean_str = f"\\textbf{{{mean_str}}}"
                    # Apply OliveGreen color to soft cost values if within threshold
                    if metric_type == 'cost' and suffix == '' and mean is not None:
                        threshold = SAFETY_THRESHOLDS[env] * 1.15  # 15% higher than the safety threshold
                        if mean <= threshold:
                            mean_str = f"\\textcolor{{OliveGreen}}{{{mean_str}}}"
                    latex_str += mean_str + " & "
            latex_str = latex_str.rstrip(' & ') + " \\\\\n"
        if j == 0:
            latex_str += "\\midrule\n"

    latex_str += "\\bottomrule\n\\end{tabularx}\n}"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += "\\end{table}\n"
    return latex_str





def main(args):
    data = process_data(args.input, args.envs, args.algos, args.seeds, args.metrics)
    table = generate_latex_table(data)
    print(table)


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from RL data.")
    parser.add_argument("--input", type=str, default='data', help="Base input directory containing the data")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3], help="Seed(s) of the run(s) to plot")
    parser.add_argument("--algos", type=str, nargs='+', default=["PPO", "PPOCost", "PPOLag"],
                        help="Algorithms to analyze")
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