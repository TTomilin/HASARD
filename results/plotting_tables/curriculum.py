import argparse
import json
import os

import numpy as np


TRANSLATIONS = {
    'armament_burden': 'Armament Burden',
    'volcanic_venture': 'Volcanic Venture',
    'remedy_rush': 'Remedy Rush',
    'collateral_damage': 'Collateral Damage',
    'precipice_plunge': 'Precipice Plunge',
    'detonators_dilemma': 'Detonator\'s Dilemma',
    'reward': 'Reward',
    'cost': 'Cost',
    'data/main': 'Regular',
    'data/curriculum': 'Curriculum',
    'diff': 'Difference',
}


# Data Loading
def load_data(base_path, method, environment, seed, level, metric_key):
    file_path = os.path.join(base_path, environment, method, f"level_{level}", f"seed_{seed}", f"{metric_key}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    return None


# Data Processing
def process_data(base_paths, method, environments, seeds, metrics, level, n_data_points):
    results = {}
    data = {base_path: {} for base_path in base_paths}

    for env in environments:
        for base_path in base_paths:
            for metric in metrics:
                processed_data = process_metric(base_path, method, env, seeds, metric, level, n_data_points)
                key = (base_path, env, metric)
                data[key] = processed_data

    # Calculate percentage decreases and store in results
    for env in environments:
        results[env] = {}
        for metric in metrics:
            regular = data[('data/main', env, metric)]['mean']
            curriculum = data[('data/curriculum', env, metric)]['mean']
            if regular and curriculum:
                percent_decrease = -((regular - curriculum) / regular) * 100
                results[env][metric] = {
                    'data/main': regular,
                    'data/curriculum': curriculum,
                    'diff': percent_decrease
                }
    return results


def process_metric(base_path, method, env, seeds, metric, level, n_data_points):
    metric_values = []
    for seed in seeds:
        data = load_data(base_path, method, env, seed, level, metric)
        if data and len(data) >= n_data_points:
            if base_path == 'data/main':
                data = data[:300]
            last_data_points = data[-n_data_points:]
            metric_values.extend(last_data_points)
    mean = np.mean(metric_values)
    ci = 1.96 * np.std(metric_values) / np.sqrt(len(metric_values))
    return {'mean': mean, 'ci': ci}


def generate_latex_table(data, row_headers, caption=''):
    environments = list(data.keys())
    row_headers.append('diff')

    # Prepare headers with corrected approach for backslashes
    headers = []
    for env in environments:
        translated_env = TRANSLATIONS[env].replace(' ', '\\\\ ')
        headers.append(f"\\multicolumn{{2}}{{c}}{{\\makecell{{{translated_env}}}}}")

    # Start the LaTeX table construction
    latex_str = "\\begin{table}[h!]\n\\centering\n\\small{\n\\begin{tabularx}{\\textwidth}{c " + "X@{\\hspace{0.5cm}}X" * len(
        environments) + "}\n"
    latex_str += "\\toprule\n"
    latex_str += "\\multirow{2}{*}{Training} & " + " & ".join(headers) + " \\\\\n"
    subheader_row = "& " + "\\textbf{R $\\uparrow$} & \\textbf{C $\\downarrow$} & " * len(environments)
    latex_str += subheader_row.rstrip(' & ') + "\\\\\n\\midrule\n"

    # Iterate over each constraint type, method, and environment to fill the table
    for j, header in enumerate(row_headers):
        latex_str += f"{TRANSLATIONS[header]} & "
        for env in environments:
            for metric_type in ['reward', 'cost']:
                metric = data[env][metric_type][header]
                percentage_sign = "\%" if header == 'diff' else ''
                latex_str += f"{metric:.2f}{percentage_sign} & "
        latex_str = latex_str.rstrip(' & ') + " \\\\\n"

    latex_str += "\\bottomrule\n\\end{tabularx}\n}"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += "\\end{table}\n"
    return latex_str


def main(args):
    data = process_data(args.inputs, args.method, args.envs, args.seeds, args.metrics, args.level, args.n_data_points)
    table = generate_latex_table(data, args.inputs)
    print(table)


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from RL data.")
    parser.add_argument("--inputs", type=str, nargs='+', default=['data/main', 'data/curriculum'],
                        help="Base input directories containing the data")
    parser.add_argument("--method", type=str, default="PPOPID", choices=["PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O", "TRPO", "TRPOLag", "TRPOPID"],
                        help="Algorithm to analyze")
    parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3], help="Level of the run(s) to compute")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3], help="Seed(s) of the run(s) to compute")
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
