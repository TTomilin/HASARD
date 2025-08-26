import argparse

import numpy as np

from results.commons import load_data, TRANSLATIONS

TRANSLATIONS['data/main'] = 'Simplified Actions'


# Data Processing
# Modified process_data function to include percentage decrease calculation
def process_data(base_paths, algo, environments, seeds, metrics, n_data_points):
    results = {}
    action_space_data = {base_path: {} for base_path in base_paths}

    for env in environments:
        for base_path in base_paths:
            for metric in metrics:
                processed_data = process_metric(base_path, algo, env, seeds, metric, n_data_points)
                key = (base_path, env, metric)
                action_space_data[key] = processed_data

    # Calculate percentage decreases and store in results
    for env in environments:
        results[env] = {}
        for metric in metrics:
            simplified = action_space_data[('data/main', env, metric)]['mean']
            full = action_space_data[('data/full_actions', env, metric)]['mean']
            if simplified and full:
                percent_decrease = -((simplified - full) / simplified) * 100
                results[env][metric] = {
                    'data/main': simplified,
                    'data/full_actions': full,
                    'diff': percent_decrease
                }
    return results


def process_metric(action_space, algo, env, seeds, metric, n_data_points):
    level = 1
    metric_values = []

    for seed in seeds:
        data = load_data(action_space, algo, env, seed, level, metric)
        if data and len(data) >= n_data_points:
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
    latex_str += "\\multirow{2}{*}{Action Space} & " + " & ".join(headers) + " \\\\\n"
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
    data = process_data(args.inputs, args.algo, args.envs, args.seeds, args.metrics, args.n_data_points)
    table = generate_latex_table(data, args.inputs)
    print(table)


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from RL data.")
    parser.add_argument("--inputs", type=str, nargs='+', default=['data/main', 'data/full_actions'],
                        help="Base input directories containing the data")
    parser.add_argument("--algo", type=str, default='PPOLag',
                        choices=["PPO", "PPOCost", "PPOLag", "PPOSaute",
                                 "PPOPID", "P3O", "TRPO", "TRPOLag", "TRPOPID"])
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
