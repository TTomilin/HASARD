import argparse
import json
import os

import numpy as np

SAFETY_THRESHOLDS = {
    "armament_burden": 50,
    "volcanic_venture": 50,
    "remedy_rush": 5,
    "collateral_damage": 5,
    "precipice_plunge": 50,
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
def load_data(base_path, environment, method, seed, level, metric_key):
    file_path = os.path.join(base_path, environment, method, f"level_{level}", f"seed_{seed}", f"{metric_key}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    return None


# Data Processing
def process_data(base_path, environments, methods, seeds, levels, metrics, constraints, n_data_points):
    results = {}
    for env in environments:
        results[env] = {}
        for method in methods:
            results[env][method] = {}
            for level in levels:
                for metric in metrics:
                    process_metric(results, base_path, env, method, seeds, level, metric, constraints, n_data_points)
    return results


def process_metric(results, base_path, env, method, seeds, level, metric, constraints, n_data_points):
    for constraint in constraints:
        suffix = '' if constraint == 'Soft' else '_hard'
        metric_key = f"{metric}{suffix}"
        metric_values = []
        cost_data_combined = []

        for seed in seeds:
            # Load the metric data
            data = load_data(base_path, env, method, seed, level, metric_key)

            # Load cost data if the metric is 'reward' and the method is 'PPOCost'
            if metric == 'reward' and method == 'PPOCost':
                cost_key = f"cost{suffix}"
                cost_data = load_data(base_path, env, method, seed, level, cost_key)
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
            results[env][method][f"{metric_key}_level_{level}"] = {
                'mean': np.mean(mean_values),
                'ci': np.mean(ci_values)
            }
        else:
            results[env][method][f"{metric_key}_level_{level}"] = {'mean': None, 'ci': None}


def generate_latex_table(data, constraints, caption=''):
    environments = list(data.keys())
    methods = list(next(iter(data.values())).keys())
    levels = sorted(set(k.split('_')[-1] for k in next(iter(next(iter(data.values())).values())).keys() if k.startswith('reward')))

    # New logic to track rewards meeting cost constraints
    max_reward_meeting_cost_constraint = {}

    for env in environments:
        for level in levels:
            max_reward = float('-inf')
            for method in methods:
                reward_key = f'reward_level_{level}'
                cost_key = f'cost_level_{level}'
                mean_reward = data[env][method][reward_key]['mean']
                mean_cost = data[env][method][cost_key]['mean']
                if mean_cost is not None and mean_cost <= SAFETY_THRESHOLDS[env]:
                    if mean_reward is not None and mean_reward > max_reward:
                        max_reward = mean_reward
                        max_reward_meeting_cost_constraint[f"{env}_{level}"] = (method, mean_reward)

    # Determine max rewards and min costs per environment for bold formatting
    max_rewards = {}
    min_costs = {}
    for env in environments:
        env_data = data[env]
        for level in levels:
            key = f"{env}_level_{level}"
            max_rewards[key] = {}
            min_costs[key] = {}
            for constraint in constraints:
                suffix = '' if constraint == 'Soft' else '_hard'
                all_rewards = [env_data[m][f'reward{suffix}_level_{level}']['mean'] for m in methods if env_data[m][f'reward{suffix}_level_{level}']['mean'] is not None]
                all_costs = [env_data[m][f'cost{suffix}_level_{level}']['mean'] for m in methods if env_data[m][f'cost{suffix}_level_{level}']['mean'] is not None]
                max_rewards[key][suffix] = max(all_rewards, default=None)
                min_costs[key][suffix] = min(all_costs, default=None)

    # Prepare headers with corrected approach for backslashes
    headers = []
    for env in environments:
        translated_env = TRANSLATIONS[env].replace(' ', '\\\\ ')
        headers.append(f"\\multicolumn{{2}}{{c}}{{\\makecell{{{translated_env}}}}}")

    # Start the LaTeX table construction
    latex_str = "\\begin{table}[h!]\n\\centering\n\\small{\n\\begin{tabularx}{\\textwidth}{c l " + "X@{\\hspace{0.5cm}}X" * len(environments) + "}\n"
    latex_str += "\\toprule\n\\rule{0pt}{4ex}\n\\multirow{-2}{*}[0.15em]{\\rotatebox[origin=c]{90}{Level}} & "
    latex_str += "\\multirow{2}{*}{Method} & " + " & ".join(headers) + " \\\\\n"
    subheader_row = "& & " + "\\textbf{R $\\uparrow$} & \\textbf{C $\\downarrow$} & " * len(environments)
    latex_str += subheader_row.rstrip(' & ') + "\\\\\n\\midrule\n"

    # Iterate over each level, constraint type, method, and environment to fill the table
    for k, level in enumerate(levels):
        for j, constraint_type in enumerate(constraints):
            suffix = '_hard' if constraint_type == 'Hard' else ''
            for i, method in enumerate(methods):
                method_escaped = method.replace('_', '\\_')
                # row_lead = "\\multirow{3}{*}{\\rotatebox[origin=c]{45}{Level " + level + " " + constraint_type + "}} & " if i == 0 else "& "
                row_lead = "\\multirow{3}{*}{\\rotatebox[origin=c]{90}{Level " + level + "}} & " if i == 0 else "& "
                latex_str += row_lead + f"{method_escaped} & "
                for env in environments:
                    bold_key = f"{env}_level_{level}"
                    for metric_type in ['reward', 'cost']:
                        key = f"{metric_type}{suffix}_level_{level}"
                        metric = data[env][method].get(key, {'mean': None, 'ci': None})
                        mean = metric['mean']
                        mean_str = f"{mean:.2f}" if mean is not None else "N/A"
                        # Apply bold formatting for max reward and min cost
                        if mean is not None:
                            if (metric_type == 'reward' and mean == max_rewards[bold_key][suffix]) or (metric_type == 'cost' and mean == min_costs[bold_key][suffix]):
                                mean_str = f"\\textbf{{{mean_str}}}"
                        # Apply OliveGreen color to soft cost values if within threshold
                        if metric_type == 'cost' and suffix == '' and mean is not None:
                            threshold = SAFETY_THRESHOLDS[env]
                            if mean <= threshold:
                                mean_str = f"\\textcolor{{OliveGreen}}{{{mean_str}}}"
                        elif metric_type == 'reward' and suffix == '' and mean is not None:
                            if max_reward_meeting_cost_constraint.get(f"{env}_{level}", (None, None))[0] == method:
                                mean_str = f"\\textbf{{\\textcolor{{Fuchsia}}{{{mean_str}}}}}"

                        latex_str += mean_str + " & "
                latex_str = latex_str.rstrip(' & ') + " \\\\\n"
            if (k + 1) * (j + 1) < len(levels) * len(constraints):
                latex_str += "\\midrule\n"

    latex_str += "\\bottomrule\n\\end{tabularx}\n}"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += "\\end{table}\n"
    return latex_str


def main(args):
    data = process_data(args.input, args.envs, args.algos, args.seeds, args.levels, args.metrics, args.constraints, args.n_data_points)
    table = generate_latex_table(data, args.constraints)
    print(table)


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from RL data.")
    parser.add_argument("--input", type=str, default='data/main', help="Base input directory containing the data")
    parser.add_argument("--levels", type=int, nargs='+', default=[1, 2, 3], help="Level(s) of the run(s) to compute")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3], help="Seed(s) of the run(s) to compute")
    parser.add_argument("--n_data_points", type=int, default=10, help="How many final data points to select")
    parser.add_argument("--constraints", type=str, nargs='+', default=["Soft", "Hard"], help="Constraints to analyze")
    parser.add_argument("--algos", type=str, nargs='+', default=["PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O"],
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
