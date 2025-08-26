import argparse
import os
import sys

import numpy as np

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import load_data, TRANSLATIONS, check_multiple_paths_data_availability, create_common_parser

TRANSLATIONS['data/main'] = 'Simplified Actions'

def get_path_translation(path):
    """Get a human-readable translation for a path."""
    # Extract the last directory name from the path
    path_name = os.path.basename(os.path.normpath(path))

    # Check if we have a specific translation
    if path in TRANSLATIONS:
        return TRANSLATIONS[path]
    elif f'data/{path_name}' in TRANSLATIONS:
        return TRANSLATIONS[f'data/{path_name}']
    else:
        # Generate a readable name from the path
        return path_name.replace('_', ' ').title()


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
            # Use the first base_path as simplified and second as full
            simplified_path = base_paths[0]
            full_path = base_paths[1] if len(base_paths) > 1 else base_paths[0]

            simplified = action_space_data[(simplified_path, env, metric)]['mean']
            full = action_space_data[(full_path, env, metric)]['mean']
            if simplified and full:
                percent_decrease = -((simplified - full) / simplified) * 100
                results[env][metric] = {
                    simplified_path: simplified,
                    full_path: full,
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
        header_translation = TRANSLATIONS.get(header, get_path_translation(header))
        latex_str += f"{header_translation} & "
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
    # Check if any data is available for the specified paths
    if not check_multiple_paths_data_availability(args.inputs, args.method, args.envs, args.seeds, args.metrics, 1):
        paths_str = "', '".join(args.inputs)
        print(f"Error: No data found at the specified paths ['{paths_str}']. Please check that at least one path contains data for the specified environments, algorithm, seeds, and metrics.")
        return

    data = process_data(args.inputs, args.method, args.envs, args.seeds, args.metrics, args.n_data_points)
    table = generate_latex_table(data, args.inputs)
    print(table)


def common_plot_args() -> argparse.ArgumentParser:
    parser = create_common_parser("Generate a LaTeX table from RL data.")

    # Get the script's directory and construct default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, '..', 'data')
    default_main = os.path.join(default_data_dir, 'main')
    default_full_actions = os.path.join(default_data_dir, 'full_actions')

    # Set defaults for common arguments
    parser.set_defaults(
        inputs=[default_main, default_full_actions],
        method='PPOLag',
        seeds=[1, 2, 3],
        envs=["armament_burden", "volcanic_venture", "remedy_rush", "collateral_damage",
              "precipice_plunge", "detonators_dilemma"],
        metrics=['reward', 'cost'],
        n_data_points=10
    )

    # Add specific arguments for this script (--algo is mapped to --method by common parser)
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
