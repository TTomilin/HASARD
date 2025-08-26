import argparse
import json
import os
import sys

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
    'diff': 'Difference',
    'data/main': 'Default Obs',
    'data/depth': 'Default Obs + Depth Buffer',
    'data/segment': 'Segmentation',
    'data/curriculum': 'Curriculum',
    'data/full_actions': 'Full Actions',
}

ENV_INITIALS = {
    'armament_burden': 'AB',
    'volcanic_venture': 'VV',
    'remedy_rush': 'RR',
    'collateral_damage': 'CD',
    'precipice_plunge': 'PP',
    'detonators_dilemma': 'DD'
}


# Common Path Handling Functions
def setup_script_path():
    """Add the parent directory to the path so we can import results.commons from any script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)


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


def create_default_paths(script_file, *subdirs):
    """Create default paths relative to the script's directory.

    Args:
        script_file: __file__ from the calling script
        *subdirs: subdirectory names to create paths for

    Returns:
        List of paths or single path if only one subdir provided
    """
    script_dir = os.path.dirname(os.path.abspath(script_file))
    default_data_dir = os.path.join(script_dir, '..', 'data')

    paths = [os.path.join(default_data_dir, subdir) for subdir in subdirs]
    return paths[0] if len(paths) == 1 else paths


# Data Loading
def load_data(base_path, method, environment, seed, level, metric_key, ext_name='', ext_value=''):
    level_key = f"level_{level}"
    if ext_name or ext_value:
        level_key += f"/{ext_name}_{ext_value}"
    file_path = os.path.join(base_path, environment, method, level_key, f"seed_{seed}", f"{metric_key}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    else:
        print(f"File not found: {file_path}")
    return None


def load_full_data(base_path, environments, methods, seeds, metrics, level, hard_constraint):
    data = {}
    for env in environments:
        for method in methods:
            for seed in seeds:
                for metric in metrics:
                    key = (env, method, metric)
                    if key not in data:
                        data[key] = []
                    metric_name = f"{metric}_hard" if hard_constraint else metric
                    exp_data = load_data(base_path, method, env, seed, level, metric_name)
                    if exp_data:
                        data[key].append(exp_data)
    return data


def load_full_data_with_scales(base_path, environments, method, seeds, metrics, scales, level):
    """Load data from structured directory with scale support."""
    data = {}
    for env in environments:
        for seed in seeds:
            for metric in metrics:
                for scale in scales:
                    key = (env, scale, metric)
                    if key not in data:
                        data[key] = []
                    exp_data = load_data(base_path, method, env, seed, level, metric, "scale", scale)
                    if exp_data:
                        data[key].append(exp_data)
    return data


def process_metric_data(base_path, method, env, seeds, metric, level, n_data_points, ext_name='', ext_value=''):
    """Process metric data for multiple seeds and return statistics."""
    metric_values = []
    for seed in seeds:
        data = load_data(base_path, method, env, seed, level, metric, ext_name, ext_value)
        if data and len(data) >= n_data_points:
            # Special handling for main data path
            if base_path == 'data/main':
                data = data[:300]
            last_data_points = data[-n_data_points:]
            metric_values.extend(last_data_points)

    if not metric_values:
        return {'mean': None, 'ci': None}

    mean = np.mean(metric_values)
    ci = 1.96 * np.std(metric_values) / np.sqrt(len(metric_values))
    return {'mean': mean, 'ci': ci}


def create_common_parser(description="Process RL experiment data"):
    """Create a common argument parser with standard options."""
    parser = argparse.ArgumentParser(description=description)

    # Common input/output arguments
    parser.add_argument("--input", type=str, help="Base input directory containing the data")
    parser.add_argument("--inputs", type=str, nargs='+', help="Base input directories containing the data")

    # Algorithm and method arguments
    parser.add_argument("--algo", "--method", type=str, dest="method", 
                        choices=["PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O", "TRPO", "TRPOLag", "TRPOPID"],
                        help="Algorithm to analyze")

    # Environment and level arguments
    parser.add_argument("--envs", type=str, nargs='+',
                        default=["armament_burden", "volcanic_venture", "remedy_rush", "collateral_damage",
                                 "precipice_plunge", "detonators_dilemma"],
                        choices=["armament_burden", "volcanic_venture", "remedy_rush", "collateral_damage",
                                 "precipice_plunge", "detonators_dilemma"],
                        help="Environments to analyze")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3], 
                        help="Level of the run(s) to process")

    # Seed and data arguments
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2], 
                        help="Seed(s) of the run(s) to process")
    parser.add_argument("--metrics", type=str, nargs='+', default=['reward', 'cost'], 
                        help="Metrics to process")
    parser.add_argument("--n_data_points", type=int, default=10, 
                        help="How many final data points to select")

    # Additional common arguments
    parser.add_argument('--hard_constraint', default=False, action='store_true', 
                        help='Soft/Hard safety constraint')

    return parser


def generate_latex_table_header(environments, row_label=""):
    """Generate common LaTeX table header."""
    headers = []
    for env in environments:
        translated_env = TRANSLATIONS[env].replace(' ', '\\\\ ')
        headers.append(f"\\multicolumn{{2}}{{c}}{{\\makecell{{{translated_env}}}}}")

    latex_str = "\\begin{table}[h!]\n\\centering\n\\small{\n\\begin{tabularx}{\\textwidth}{c " + "X@{\\hspace{0.5cm}}X" * len(environments) + "}\n"
    latex_str += "\\toprule\n"
    latex_str += f"\\multirow{{2}}{{*}}{{{row_label}}} & " + " & ".join(headers) + " \\\\\n"
    subheader_row = "& " + "\\textbf{R $\\uparrow$} & \\textbf{C $\\downarrow$} & " * len(environments)
    latex_str += subheader_row.rstrip(' & ') + "\\\\\n\\midrule\n"

    return latex_str


def generate_latex_table_footer(caption=''):
    """Generate common LaTeX table footer."""
    latex_str = "\\bottomrule\n\\end{tabularx}\n}"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += "\\end{table}\n"
    return latex_str


def check_data_availability(base_path, method, environments, seeds, metrics, level, ext_name='', ext_value=''):
    """Check if any data is available for the given parameters."""
    data_found = False
    for env in environments:
        for seed in seeds:
            for metric in metrics:
                data = load_data(base_path, method, env, seed, level, metric, ext_name, ext_value)
                if data is not None:
                    data_found = True
                    break
            if data_found:
                break
        if data_found:
            break
    return data_found


def check_data_availability_with_scales(base_path, method, environments, seeds, metrics, scales, level):
    """Check if any data is available for the given parameters with scales."""
    data_found = False
    for env in environments:
        for seed in seeds:
            for metric in metrics:
                for scale in scales:
                    data = load_data(base_path, method, env, seed, level, metric, "scale", scale)
                    if data is not None:
                        data_found = True
                        break
                if data_found:
                    break
            if data_found:
                break
        if data_found:
            break
    return data_found


def check_multiple_paths_data_availability(base_paths, method, environments, seeds, metrics, level):
    """Check if any data is available across multiple base paths."""
    for base_path in base_paths:
        if check_data_availability(base_path, method, environments, seeds, metrics, level):
            return True
    return False


def check_data_availability_multiple_levels(base_path, method, environments, seeds, metrics, levels):
    """Check if any data is available for the given parameters across multiple levels."""
    for level in levels:
        if check_data_availability(base_path, method, environments, seeds, metrics, level):
            return True
    return False
