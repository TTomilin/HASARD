import argparse
import json
import os
import sys
from typing import Dict, List, Any, Optional, Union, Tuple

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
def setup_script_path() -> None:
    """
    Add the parent directory to the Python path for module imports.

    This function ensures that the results.commons module can be imported
    from any script within the project by adding the project root directory
    to sys.path if it's not already present.

    Returns:
        None
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)


def get_path_translation(path: str) -> str:
    """
    Get a human-readable translation for a data path.

    This function converts internal path names to user-friendly display names
    using the TRANSLATIONS dictionary. If no specific translation exists,
    it generates a readable name by formatting the path basename.

    Args:
        path: The path string to translate (e.g., 'data/main', 'armament_burden')

    Returns:
        Human-readable string representation of the path

    Examples:
        >>> get_path_translation('data/main')
        'Default Obs'
        >>> get_path_translation('armament_burden')
        'Armament Burden'
        >>> get_path_translation('unknown_path')
        'Unknown Path'
    """
    # Extract the last directory name from the path
    path_name = os.path.basename(os.path.normpath(path))

    # Check if we have a specific translation for the full path
    if path in TRANSLATIONS:
        return TRANSLATIONS[path]
    # Check if we have a translation for the data/ prefixed version
    elif f'data/{path_name}' in TRANSLATIONS:
        return TRANSLATIONS[f'data/{path_name}']
    else:
        # Generate a readable name by replacing underscores and title-casing
        return path_name.replace('_', ' ').title()


def create_default_paths(script_file: str, *subdirs: str) -> Union[str, List[str]]:
    """
    Create default data paths relative to the calling script's directory.

    This function constructs paths to data subdirectories relative to the
    script's location, assuming a standard project structure where data
    is stored in a 'data' directory at the same level as the script's parent.

    Args:
        script_file: The __file__ variable from the calling script
        *subdirs: Variable number of subdirectory names to create paths for

    Returns:
        If one subdirectory is provided, returns a single path string.
        If multiple subdirectories are provided, returns a list of path strings.

    Examples:
        >>> create_default_paths(__file__, 'main')
        '/path/to/project/data/main'
        >>> create_default_paths(__file__, 'main', 'full_actions')
        ['/path/to/project/data/main', '/path/to/project/data/full_actions']
    """
    script_dir = os.path.dirname(os.path.abspath(script_file))
    default_data_dir = os.path.join(script_dir, '..', 'data')

    # Create paths for all requested subdirectories
    paths = [os.path.join(default_data_dir, subdir) for subdir in subdirs]

    # Return single path if only one subdir, otherwise return list
    return paths[0] if len(paths) == 1 else paths


# Data Loading Functions
def load_data(base_path: str, method: str, environment: str, seed: int, level: int, 
              metric_key: str, ext_name: str = '', ext_value: str = '') -> Optional[List[float]]:
    """
    Load experimental data from a JSON file for a specific configuration.

    This function loads metric data (e.g., reward, cost) from the standardized
    directory structure used for storing experimental results. The data is
    organized by environment, method, level, and seed.

    Args:
        base_path: Root directory containing the experimental data
        method: Algorithm/method name (e.g., 'PPOLag', 'SAC')
        environment: Environment name (e.g., 'armament_burden')
        seed: Random seed number for the experiment
        level: Environment difficulty level
        metric_key: Name of the metric to load (e.g., 'reward', 'cost')
        ext_name: Optional extension name for specialized experiments
        ext_value: Optional extension value for specialized experiments

    Returns:
        List of metric values if file exists and loads successfully,
        None if file doesn't exist or loading fails

    File Structure:
        base_path/environment/method/level_X/[ext_name_ext_value/]seed_Y/metric_key.json

    Examples:
        >>> load_data('data/main', 'PPOLag', 'armament_burden', 1, 1, 'reward')
        [0.1, 0.2, 0.3, ...]  # List of reward values
        >>> load_data('data/main', 'PPOLag', 'armament_burden', 1, 1, 'cost', 'scale', '10')
        [5.0, 4.8, 4.2, ...]  # List of cost values with scale extension
    """
    # Construct the level directory name
    level_key = f"level_{level}"
    if ext_name or ext_value:
        level_key += f"/{ext_name}_{ext_value}"

    # Build the complete file path
    file_path = os.path.join(base_path, environment, method, level_key, f"seed_{seed}", f"{metric_key}.json")

    # Attempt to load the data
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading file {file_path}: {e}")
            return None
    else:
        print(f"File not found: {file_path}")
        return None


def load_full_data(base_path: str, environments: List[str], methods: List[str], 
                   seeds: List[int], metrics: List[str], level: int, 
                   hard_constraint: bool) -> Dict[Tuple[str, str, str], List[float]]:
    """
    Load experimental data for multiple environments, methods, seeds, and metrics.

    This function aggregates data across multiple experimental configurations,
    combining results from different seeds for each environment-method-metric
    combination. Supports both regular and hard constraint versions of metrics.

    Args:
        base_path: Root directory containing the experimental data
        environments: List of environment names to load data for
        methods: List of algorithm/method names to load data for
        seeds: List of random seeds to aggregate data from
        metrics: List of metric names to load (e.g., ['reward', 'cost'])
        level: Environment difficulty level
        hard_constraint: If True, loads metrics with '_hard' suffix

    Returns:
        Dictionary mapping (environment, method, metric) tuples to lists of
        aggregated metric values across all seeds

    Examples:
        >>> data = load_full_data('data/main', ['armament_burden'], ['PPOLag'], 
        ...                       [1, 2, 3], ['reward', 'cost'], 1, False)
        >>> data[('armament_burden', 'PPOLag', 'reward')]
        [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, ...]  # Combined data from all seeds
    """
    data = {}

    # Iterate through all combinations of experimental parameters
    for env in environments:
        for method in methods:
            for seed in seeds:
                for metric in metrics:
                    key = (env, method, metric)
                    if key not in data:
                        data[key] = []

                    # Use hard constraint version of metric if requested
                    metric_name = f"{metric}_hard" if hard_constraint else metric
                    exp_data = load_data(base_path, method, env, seed, level, metric_name)

                    # Aggregate data from this seed if it exists
                    if exp_data:
                        data[key].extend(exp_data)

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


def process_metric_data(base_path: str, method: str, env: str, seeds: List[int], 
                        metric: str, level: int, n_data_points: int, 
                        ext_name: str = '', ext_value: str = '') -> Dict[str, Optional[float]]:
    """
    Process metric data across multiple seeds and compute statistics.

    This function loads metric data for multiple seeds, extracts the final
    n_data_points from each seed's data, and computes mean and confidence
    interval statistics across all collected data points.

    Args:
        base_path: Root directory containing the experimental data
        method: Algorithm/method name (e.g., 'PPOLag', 'SAC')
        env: Environment name (e.g., 'armament_burden')
        seeds: List of random seeds to process
        metric: Metric name to extract (e.g., 'reward', 'cost')
        level: Environment difficulty level
        n_data_points: Number of final data points to extract from each seed
        ext_name: Optional extension name for specialized experiments
        ext_value: Optional extension value for specialized experiments

    Returns:
        Dictionary containing:
            - 'mean': Mean value across all seeds (None if no data)
            - 'ci': 95% confidence interval (None if no data)

    Examples:
        >>> stats = process_metric_data('data/main', 'PPOLag', 'armament_burden', 
        ...                            [1, 2, 3], 'reward', 1, 10)
        >>> stats['mean']
        0.85  # Average reward across final 10 points from all seeds
        >>> stats['ci']
        0.05  # 95% confidence interval
    """
    metric_values = []

    # Process data from each seed
    for seed in seeds:
        data = load_data(base_path, method, env, seed, level, metric, ext_name, ext_value)
        if data and len(data) >= n_data_points:
            # Special handling for main data path - limit to first 300 points
            if base_path == 'data/main':
                data = data[:300]

            # Extract the final n_data_points from this seed
            last_data_points = data[-n_data_points:]
            metric_values.extend(last_data_points)

    # Return None values if no data was collected
    if not metric_values:
        return {'mean': None, 'ci': None}

    # Calculate statistics across all collected data points
    mean = np.mean(metric_values)
    ci = 1.96 * np.std(metric_values) / np.sqrt(len(metric_values))  # 95% CI
    return {'mean': mean, 'ci': ci}


def create_common_parser(description: str = "Process RL experiment data") -> argparse.ArgumentParser:
    """
    Create a standardized argument parser with common options for RL experiments.

    This function creates an ArgumentParser with standard command-line arguments
    used across multiple analysis scripts in the project. It includes options for
    data paths, algorithms, environments, seeds, metrics, and other common parameters.

    Args:
        description: Description text for the argument parser

    Returns:
        Configured ArgumentParser with standard options for RL experiment analysis

    Standard Arguments:
        - --input/--inputs: Data directory paths
        - --algo/--method: Algorithm selection (PPO variants, TRPO variants, etc.)
        - --envs: Environment selection (safety-critical environments)
        - --level: Difficulty level (1, 2, or 3)
        - --seeds: Random seeds to process
        - --metrics: Metrics to analyze (reward, cost, etc.)
        - --n_data_points: Number of final data points to use
        - --hard_constraint: Whether to use hard safety constraints

    Examples:
        >>> parser = create_common_parser("Analyze safety performance")
        >>> args = parser.parse_args(['--algo', 'PPOLag', '--envs', 'armament_burden'])
        >>> args.method
        'PPOLag'
    """
    parser = argparse.ArgumentParser(description=description)

    # Data input arguments
    parser.add_argument("--input", type=str, 
                        help="Base input directory containing the experimental data")
    parser.add_argument("--inputs", type=str, nargs='+', 
                        help="Multiple base input directories containing the experimental data")

    # Algorithm selection
    parser.add_argument("--algo", "--method", type=str, dest="method", 
                        choices=["PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O", 
                                "TRPO", "TRPOLag", "TRPOPID"],
                        help="Reinforcement learning algorithm to analyze")

    # Environment configuration
    parser.add_argument("--envs", type=str, nargs='+',
                        default=["armament_burden", "volcanic_venture", "remedy_rush", 
                                "collateral_damage", "precipice_plunge", "detonators_dilemma"],
                        choices=["armament_burden", "volcanic_venture", "remedy_rush", 
                                "collateral_damage", "precipice_plunge", "detonators_dilemma"],
                        help="Safety-critical environments to analyze")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3], 
                        help="Environment difficulty level to process")

    # Experimental parameters
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2], 
                        help="Random seed(s) of the experimental runs to process")
    parser.add_argument("--metrics", type=str, nargs='+', default=['reward', 'cost'], 
                        help="Performance metrics to analyze")
    parser.add_argument("--n_data_points", type=int, default=10, 
                        help="Number of final data points to select for analysis")

    # Safety constraint configuration
    parser.add_argument('--hard_constraint', default=False, action='store_true', 
                        help='Use hard safety constraints instead of soft constraints')

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
