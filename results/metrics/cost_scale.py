import os
import sys
import numpy as np

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import TRANSLATIONS, load_data, create_common_parser, generate_latex_table_header, generate_latex_table_footer, check_data_availability_with_scales, create_default_paths


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

    # Use common header function
    latex_str = generate_latex_table_header(environments, "Cost Scale")

    # Iterate over each constraint type, method, and environment to fill the table
    for j, scale in enumerate(scales):
        latex_str += f"{scale} & "
        for env in environments:
            for metric_type in ['reward', 'cost']:
                key = f"{metric_type}_scale_{scale}"
                metric = data[env][scale].get(key, {'mean': None, 'ci': None})
                mean = max(0, metric['mean']) if metric['mean'] is not None else None
                mean_str = f"{mean:.2f}" if mean is not None else "N/A"
                latex_str += mean_str + " & "
        latex_str = latex_str.rstrip(' & ') + " \\\\\n"

    # Use common footer function
    latex_str += generate_latex_table_footer(caption)
    return latex_str


def main(args):
    # Check if any data is available for the specified path
    if not check_data_availability_with_scales(args.input, args.method, args.envs, args.seeds, args.metrics, args.scales, 1):
        print(f"Error: No data found at the specified path '{args.input}'. Please check that the path contains data for the specified environments, method, seeds, metrics, and scales.")
        return

    data = process_data(args.input, args.method, args.envs, args.scales, args.seeds, args.metrics, args.n_data_points)
    table = generate_latex_table(data, args.scales)
    print(table)


def common_plot_args():
    parser = create_common_parser("Generate a LaTeX table from RL data.")

    # Create default path dynamically
    default_cost_scale = create_default_paths(__file__, 'cost_scale')

    parser.set_defaults(
        input=default_cost_scale,
        method='PPOCost'
    )
    parser.add_argument("--scales", type=float, nargs='+', default=[0.1, 0.5, 1, 2],
                        help="Cost scales to analyze")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
