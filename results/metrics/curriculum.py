import os
import sys
import numpy as np

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import TRANSLATIONS, load_data, process_metric_data, create_common_parser, generate_latex_table_header, generate_latex_table_footer, check_multiple_paths_data_availability, get_path_translation, create_default_paths

TRANSLATIONS['data/main'] = 'Regular'


# Data Processing
def process_data(base_paths, method, environments, seeds, metrics, level, n_data_points):
    results = {}
    data = {base_path: {} for base_path in base_paths}

    for env in environments:
        for base_path in base_paths:
            for metric in metrics:
                processed_data = process_metric_data(base_path, method, env, seeds, metric, level, n_data_points)
                key = (base_path, env, metric)
                data[key] = processed_data

    # Calculate percentage decreases and store in results
    for env in environments:
        results[env] = {}
        for metric in metrics:
            # Use the first base_path as regular and second as curriculum
            regular_path = base_paths[0]
            curriculum_path = base_paths[1] if len(base_paths) > 1 else base_paths[0]

            regular = data[(regular_path, env, metric)]['mean']
            curriculum = data[(curriculum_path, env, metric)]['mean']
            if regular and curriculum:
                percent_decrease = -((regular - curriculum) / regular) * 100
                results[env][metric] = {
                    regular_path: regular,
                    curriculum_path: curriculum,
                    'diff': percent_decrease
                }
    return results


def generate_latex_table(data, row_headers, caption=''):
    environments = list(data.keys())
    row_headers.append('diff')

    # Use common header function
    latex_str = generate_latex_table_header(environments, "Training")

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

    # Use common footer function
    latex_str += generate_latex_table_footer(caption)
    return latex_str


def main(args):
    # Check if any data is available for the specified paths
    if not check_multiple_paths_data_availability(args.inputs, args.method, args.envs, args.seeds, args.metrics, args.level):
        paths_str = "', '".join(args.inputs)
        print(f"Error: No data found at the specified paths ['{paths_str}']. Please check that at least one path contains data for the specified environments, method, seeds, and metrics.")
        return

    data = process_data(args.inputs, args.method, args.envs, args.seeds, args.metrics, args.level, args.n_data_points)
    table = generate_latex_table(data, args.inputs)
    print(table)


def common_plot_args():
    parser = create_common_parser("Generate a LaTeX table from RL data.")

    # Create default paths dynamically
    default_main, default_curriculum = create_default_paths(__file__, 'main', 'curriculum')

    parser.set_defaults(
        method="PPOPID",
        level=3,
        seeds=[1, 2, 3],
        envs=["remedy_rush", "collateral_damage"],
        inputs=[default_main, default_curriculum]
    )
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
