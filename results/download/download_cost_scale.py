import argparse
from download_common import main, create_common_parser, build_cost_scale_folder_path


def common_dl_args() -> argparse.ArgumentParser:
    parser = create_common_parser()
    parser.add_argument("--algos", type=str, nargs='+', default=["PPOCost"],
                        help="Algorithms to download")
    parser.add_argument("--output", type=str, default='data/cost_scale', help="Base output directory to store the data")
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=['COST_SCALING'], help="WandB tags to filter runs")
    return parser


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args(), build_cost_scale_folder_path)
