import argparse
from download_common import main, create_common_parser, build_tag_based_folder_path


def common_dl_args() -> argparse.ArgumentParser:
    parser = create_common_parser()
    parser.add_argument("--algos", type=str, nargs='+', default=["PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O", "TRPO", "TRPOLag", "TRPOPID"],
                        help="Algorithms to download/plot")
    parser.add_argument("--output", type=str, default='data', help="Base output directory to store the data")
    parser.add_argument('--hard_constraint', default=False, action='store_true', help='Soft/Hard safety constraint')
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=[], help="WandB tags to filter runs")
    return parser


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args(), build_tag_based_folder_path)
