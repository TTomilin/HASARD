import sys
from typing import Tuple

from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.runners.runner_parallel import ParallelRunner
from sample_factory.algo.runners.runner_serial import SerialRunner
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import maybe_load_from_checkpoint, parse_full_cfg, parse_sf_args
from sample_factory.pbt.population_based_training import PopulationBasedTraining
from sample_factory.utils.typing import Config


def make_runner(cfg: Config) -> Tuple[Config, Runner]:
    if cfg.restart_behavior == "resume":
        # if we're resuming from checkpoint, we load all of the config parameters from the checkpoint
        # unless they're explicitly specified in the command line
        cfg = maybe_load_from_checkpoint(cfg)

    if cfg.serial_mode:
        runner_cls = SerialRunner
    else:
        runner_cls = ParallelRunner

    runner = runner_cls(cfg)

    if cfg.with_pbt:
        runner.register_observer(PopulationBasedTraining(cfg, runner))

    return cfg, runner


def run_rl(cfg: Config):
    cfg, runner = make_runner(cfg)

    # here we can register additional message or summary handlers
    # see sf_examples/dmlab/train_dmlab.py for example

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status


def register_vizdoom_components():
    """Register VizDoom environments and models."""
    try:
        import functools
        from sample_factory.algo.utils.context import global_model_factory
        from sample_factory.envs.env_utils import register_env
        from sample_factory.doom.doom_model import make_vizdoom_encoder
        from sample_factory.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec

        # Register VizDoom environments
        for env_spec in DOOM_ENVS:
            make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
            register_env(env_spec.name, make_env_func)

        # Register VizDoom models
        global_model_factory().register_encoder_factory(make_vizdoom_encoder)
    except ImportError:
        # VizDoom components not available, skip registration
        pass


def parse_vizdoom_cfg(argv=None, evaluation=False):
    """Parse configuration with VizDoom-specific parameters."""
    try:
        from sample_factory.doom.doom_params import add_doom_env_args, doom_override_defaults

        parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
        # parameters specific to Doom envs
        add_doom_env_args(parser)
        # override Doom default values for algo parameters
        doom_override_defaults(parser)
        # second parsing pass yields the final configuration
        final_cfg = parse_full_cfg(parser, argv)
        return final_cfg
    except ImportError:
        # VizDoom not available, use standard parsing
        parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
        return parse_full_cfg(parser, argv)


def main():
    """Script entry point."""
    # Try to register VizDoom components if available
    register_vizdoom_components()

    # Check if we should use VizDoom-specific parsing
    try:
        from sample_factory.doom.doom_params import add_doom_env_args
        cfg = parse_vizdoom_cfg()
    except ImportError:
        # Fall back to standard parsing
        parser, _ = parse_sf_args()
        cfg = parse_full_cfg(parser)

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
