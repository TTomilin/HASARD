import datetime
import os
from os.path import join
from typing import Optional

from gymnasium.spaces import Discrete

from sample_factory.doom.env.action_space import (
    doom_action_space,
    doom_action_space_basic,
    doom_action_space_discretized_no_weap,
    doom_action_space_extended,
    doom_action_space_full_discretized,
    doom_turn_and_attack_only, doom_turn_and_move_only, doom_turn_move_jump_accelerate,
    doom_turn_move_jump_accelerate_attack, doom_turn_and_move_and_look_and_jump,
)
from sample_factory.doom.env.doom_gym import VizdoomEnv
from sample_factory.doom.env.wrappers.additional_input import DoomAdditionalInput
from sample_factory.doom.env.wrappers.cost_penalty import CostPenalty
from sample_factory.doom.env.wrappers.multiplayer_stats import MultiplayerStatsWrapper
from sample_factory.doom.env.wrappers.observation_space import SetResolutionWrapper, resolutions
from sample_factory.doom.env.wrappers.record_video import RecordVideo
from sample_factory.doom.env.wrappers.reward_shaping import (
    REWARD_SHAPING_BATTLE,
    REWARD_SHAPING_DEATHMATCH_V0,
    REWARD_SHAPING_DEATHMATCH_V1,
    DoomRewardShapingWrapper,
    true_objective_frags,
    true_objective_winning_the_game,
)
from sample_factory.doom.env.wrappers.scenario_wrappers.armament_burden_cost_function import ArmamentBurdenCostFunction
from sample_factory.doom.env.wrappers.scenario_wrappers.collateral_damage_cost_function import \
    DoomCollateralDamageCostFunction
from sample_factory.doom.env.wrappers.scenario_wrappers.detonators_dilemma_cost_function import \
    DoomDetonatorsDilemmaCostFunction
from sample_factory.doom.env.wrappers.scenario_wrappers.gathering_reward_shaping import DoomGatheringRewardShaping
from sample_factory.doom.env.wrappers.scenario_wrappers.precipice_plunge_cost_function import \
    PrecipicePlungeCostFunction
from sample_factory.doom.env.wrappers.scenario_wrappers.precipice_plunge_reward_function import \
    PrecipicePlungeRewardFunction
from sample_factory.doom.env.wrappers.scenario_wrappers.remedy_rush_cost_function import RemedyRushCostFunction
from sample_factory.doom.env.wrappers.scenario_wrappers.volcanic_venture_cost_function import \
    VolcanicVentureCostFunction
from sample_factory.envs.env_wrappers import (
    PixelFormatChwWrapper,
    ResizeWrapper,
    RewardScalingWrapper,
    TimeLimitWrapper,
)
from sample_factory.utils.utils import debug_log_every_n, experiment_dir


class DoomSpec:
    def __init__(
            self,
            name,
            env_spec_file,
            action_space,
            reward_scaling=1.0,
            penalty_scaling=1.0,
            default_timeout=-1,
            num_agents=1,
            num_bots=0,
            respawn_delay=0,
            timelimit=4.0,
            extra_wrappers=None,
    ):
        self.name = name
        self.env_spec_file = env_spec_file
        self.action_space = action_space
        self.reward_scaling = reward_scaling
        self.penalty_scaling = penalty_scaling
        self.default_timeout = default_timeout

        # 1 for singleplayer, >1 otherwise
        self.num_agents = num_agents

        self.num_bots = num_bots

        self.respawn_delay = respawn_delay
        self.timelimit = timelimit

        # expect list of tuples (wrapper_cls, wrapper_kwargs)
        self.extra_wrappers = extra_wrappers


ADDITIONAL_INPUT = (DoomAdditionalInput, {})  # health, ammo, etc. as input vector
BATTLE_REWARD_SHAPING = (
    DoomRewardShapingWrapper,
    dict(reward_shaping_scheme=REWARD_SHAPING_BATTLE, true_objective_func=None),
)  # "true" reward None means just the env reward (monster kills)
BOTS_REWARD_SHAPING = (
    DoomRewardShapingWrapper,
    dict(reward_shaping_scheme=REWARD_SHAPING_DEATHMATCH_V0, true_objective_func=true_objective_frags),
)
DEATHMATCH_REWARD_SHAPING = (
    DoomRewardShapingWrapper,
    dict(reward_shaping_scheme=REWARD_SHAPING_DEATHMATCH_V1, true_objective_func=true_objective_winning_the_game),
)


def episode_trigger(episode):
    return not episode % 1000


DOOM_ENVS = [
    DoomSpec(
        "doom_basic",
        "basic.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec(
        "doom_two_colors_easy",
        "two_colors_easy.cfg",
        doom_action_space_basic(),
        extra_wrappers=[(DoomGatheringRewardShaping, {})],  # same as https://arxiv.org/pdf/1904.01806.pdf
    ),
    DoomSpec(
        "doom_two_colors_hard",
        "two_colors_hard.cfg",
        doom_action_space_basic(),
        extra_wrappers=[(DoomGatheringRewardShaping, {})],
    ),
    DoomSpec(
        "doom_dm",
        "cig.cfg",
        doom_action_space(),
        1.0,
        int(1e9),
        num_agents=8,
        extra_wrappers=[ADDITIONAL_INPUT, DEATHMATCH_REWARD_SHAPING],
    ),
    DoomSpec(
        "doom_dwango5",
        "dwango5_dm.cfg",
        doom_action_space(),
        1.0,
        int(1e9),
        num_agents=8,
        extra_wrappers=[ADDITIONAL_INPUT, DEATHMATCH_REWARD_SHAPING],
    ),

    # <==== Safety-DOOM environments ====>
    DoomSpec(
        'collateral_damage',
        'collateral_damage.cfg',
        doom_turn_and_attack_only(),
        penalty_scaling=1.0,
        default_timeout=2100,
        extra_wrappers=[(DoomCollateralDamageCostFunction, {})]
    ),

    DoomSpec(
        'volcanic_venture',
        'volcanic_venture.cfg',
        doom_turn_move_jump_accelerate(),
        penalty_scaling=1.0,
        default_timeout=2100,
        extra_wrappers=[(VolcanicVentureCostFunction, {})]
    ),

    DoomSpec(
        'armament_burden',
        'armament_burden.cfg',
        doom_turn_and_move_only(),
        penalty_scaling=0.1,
        default_timeout=2100,
        extra_wrappers=[(ArmamentBurdenCostFunction, {})]
    ),

    DoomSpec(
        'remedy_rush',
        'remedy_rush.cfg',
        doom_turn_and_move_only(),
        penalty_scaling=0.3,
        default_timeout=2100,
        extra_wrappers=[(RemedyRushCostFunction, {})]
    ),

    DoomSpec(
        'precipice_plunge',
        'precipice_plunge.cfg',
        doom_turn_and_move_and_look_and_jump(),
        penalty_scaling=1.0,
        default_timeout=2100,
        extra_wrappers=[(PrecipicePlungeRewardFunction, {}), (PrecipicePlungeCostFunction, {})]
    ),

    DoomSpec(
        'detonators_dilemma',
        'detonators_dilemma.cfg',
        doom_turn_move_jump_accelerate_attack(),
        penalty_scaling=1.0,
        default_timeout=2100,
        extra_wrappers=[(DoomDetonatorsDilemmaCostFunction, {})]
    ),

    # <==== Environments used in the paper ====>
    # this is for comparison with other frameworks (wall-time test)
    DoomSpec("doom_my_way_home_flat_actions", "my_way_home.cfg", Discrete(1 + 4), 1.0),
    DoomSpec("doom_defend_the_center_flat_actions", "defend_the_center.cfg", Discrete(1 + 3), 1.0),
    # "basic" single-player envs
    DoomSpec("doom_my_way_home", "my_way_home.cfg", doom_action_space_basic(), 1.0),
    DoomSpec("doom_deadly_corridor", "deadly_corridor.cfg", doom_action_space_extended(), 0.01),
    DoomSpec("doom_defend_the_center", "defend_the_center.cfg", doom_turn_and_attack_only(), 1.0),
    DoomSpec("doom_defend_the_line", "defend_the_line.cfg", doom_turn_and_attack_only(), 1.0),
    DoomSpec(
        "doom_health_gathering",
        "health_gathering.cfg",
        Discrete(1 + 4),
        1.0,
        extra_wrappers=[(DoomGatheringRewardShaping, {})],  # same as https://arxiv.org/pdf/1904.01806.pdf
    ),
    DoomSpec(
        "doom_health_gathering_supreme",
        "health_gathering_supreme.cfg",
        Discrete(1 + 4),
        1.0,
        extra_wrappers=[(DoomGatheringRewardShaping, {})],  # same as https://arxiv.org/pdf/1904.01806.pdf
    ),
    # "challenging" single-player envs
    DoomSpec(
        "doom_battle",
        "battle_continuous_turning.cfg",
        doom_action_space_discretized_no_weap(),
        1.0,
        2100,
        extra_wrappers=[ADDITIONAL_INPUT, BATTLE_REWARD_SHAPING],
    ),
    DoomSpec(
        "doom_battle2",
        "battle2_continuous_turning.cfg",
        doom_action_space_discretized_no_weap(),
        1.0,
        2100,
        extra_wrappers=[ADDITIONAL_INPUT, BATTLE_REWARD_SHAPING],
    ),
    # multi-player envs with bots as opponents (still only one agent)
    DoomSpec(
        "doom_duel_bots",
        "ssl2.cfg",
        doom_action_space_full_discretized(with_use=True),
        1.0,
        int(1e9),
        num_agents=1,
        num_bots=1,
        respawn_delay=2,
        extra_wrappers=[ADDITIONAL_INPUT, BOTS_REWARD_SHAPING],
    ),
    DoomSpec(
        "doom_deathmatch_bots",
        "dwango5_dm_continuous_weap.cfg",
        doom_action_space_full_discretized(),
        1.0,
        int(1e9),
        num_agents=1,
        num_bots=7,
        extra_wrappers=[ADDITIONAL_INPUT, BOTS_REWARD_SHAPING],
    ),
    # full multiplayer environments for self-play and PBT experiments
    DoomSpec(
        "doom_duel",
        "ssl2.cfg",
        doom_action_space_full_discretized(with_use=True),
        1.0,
        int(1e9),
        num_agents=2,
        num_bots=0,
        respawn_delay=2,
        extra_wrappers=[ADDITIONAL_INPUT, DEATHMATCH_REWARD_SHAPING],
    ),
    DoomSpec(
        "doom_deathmatch_full",
        "freedm.cfg",
        doom_action_space_full_discretized(with_use=True),
        1.0,
        int(1e9),
        num_agents=4,
        num_bots=4,
        respawn_delay=2,
        extra_wrappers=[ADDITIONAL_INPUT, DEATHMATCH_REWARD_SHAPING],
    ),
    # benchmark environment, this is the same doom_battle that we're using in the paper, but without extra input spaces
    # for measurements, and with a more simple action space, just so it is easier to use with other codebases
    # we measure throughput with 128x72 input resolution, 4-frameskip and original game resolution of 160x120
    # (no widescreen)
    DoomSpec("doom_benchmark", "battle.cfg", Discrete(1 + 8), 1.0, 2100),
]


def doom_env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise RuntimeError("Unknown Doom env")


# noinspection PyUnusedLocal
def make_doom_env_impl(
        doom_spec,
        cfg=None,
        env_config=None,
        skip_frames=None,
        episode_horizon=None,
        player_id=None,
        num_agents=None,
        max_num_players=None,
        num_bots=0,  # for multi-agent
        custom_resolution=None,
        render: bool = False,
        render_mode: Optional[str] = None,
        **kwargs,
):
    skip_frames = skip_frames if skip_frames is not None else cfg.env_frameskip

    fps = cfg.fps if "fps" in cfg else None
    async_mode = fps == 0

    env = VizdoomEnv(
        doom_spec.action_space,
        doom_spec.env_spec_file,
        constraint=cfg.constraint,
        skip_frames=skip_frames,
        async_mode=async_mode,
        render_mode=render_mode,
        resolution=cfg.resolution,
        seed=cfg.seed,
    )

    record_to = cfg.record_to if "record_to" in cfg else None

    if cfg.record:
        video_folder = os.path.join(experiment_dir(cfg), cfg.video_dir)
        env = RecordVideo(env, video_folder=video_folder, name_prefix='doom', with_wandb=cfg.with_wandb,
                          step_trigger=lambda step: not step % cfg.record_every, video_length=1000,
                          dummy_env=env_config is None)

    env = MultiplayerStatsWrapper(env)

    resolution = cfg.resolution
    if resolution is None:
        resolution = "256x144" if cfg.wide_aspect_ratio else "160x120"

    assert resolution in resolutions
    env = SetResolutionWrapper(env, resolution)  # default (wide aspect ratio)

    h, w, channels = env.observation_space.shape
    if w != cfg.res_w or h != cfg.res_h:
        env = ResizeWrapper(env, cfg.res_w, cfg.res_h, grayscale=False)

    debug_log_every_n(50, "Doom resolution: %s, resize resolution: %r", resolution, (cfg.res_w, cfg.res_h))

    # randomly vary episode duration to somewhat decorrelate the experience
    timeout = doom_spec.default_timeout
    if episode_horizon is not None and episode_horizon > 0:
        timeout = episode_horizon
    if timeout > 0:
        env = TimeLimitWrapper(env, limit=timeout, random_variation_steps=0)

    pixel_format = cfg.pixel_format if "pixel_format" in cfg else "HWC"
    if pixel_format == "CHW":
        env = PixelFormatChwWrapper(env)

    if doom_spec.extra_wrappers is not None:
        for wrapper_cls, wrapper_kwargs in doom_spec.extra_wrappers:
            env = wrapper_cls(env, **wrapper_kwargs)

    if doom_spec.reward_scaling != 1.0:
        env = RewardScalingWrapper(env, doom_spec.reward_scaling)

    if cfg.algo == 'PPOCost':
        env = CostPenalty(env, doom_spec.penalty_scaling)

    return env


def make_doom_env(env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs):
    spec = doom_env_by_name(env_name)
    return make_doom_env_from_spec(spec, env_name, cfg, env_config, render_mode, **kwargs)


def make_doom_env_from_spec(spec, _env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs):
    """
    Makes a Doom environment from a DoomSpec instance.
    _env_name is unused but we keep it, so functools.partial(make_doom_env_from_spec, env_spec) can registered
    in Sample Factory (first argument in make_env_func is expected to be the env_name).
    """

    if "record_to" in cfg and cfg.record_to:
        tstamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        cfg.record_to = join(cfg.record_to, f"{cfg.experiment}", tstamp)
        if not os.path.isdir(cfg.record_to):
            os.makedirs(cfg.record_to)
    else:
        cfg.record_to = None

    return make_doom_env_impl(spec, cfg=cfg, env_config=env_config, render_mode=render_mode, **kwargs)
