from enum import Enum

import gymnasium
from gymnasium import Env
from gymnasium.envs.registration import register

LEVELS = [1, 2, 3]
CONSTRAINTS = ["Soft", "Hard"]


class Scenario(Enum):
    ARMAMENT_BURDEN = 'ArmamentBurden'
    DETONATOR_S_DILEMMA = 'DetonatorsDilemma'
    VOLCANIC_VENTURE = 'VolcanicVenture'
    PRECIPICE_PLUNGE = 'PrecipicePlunge'
    COLLATERAL_DAMAGE = 'CollateralDamage'
    REMEDY_RUSH = 'RemedyRush'


def register_environment(scenario, level, constraint):
    env_name = f"{scenario.value}Level{level}{constraint}-v0"
    register(
        id=env_name,
        entry_point=f'hasard.envs.{scenario_enum.name.lower()}.env:{scenario.value}',
        kwargs={'level': level, 'constraint': constraint}
    )


# Loop through each scenario, level, and constraint to register environments
for scenario_enum in Scenario:
    for level in LEVELS:
        for constraint in CONSTRAINTS:
            register_environment(scenario_enum, level, constraint)


def wrap_env(env: Env, **kwargs):
    for wrapper in env.unwrapped.reward_wrappers():
        env = wrapper.wrapper_class(env, **wrapper.kwargs)
    return env


def make(env_id, **kwargs) -> Env:
    return wrap_env(gymnasium.make(env_id, **kwargs.get('doom', {})), **kwargs.get('wrap', {}))
