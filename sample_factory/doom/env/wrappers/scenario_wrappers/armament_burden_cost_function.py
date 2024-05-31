import gymnasium as gym
import numpy as np
from vizdoom import GameVariable

WEAPON_REWARDS = np.linspace(0.1, 1.0, 7)
WEAPON_WEIGHTS = [0.05, 0.15, 0.3, 0.6, 1.0, 3.0, 6.0]
CARRYING_CAPACITY = 1.0
MIN_SPEED = 0.1
HARD_CONSTRAINT_PENALTY = 10
POSTPONED_PENALTY_MULTIPLIER = 0.1


class ArmamentBurdenCostFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.load = 0
        self.deliveries = 0
        self.total_cost = 0
        self.delivery_cost = 0
        self.num_weapons_carried = 0
        self.total_weapons_acquired = 0
        self.reward_current_delivery = 0
        self.hard_constraint = env.unwrapped.hard_constraint

    def reset(self, **kwargs):
        self.load = 0
        self.deliveries = 0
        self.total_cost = 0
        self.delivery_cost = 0
        self.num_weapons_carried = 0
        self.total_weapons_acquired = 0
        self.reward_current_delivery = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        weapon_id = int(self.game.get_game_variable(GameVariable.USER1))
        num_weapons_carried = int(self.game.get_game_variable(GameVariable.USER2))

        player_speed = 1.0
        excess = 0.0
        cost_this_step = 0.0

        # Check if a weapon has been picked up
        weapon_obtained = num_weapons_carried > self.num_weapons_carried
        if weapon_obtained:
            self.num_weapons_carried += 1  # Number of weapons carried for this delivery
            self.total_weapons_acquired += 1  # Number of weapons picked up throughout the entire episode

            # Calculate the new potential reward to see if it would exceed the carrying capacity
            self.reward_current_delivery += WEAPON_REWARDS[weapon_id]

            # Increase load with the weight of the new weapon
            self.load += WEAPON_WEIGHTS[weapon_id]

        # Check if the total reward from weapons exceeds the carrying capacity and apply the soft constraint penalty
        if self.load > CARRYING_CAPACITY and not self.hard_constraint:
            # Calculate the excess amount over the capacity
            excess = self.load - CARRYING_CAPACITY

            # Calculate the excess percentage over the initial carrying capacity threshold
            excess_percentage = excess / CARRYING_CAPACITY

            # Calculate new speed
            player_speed -= excess_percentage
            player_speed = max(player_speed, MIN_SPEED)

            # The excess weight is proportional to the cost
            # Incur full cost if a weapon was obtained this step, otherwise apply a fraction
            cost_this_step = excess_percentage if weapon_obtained else POSTPONED_PENALTY_MULTIPLIER * excess_percentage
            self.delivery_cost += cost_this_step

        # Determine whether the player has reached the delivery zone
        in_delivery_zone = bool(self.game.get_game_variable(GameVariable.USER3))
        if in_delivery_zone:
            self.load = 0
            self.delivery_cost = 0
            if self.num_weapons_carried > 0:
                self.deliveries += 1  # Increment the number of successful deliveries made
            self.num_weapons_carried = 0
            self.reward_current_delivery = 0

        # Carrying capacity breached for hard constraint
        if self.hard_constraint:
            if num_weapons_carried < self.num_weapons_carried:
                cost_this_step += HARD_CONSTRAINT_PENALTY
                self.delivery_cost += cost_this_step
                self.num_weapons_carried = 0
            speed_reduction = bool(self.game.get_game_variable(GameVariable.USER4))
            if speed_reduction:
                cost_this_step += HARD_CONSTRAINT_PENALTY * POSTPONED_PENALTY_MULTIPLIER
                player_speed = 0.1

        self.total_cost += self.delivery_cost

        info['cost'] = cost_this_step
        info["episode_extra_stats"] = {
            'cost': cost_this_step,
            'delivery_cost': self.delivery_cost,
            'total_cost': self.total_cost,
            'weapons_acquired': self.total_weapons_acquired,
            'deliveries': self.deliveries,
            'player_speed': player_speed,
            'excess_weight': excess,
        }

        return observation, reward, terminated, truncated, info
