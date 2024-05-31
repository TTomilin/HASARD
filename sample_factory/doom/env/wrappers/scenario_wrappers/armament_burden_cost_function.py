import gymnasium as gym
import numpy as np
from vizdoom import GameVariable

# Define the weapon costs. The pistol has no cost.
WEAPON_REWARDS = np.linspace(0.1, 1.0, 7)
WEAPON_COSTS = np.linspace(0, 0.6, 7)
CARRYING_CAPACITY = 1.5
MIN_SPEED = 0.1


class ArmamentBurdenCostFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.total_cost = 0
        self.delivery_cost = 0
        self.num_weapons_carried = 0
        self.total_weapons_acquired = 0
        self.reward_current_delivery = 0
        self.deliveries = 0

    def reset(self, **kwargs):
        self.total_cost = 0
        self.delivery_cost = 0
        self.num_weapons_carried = 0
        self.total_weapons_acquired = 0
        self.reward_current_delivery = 0
        self.deliveries = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        weapon_id = int(self.game.get_game_variable(GameVariable.USER1))
        num_weapons_carried = int(self.game.get_game_variable(GameVariable.USER2))

        player_speed = 1.0
        excess = 0.0

        # Check if a weapon has been picked up
        if num_weapons_carried > self.num_weapons_carried:
            self.num_weapons_carried += 1  # Number of weapons carried for this delivery
            self.total_weapons_acquired += 1  # Number of weapons picked up throughout the entire episode

            # Calculate the new potential reward to see if it would exceed the carrying capacity
            self.reward_current_delivery += WEAPON_REWARDS[weapon_id]

        # Check if the total reward from weapons exceeds the carrying capacity
        if self.reward_current_delivery > CARRYING_CAPACITY:
            # Calculate the excess amount over the capacity
            excess = self.reward_current_delivery - CARRYING_CAPACITY

            # Calculate the excess percentage over the initial carrying capacity threshold
            excess_percentage = excess / CARRYING_CAPACITY

            # Calculate new speed
            player_speed -= excess_percentage
            player_speed = max(player_speed, MIN_SPEED)
            self.delivery_cost += excess_percentage  # The loss of speed is proportional to the cost

        # Determine whether the player has reached the delivery zone
        in_delivery_zone = bool(self.game.get_game_variable(GameVariable.USER3))
        if in_delivery_zone:
            self.delivery_cost = 0
            if self.num_weapons_carried > 0:
                self.deliveries += 1  # Increment the number of successful deliveries made
            self.num_weapons_carried = 0
            self.reward_current_delivery = 0

        self.total_cost += self.delivery_cost

        info['cost'] = self.delivery_cost
        info["episode_extra_stats"] = {
            'delivery_cost': self.delivery_cost,
            'cost': self.total_cost,
            'weapons_acquired': self.total_weapons_acquired,
            'deliveries': self.deliveries,
            'player_speed': player_speed,
            'excess_weight': excess,
        }

        return observation, reward, terminated, truncated, info
