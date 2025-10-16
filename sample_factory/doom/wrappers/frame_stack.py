from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import spaces


class DoomFrameStack(gym.Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        orig = env.observation_space
        if isinstance(orig, spaces.Dict):
            new = {}
            for k, sp in orig.spaces.items():
                if k == "obs" and isinstance(sp, spaces.Box) and len(sp.shape) == 3:
                    C, H, W = sp.shape  # CHW expected
                    new[k] = spaces.Box(
                        low=sp.low.min(), high=sp.high.max(),
                        shape=(C * num_stack, H, W), dtype=sp.dtype
                    )
                else:
                    new[k] = sp
            self.observation_space = spaces.Dict(new)
        elif isinstance(orig, spaces.Box) and len(orig.shape) == 3:
            C, H, W = orig.shape  # CHW expected
            self.observation_space = spaces.Box(
                low=orig.low.min(), high=orig.high.max(),
                shape=(C * num_stack, H, W), dtype=orig.dtype
            )
        else:
            raise ValueError(f"Unsupported observation space: {orig}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = obs["obs"] if isinstance(obs, dict) else obs  # CHW
        for _ in range(self.num_stack):
            self.frames.append(frame)
        return self._stack(obs), info

    def step(self, action):
        obs, r, t, tr, info = self.env.step(action)
        frame = obs["obs"] if isinstance(obs, dict) else obs  # CHW
        self.frames.append(frame)
        return self._stack(obs), r, t, tr, info

    def _stack(self, obs):
        # CHW frames → concat on channel axis
        stacked = np.concatenate(list(self.frames), axis=0)  # (C*k, H, W)
        if isinstance(obs, dict):
            out = obs.copy()
            out["obs"] = stacked
            return out
        return stacked
