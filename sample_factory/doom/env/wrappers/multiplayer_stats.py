import gymnasium as gym

from sample_factory.algo.utils.rl_utils import make_dones


class MultiplayerStatsWrapper(gym.Wrapper):
    """Add to info things like place in the match, gap to leader, kill-death ratio, etc."""

    def __init__(self, env):
        super().__init__(env)
        self.timestep = 0
        self.prev_extra_info = []
        self.num_agents = getattr(env, 'num_agents', None)
        if self.num_agents is None:
            # Try to determine the number of agents from the observation space
            if isinstance(env.observation_space, gym.spaces.Tuple):
                self.num_agents = len(env.observation_space.spaces)
            else:
                self.num_agents = 1  # Default to 1 if unable to determine

        # Initialize per-agent previous extra info
        self.prev_extra_info = [{} for _ in range(self.num_agents)]

    def _parse_info(self, infos, dones):
        new_infos = []
        for idx, (info, done) in enumerate(zip(infos, dones)):
            if (self.timestep % 20 == 0 or done) and "FRAGCOUNT" in info:
                # No need to update these stats every frame
                kdr = info.get("FRAGCOUNT", 0.0) / (info.get("DEATHCOUNT", 0.0) + 1)
                extra_info = {"KDR": float(kdr)}

                player_count = int(info.get("PLAYER_COUNT", self.num_agents))
                player_num = int(info.get("PLAYER_NUMBER", idx))

                # Get fragcounts and player numbers
                fragcounts = {}
                for pi in range(1, player_count + 1):
                    p_num = int(info.get(f"PLAYER{pi}_NUMBER", pi - 1))
                    p_fragcount = int(info.get(f"PLAYER{pi}_FRAGCOUNT", -100000))
                    fragcounts[p_num] = p_fragcount

                # Sort players by fragcount
                sorted_players = sorted(fragcounts.items(), key=lambda x: x[1])

                # Find our agent's place
                for rank, (p_num, fragcount) in enumerate(sorted_players):
                    if p_num == player_num:
                        final_place = player_count - rank  # Invert rank to get place
                        break
                else:
                    final_place = 1  # Default to last place if not found

                extra_info["FINAL_PLACE"] = final_place

                if final_place > 1:
                    leader_fragcount = sorted_players[-1][1]  # Highest fragcount
                    our_fragcount = fragcounts[player_num]
                    extra_info["LEADER_GAP"] = leader_fragcount - our_fragcount
                elif player_count > 1:
                    # We won
                    if player_count >= 2:
                        second_fragcount = sorted_players[-2][1]
                        leader_fragcount = sorted_players[-1][1]
                        extra_info["LEADER_GAP"] = second_fragcount - leader_fragcount  # Should be negative or zero
                        assert extra_info["LEADER_GAP"] <= 0
                    else:
                        extra_info["LEADER_GAP"] = 0
                else:
                    extra_info["LEADER_GAP"] = 0

                self.prev_extra_info[idx] = extra_info
            else:
                extra_info = self.prev_extra_info[idx]

            # Update info with extra_info
            info.update(extra_info)
            new_infos.append(info)
        return new_infos

    def reset(self, **kwargs):
        self.timestep = 0
        obs, infos = self.env.reset(**kwargs)
        num_agents = len(infos) if isinstance(infos, (list, tuple)) else 1
        self.prev_extra_info = [{} for _ in range(num_agents)]
        return obs, infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        if obs is None:
            return obs, rewards, terminated, truncated, infos

        # Ensure terminated and truncated are lists, even if they're single booleans
        if isinstance(terminated, bool):
            terminated = [terminated] * self.num_agents
        if isinstance(truncated, bool):
            truncated = [truncated] * self.num_agents

        dones = [make_dones(t, tr) for t, tr in zip(terminated, truncated)]

        # Ensure infos is a list
        if not isinstance(infos, (list, tuple)):
            infos = [infos] * self.num_agents

        # Parse info for each agent
        infos = self._parse_info(infos, dones)
        self.timestep += 1
        return obs, rewards, terminated, truncated, infos
