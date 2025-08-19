from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List

import numpy as np

from sample_factory.algo.logging.interfaces import StatsConfig
from sample_factory.utils.dicts import iterate_recursively


class StatsProcessor:
    """Handles aggregation, windowing, and formatting of stats."""

    def __init__(self, config: StatsConfig, num_policies: int):
        self.config = config
        self.num_policies = num_policies
        
        # Policy-specific averaged stats storage
        self.policy_avg_stats: Dict[str, List[deque]] = {}
        
        # Regular (non-averaged) stats
        self.stats: Dict[str, Any] = {}
        self.avg_stats: Dict[str, deque] = {}

    def process_episodic_stats(self, stats: Dict[str, Any], policy_id: int) -> None:
        """Process episodic stats from a single policy."""
        for _, key, value in iterate_recursively(stats):
            if key not in self.policy_avg_stats:
                max_len = self.config.heatmap_avg if key == 'heatmap' else self.config.stats_avg
                self.policy_avg_stats[key] = [
                    deque(maxlen=max_len) for _ in range(self.num_policies)
                ]

            if isinstance(value, np.ndarray) and value.ndim > 0 and key != 'heatmap':
                if len(value) > self.policy_avg_stats[key][policy_id].maxlen:
                    # Increase maxlen to make sure we never ignore any stats from the environments
                    self.policy_avg_stats[key][policy_id] = deque(maxlen=len(value))

                self.policy_avg_stats[key][policy_id].extend(value)
            else:
                self.policy_avg_stats[key][policy_id].append(value)

    def get_policy_stats_for_logging(self, policy_id: int, env_steps: int, total_train_seconds: float) -> Dict[str, Any]:
        """Get processed stats for a specific policy ready for logging."""
        stats_to_log = {}
        
        for key, stat in self.policy_avg_stats.items():
            if key == 'heatmap':
                continue  # Heatmaps are handled separately
                
            if len(stat[policy_id]) >= stat[policy_id].maxlen or (
                    len(stat[policy_id]) > 10 and total_train_seconds > 300
            ):
                stat_value = np.mean(stat[policy_id])

                # Determine the appropriate tag based on the key
                if "/" in key:
                    # Custom summaries have their own sections in tensorboard
                    avg_tag = key
                    min_tag = f"{key}_min"
                    max_tag = f"{key}_max"
                elif key in ("reward", "len"):
                    # Reward and length get special treatment
                    avg_tag = f"{key}/{key}"
                    min_tag = f"{key}/{key}_min"
                    max_tag = f"{key}/{key}_max"
                else:
                    avg_tag = f"policy_stats/avg_{key}"
                    min_tag = f"policy_stats/avg_{key}_min"
                    max_tag = f"policy_stats/avg_{key}_max"

                stats_to_log[avg_tag] = float(stat_value)

                # For key stats report min/max as well
                if key in ("reward", "cost", "true_objective", "len"):
                    min_val = float(min(stat[policy_id]))
                    max_val = float(max(stat[policy_id]))
                    stats_to_log[min_tag] = min_val
                    stats_to_log[max_tag] = max_val

                # Log basic health and armor stats for all tasks (episodic averages only)
                if key in ("health", "armor"):
                    stats_to_log[f"basic_stats/{key}"] = float(stat_value)

                # Log multi-agent specific stats dynamically
                # These come from the multi-agent stats collection system
                if key.startswith("combined_stats/"):
                    # Extract the actual stat name from the nested key
                    stat_name = key.split("/", 1)[1]
                    # Log all combined stats as team stats (dynamic approach)
                    stats_to_log[f"team_stats/{stat_name}"] = float(stat_value)

                elif key.startswith("individual_stats/"):
                    # Parse individual stats key to extract agent ID and stat name
                    # Expected format: individual_stats/agent_X/stat_name or individual_stats/stat_name
                    key_parts = key.split("/")
                    if len(key_parts) >= 3 and key_parts[1].startswith("agent_"):
                        # Format: individual_stats/agent_X/stat_name
                        agent_id = key_parts[1]  # e.g., "agent_0", "agent_1"
                        stat_name = "/".join(key_parts[2:])  # Handle nested stat names
                        stats_to_log[f"{agent_id}/{stat_name}"] = float(stat_value)
                    else:
                        # Fallback format: individual_stats/stat_name (aggregate across agents)
                        stat_name = "/".join(key_parts[1:])
                        # Categorize stats dynamically based on common patterns
                        if stat_name in ("health", "armor", "reward"):
                            stats_to_log[f"agent_stats/{stat_name}"] = float(stat_value)
                        else:
                            stats_to_log[f"task_stats/{stat_name}"] = float(stat_value)

        return stats_to_log

    def get_heatmap_data(self, policy_id: int = 0) -> np.ndarray:
        """Get heatmap data for logging."""
        if 'heatmap' not in self.policy_avg_stats:
            return None
            
        heatmap_data = self.policy_avg_stats['heatmap']
        if isinstance(heatmap_data, list) and len(heatmap_data) > 0:
            # Get the specified policy's heatmap data
            policy_heatmaps = heatmap_data[policy_id]
            if len(policy_heatmaps) > 0:
                # Convert deque to list and then to numpy array
                heatmap_list = list(policy_heatmaps)
                if len(heatmap_list) > 0:
                    # Take the mean across episodes, but preserve spatial dimensions
                    return np.mean(heatmap_list, axis=0)
        return None

    def get_console_stats(self) -> Dict[str, Any]:
        """Get stats for console logging."""
        console_stats = {}
        
        if "reward" in self.policy_avg_stats:
            policy_reward_stats = []
            for policy_id in range(self.num_policies):
                reward_stats = self.policy_avg_stats["reward"][policy_id]
                if len(reward_stats) > 0:
                    policy_reward_stats.append((policy_id, f"{np.mean(reward_stats):.3f}"))
            console_stats["reward"] = policy_reward_stats

        if "cost" in self.policy_avg_stats:
            policy_cost_stats = []
            for policy_id in range(self.num_policies):
                cost_stats = self.policy_avg_stats["cost"][policy_id]
                if len(cost_stats) > 0:
                    policy_cost_stats.append((policy_id, f"{np.mean(cost_stats):.3f}"))
            console_stats["cost"] = policy_cost_stats
            
        return console_stats

    def update_regular_stats(self, key: str, value: Any) -> None:
        """Update regular (non-averaged) stats."""
        self.stats[key] = value

    def update_averaged_stats(self, key: str, value: Any, maxlen: int = 100) -> None:
        """Update averaged stats."""
        if key not in self.avg_stats:
            self.avg_stats[key] = deque(maxlen=maxlen)
        self.avg_stats[key].append(value)

    def get_regular_stats(self) -> Dict[str, Any]:
        """Get regular stats for logging."""
        return self.stats.copy()

    def get_averaged_stats(self, total_train_seconds: float) -> Dict[str, float]:
        """Get averaged stats for logging."""
        stats_to_log = {}
        for key, value in self.avg_stats.items():
            if len(value) >= value.maxlen or (len(value) > 10 and total_train_seconds > 300):
                stats_to_log[key] = np.mean(value)
        return stats_to_log

    def clear_policy_stats(self, policy_id: int) -> None:
        """Clear stats for a specific policy (useful for testing)."""
        for key in self.policy_avg_stats:
            if policy_id < len(self.policy_avg_stats[key]):
                self.policy_avg_stats[key][policy_id].clear()

    def clear_all_stats(self) -> None:
        """Clear all stats (useful for testing)."""
        self.policy_avg_stats.clear()
        self.stats.clear()
        self.avg_stats.clear()