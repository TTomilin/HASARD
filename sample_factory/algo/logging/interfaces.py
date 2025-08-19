from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class HeatmapConfig:
    """Configuration for heatmap logging."""
    log_interval: int = 1000
    enabled: bool = True
    log_overlay: bool = True
    log_heatmap: bool = True
    avg_window: int = 100


@dataclass
class VideoConfig:
    """Configuration for video logging."""
    log_interval: int = 10000
    enabled: bool = True
    gif_duration: float = 3.0
    video_dir: str = "videos"


@dataclass
class PerfConfig:
    """Configuration for performance logging."""
    enabled: bool = True
    history_length: int = 100
    avg_intervals: tuple = (2, 12, 60)  # in report intervals


@dataclass
class StatsConfig:
    """Configuration for stats processing."""
    stats_avg: int = 100
    heatmap_avg: int = 100


@dataclass
class LoggingConfig:
    """Central configuration for all logging."""
    tensorboard_enabled: bool = True
    wandb_enabled: bool = False
    log_interval: int = 1000
    heatmap_config: HeatmapConfig = field(default_factory=HeatmapConfig)
    video_config: VideoConfig = field(default_factory=VideoConfig)
    performance_config: PerfConfig = field(default_factory=PerfConfig)
    stats_config: StatsConfig = field(default_factory=StatsConfig)

    @classmethod
    def from_cfg(cls, cfg) -> 'LoggingConfig':
        """Create LoggingConfig from experiment configuration."""
        return cls(
            tensorboard_enabled=True,  # Always enabled for now
            wandb_enabled=getattr(cfg, 'with_wandb', False),
            log_interval=getattr(cfg, 'experiment_summaries_interval', 10),
            heatmap_config=HeatmapConfig(
                log_interval=getattr(cfg, 'heatmap_log_interval', 1000),
                enabled=True,
                log_overlay=getattr(cfg, 'log_overlay', True),
                log_heatmap=getattr(cfg, 'log_heatmap', True),
                avg_window=getattr(cfg, 'heatmap_avg', 100),
            ),
            video_config=VideoConfig(
                log_interval=getattr(cfg, 'gif_log_interval', 10000),
                enabled=getattr(cfg, 'with_wandb', False),
                gif_duration=getattr(cfg, 'gif_duration', 3.0),
                video_dir=getattr(cfg, 'video_dir', 'videos'),
            ),
            performance_config=PerfConfig(
                enabled=True,
                history_length=100,
                avg_intervals=(2, 12, 60),
            ),
            stats_config=StatsConfig(
                stats_avg=getattr(cfg, 'stats_avg', 100),
                heatmap_avg=getattr(cfg, 'heatmap_avg', 100),
            ),
        )


class LoggingBackend(ABC):
    """Abstract base class for logging backends."""

    @abstractmethod
    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a scalar value."""
        pass

    @abstractmethod
    def log_image(self, key: str, image: np.ndarray, step: int) -> None:
        """Log an image."""
        pass

    @abstractmethod
    def log_video(self, key: str, video_path: str, step: int) -> None:
        """Log a video."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending logs."""
        pass


class StatsCollector(ABC):
    """Abstract base class for stats collection."""

    @abstractmethod
    def collect_agent_stats(self, game, agent_id: int, scenario_name: str) -> Dict[str, Any]:
        """Collect stats for a single agent."""
        pass

    def collect_combined_stats(self, agent_stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect combined stats from multiple agents. Default implementation is empty."""
        return {}