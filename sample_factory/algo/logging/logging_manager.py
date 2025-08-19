from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
from tensorboardX import SummaryWriter

from sample_factory.algo.logging.backends import TensorBoardBackend, WandBBackend, MultiBackend
from sample_factory.algo.logging.interfaces import LoggingConfig, LoggingBackend
from sample_factory.algo.logging.loggers import HeatmapLogger, VideoLogger, PerformanceLogger
from sample_factory.algo.logging.stats_processor import StatsProcessor
from sample_factory.utils.utils import (
    ensure_dir_exists,
    experiment_dir,
    summaries_dir,
    frames_dir,
    memory_consumption_mb,
)


class LoggingManager:
    """Central coordinator for all logging."""

    def __init__(self, config: LoggingConfig, experiment_dir_path: str, num_policies: int, env_info=None):
        self.config = config
        self.experiment_dir_path = experiment_dir_path
        self.num_policies = num_policies
        self.env_info = env_info

        # Create backends
        self.backends = self._create_backends()

        # Create specialized loggers
        self.heatmap_logger = HeatmapLogger(
            config.heatmap_config, 
            self.backends, 
            env_info
        )
        self.video_logger = VideoLogger(config.video_config, self.backends)
        self.performance_logger = PerformanceLogger(config.performance_config, self.backends)

        # Create stats processor
        self.stats_processor = StatsProcessor(config.stats_config, num_policies)

        # Frames directory for video creation
        self.frames_dir = None

        # Track total training time
        self.total_train_seconds = 0

    def _create_backends(self) -> List[LoggingBackend]:
        """Create and configure logging backends."""
        backends = []

        # TensorBoard backend (always enabled for now)
        if self.config.tensorboard_enabled:
            self.writers: Dict[int, SummaryWriter] = {}
            for policy_id in range(self.num_policies):
                summary_dir = os.path.join(summaries_dir(self.experiment_dir_path), str(policy_id))
                summary_dir = ensure_dir_exists(summary_dir)
                writer = SummaryWriter(summary_dir, flush_secs=30)  # Default flush interval
                self.writers[policy_id] = writer
                backends.append(TensorBoardBackend(writer))

        # WandB backend
        if self.config.wandb_enabled:
            backends.append(WandBBackend())

        # If multiple backends, use MultiBackend for convenience
        if len(backends) > 1:
            return [MultiBackend(backends)]
        elif len(backends) == 1:
            return backends
        else:
            # Fallback: create a dummy TensorBoard backend
            summary_dir = os.path.join(summaries_dir(self.experiment_dir_path), "0")
            summary_dir = ensure_dir_exists(summary_dir)
            writer = SummaryWriter(summary_dir)
            self.writers = {0: writer}
            return [TensorBoardBackend(writer)]

    def set_map_image(self, map_img: np.ndarray) -> None:
        """Set the background map image for heatmap overlays."""
        self.heatmap_logger.set_map_image(map_img)

    def process_episodic_stats(self, stats: Dict[str, Any], policy_id: int) -> None:
        """Process episodic stats from environment."""
        self.stats_processor.process_episodic_stats(stats, policy_id)

    def process_train_stats(self, train_stats: Dict[str, Any], policy_id: int, env_steps: int) -> None:
        """Process training stats and log immediately."""
        for key, scalar in train_stats.items():
            for backend in self.backends:
                backend.log_scalar(f"train/{key}", scalar, env_steps)

    def add_performance_sample(self, timestamp: float, total_steps: int, samples_per_policy: Dict[int, int]) -> None:
        """Add performance samples for FPS and throughput calculation."""
        self.performance_logger.add_fps_sample(timestamp, total_steps)
        for policy_id, samples in samples_per_policy.items():
            self.performance_logger.add_throughput_sample(policy_id, timestamp, samples)

    def log_step_stats(self, stats: Dict[str, Any], step: int) -> None:
        """Log step-level statistics."""
        for key, value in stats.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                for backend in self.backends:
                    backend.log_scalar(f"stats/{key}", float(value), step)

    def log_episode_summaries(self, total_train_seconds: float) -> None:
        """Log episode summaries for all policies."""
        self.total_train_seconds = total_train_seconds

        # Get performance stats
        fps_stats = self.performance_logger.get_fps_stats()
        throughput_stats = self.performance_logger.get_throughput_stats()

        # Log performance stats
        perf_stats = {
            'fps': fps_stats[0] if fps_stats else float('nan'),
            'memory_mb': memory_consumption_mb(),
        }

        # Add throughput stats
        for policy_id, throughput in throughput_stats.items():
            perf_stats[f'throughput_policy_{policy_id}'] = throughput

        # Log performance stats for the default policy
        default_policy = 0
        if default_policy in self._get_env_steps():
            step = self._get_env_steps()[default_policy]
            self.performance_logger.log_performance_stats(perf_stats, step)

        # Process and log policy stats
        for policy_id in range(self.num_policies):
            env_steps = self._get_env_steps().get(policy_id, 0)
            if env_steps == 0:
                continue

            # Get processed stats for this policy
            policy_stats = self.stats_processor.get_policy_stats_for_logging(
                policy_id, env_steps, total_train_seconds
            )

            # Log policy stats
            for key, value in policy_stats.items():
                for backend in self.backends:
                    backend.log_scalar(key, value, env_steps)

            # Handle heatmap logging
            if self.config.heatmap_config.enabled:
                heatmap_data = self.stats_processor.get_heatmap_data(policy_id)
                if heatmap_data is not None:
                    # Update cumulative heatmap
                    self.heatmap_logger.update_cumulative_heatmap(heatmap_data)

                    # Check if we should log heatmaps
                    if self.heatmap_logger.should_log(env_steps):
                        # Ensure frames directory exists
                        if not self.frames_dir:
                            self.frames_dir = frames_dir(self.experiment_dir_path)

                        # Log overlay (which also saves frames for GIF)
                        if self.config.heatmap_config.log_overlay:
                            self.heatmap_logger.log_overlay(
                                heatmap_data, env_steps, 'traversal/overlay', self.frames_dir
                            )

                        # Log heatmaps
                        if self.config.heatmap_config.log_heatmap:
                            self.heatmap_logger.log_heatmap(
                                self.heatmap_logger.cumulative_heatmap, env_steps, "traversal/cumulative"
                            )
                            self.heatmap_logger.log_heatmap(heatmap_data, env_steps, "traversal/window")

                        self.heatmap_logger.update_last_log_step(env_steps)

                    # Check if we should create and log GIF
                    if self.video_logger.should_log(env_steps) and self.frames_dir:
                        self.video_logger.create_and_log_gif(self.frames_dir, env_steps, "traversal/evolution")
                        self.video_logger.update_last_log_step(env_steps)

        # Log regular and averaged stats
        regular_stats = self.stats_processor.get_regular_stats()
        averaged_stats = self.stats_processor.get_averaged_stats(total_train_seconds)

        default_policy = 0
        if default_policy in self._get_env_steps():
            step = self._get_env_steps()[default_policy]

            # Log regular stats
            for key, value in regular_stats.items():
                for backend in self.backends:
                    backend.log_scalar(f"stats/{key}", value, step)

            # Log averaged stats
            for key, value in averaged_stats.items():
                for backend in self.backends:
                    backend.log_scalar(f"stats/{key}", value, step)

        # Handle video logging from files
        if self.config.wandb_enabled:
            self._log_new_videos_to_wandb()

    def _log_new_videos_to_wandb(self) -> None:
        """Log new video files to WandB."""
        video_dir = os.path.join(self.experiment_dir_path, self.config.video_config.video_dir)
        if not os.path.exists(video_dir):
            return

        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        episode_pattern = re.compile(r"doom-step-(\d+).mp4")

        # For simplicity, we'll log all videos found
        # In a more sophisticated implementation, we'd track which videos have been logged
        for video_file in video_files:
            match = episode_pattern.search(video_file)
            if match:
                step = int(match.group(1))
                video_path = os.path.join(video_dir, video_file)

                # Log to WandB backend only
                for backend in self.backends:
                    if isinstance(backend, WandBBackend):
                        backend.log_video("videos/episode", video_path, step)
                    elif isinstance(backend, MultiBackend):
                        for sub_backend in backend.backends:
                            if isinstance(sub_backend, WandBBackend):
                                sub_backend.log_video("videos/episode", video_path, step)

    def get_console_stats(self) -> Dict[str, Any]:
        """Get stats for console logging."""
        return self.stats_processor.get_console_stats()

    def update_regular_stats(self, key: str, value: Any) -> None:
        """Update regular stats."""
        self.stats_processor.update_regular_stats(key, value)

    def update_averaged_stats(self, key: str, value: Any, maxlen: int = 100) -> None:
        """Update averaged stats."""
        self.stats_processor.update_averaged_stats(key, value, maxlen)

    def flush(self) -> None:
        """Flush all backends."""
        for backend in self.backends:
            backend.flush()

    def set_env_steps(self, env_steps: Dict[int, int]) -> None:
        """Set environment steps from Runner."""
        self._env_steps = env_steps

    def _get_env_steps(self) -> Dict[int, int]:
        """Get environment steps."""
        return getattr(self, '_env_steps', {})
