from __future__ import annotations

import glob
import os
from collections import deque
from io import BytesIO
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from sample_factory.algo.logging.interfaces import LoggingBackend, HeatmapConfig, VideoConfig, PerfConfig


class HeatmapLogger:
    """Handles all heatmap/spatial visualization logging."""

    def __init__(self, config: HeatmapConfig, backends: List[LoggingBackend], env_info=None):
        self.config = config
        self.backends = backends
        self.env_info = env_info
        self.cumulative_heatmap: Optional[np.ndarray] = None
        self.map_img: Optional[np.ndarray] = None
        self.last_heatmap_log = 0

    def set_map_image(self, map_img: np.ndarray) -> None:
        """Set the background map image for overlay logging."""
        self.map_img = map_img

    def _reshape_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Reshape 1D heatmap to 2D if needed."""
        # Handle edge case of empty heatmap
        if heatmap.size == 0:
            return np.zeros((1, 1))

        if heatmap.ndim == 1:
            # If heatmap is 1D, we need to reshape it to 2D
            # Try to make it as square as possible
            size = heatmap.shape[0]
            if size == 0:
                return np.zeros((1, 1))

            side_length = int(np.sqrt(size))
            if side_length * side_length == size:
                heatmap = heatmap.reshape(side_length, side_length)
            else:
                # If not a perfect square, find the best rectangular shape
                # Try to find factors that are close to each other
                factors = []
                for i in range(1, int(np.sqrt(size)) + 1):
                    if size % i == 0:
                        factors.append((i, size // i))
                if factors:
                    # Choose the factor pair with the smallest difference
                    best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
                    heatmap = heatmap.reshape(best_factor[0], best_factor[1])
                else:
                    # Fallback: reshape to a single row
                    heatmap = heatmap.reshape(1, size)

        # Transpose the heatmap
        heatmap = np.flipud(heatmap.T)
        return heatmap

    def _create_heatmap_image(self, heatmap: np.ndarray) -> Image.Image:
        """Create a heatmap image from numpy array."""
        heatmap = self._reshape_heatmap(heatmap)

        # Determine aspect ratio of the histogram
        height, width = heatmap.shape

        # Handle edge case where heatmap has zero dimensions
        if height == 0 or width == 0:
            # Create a minimal 1x1 heatmap as fallback
            heatmap = np.zeros((1, 1))
            height, width = 1, 1

        aspect_ratio = width / height

        # Define additional space for the colorbar
        colorbar_width_factor = 0.25  # Approximation of colorbar width to figure width

        # Calculate figure dimensions basing the width on a fixed height
        base_height = 2 if self.env_info and self.env_info.name in ['precipice_plunge', 'detonators_dilemma'] else 7
        fig_width = base_height * aspect_ratio * (1 + colorbar_width_factor)

        # Create a BytesIO buffer to save image
        buf = BytesIO()
        plt.figure(figsize=(fig_width, base_height))
        plt.imshow(heatmap, cmap='viridis', interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)
        image = Image.open(buf)
        return image

    def _create_overlay_image(self, heatmap: np.ndarray, step: int) -> Image.Image:
        """Create an overlay image with heatmap on top of map."""
        heatmap = self._reshape_heatmap(heatmap)

        # Determine aspect ratio of the histogram
        height, width = heatmap.shape

        # Handle edge case where heatmap has zero dimensions
        if height == 0 or width == 0:
            # Create a minimal 1x1 heatmap as fallback
            heatmap = np.zeros((1, 1))
            height, width = 1, 1

        aspect_ratio = width / height

        # Define additional space for the colorbar
        colorbar_width_factor = 0.25  # Approximation of colorbar width to figure width

        # Calculate figure dimensions basing the width on a fixed height
        base_height = 2 if self.env_info and self.env_info.name in ['precipice_plunge', 'detonators_dilemma'] else 4
        fig_width = base_height * aspect_ratio * (1 + colorbar_width_factor)

        # Create a figure and axis to plot the map and heatmap
        fig, ax = plt.subplots(figsize=(fig_width, base_height))

        # Display the map
        if self.map_img is not None:
            ax.imshow(self.map_img, extent=[0, width, 0, height])

        # Overlay the heatmap: adjust 'alpha' for transparency, cmap for the color map
        ax.imshow(heatmap, cmap='viridis', alpha=0.5, interpolation='nearest', extent=[0, width, 0, height])

        # Remove x and y ticks as they are meaningless here
        plt.xticks([])
        plt.yticks([])

        # Add a step counter on the frame
        plt.text(0.99, 0.99, f'Step: {step:09d}', fontsize=12, color='white',
                 horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

        # Save the plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)
        image = Image.open(buf)
        return image

    def log_heatmap(self, heatmap: np.ndarray, step: int, tag: str = "heatmap") -> None:
        """Log a heatmap visualization."""
        if not self.config.enabled or not self.config.log_heatmap:
            return

        image = self._create_heatmap_image(heatmap)
        image_array = np.array(image)

        for backend in self.backends:
            backend.log_image(tag, image_array, step)

    def log_overlay(self, heatmap: np.ndarray, step: int, tag: str = "overlay", frames_dir: Optional[str] = None) -> None:
        """Log a heatmap overlay on the map."""
        if not self.config.enabled or not self.config.log_overlay:
            return

        image = self._create_overlay_image(heatmap, step)
        image_array = np.array(image)

        # Save frame to disk if frames_dir is provided (for GIF creation)
        if frames_dir:
            os.makedirs(frames_dir, exist_ok=True)
            frame_path = os.path.join(frames_dir, f"frame_{step:09d}.png")
            image.save(frame_path)

        for backend in self.backends:
            backend.log_image(tag, image_array, step)

    def update_cumulative_heatmap(self, heatmap: np.ndarray) -> None:
        """Update the cumulative heatmap."""
        if self.cumulative_heatmap is None:
            self.cumulative_heatmap = np.zeros_like(heatmap)
        self.cumulative_heatmap += heatmap

    def should_log(self, step: int) -> bool:
        """Check if we should log at this step."""
        return step - self.last_heatmap_log >= self.config.log_interval

    def update_last_log_step(self, step: int) -> None:
        """Update the last logged step."""
        self.last_heatmap_log = step


class VideoLogger:
    """Handles video creation and logging."""

    def __init__(self, config: VideoConfig, backends: List[LoggingBackend]):
        self.config = config
        self.backends = backends
        self.last_gif_log = 0

    def create_and_log_gif(self, frames_dir: str, step: int, tag: str = "video") -> None:
        """Create and log a GIF from frames in the directory."""
        if not self.config.enabled:
            return

        # List all the frames, sorted by extracted step number
        frame_files = sorted(
            glob.glob(os.path.join(frames_dir, "frame_*.png")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
        )

        if not frame_files:
            return

        frames = [Image.open(frame) for frame in frame_files]

        # Create a BytesIO buffer to hold the GIF
        gif_buffer = BytesIO()

        # Total GIF duration in seconds
        total_duration_secs = self.config.gif_duration

        # Calculate the duration each frame should be displayed to fit the total
        frame_duration = int((total_duration_secs * 1000) / len(frames))  # Convert seconds to milliseconds

        # Enforce minimum and maximum duration limits
        frame_duration = max(10, min(frame_duration, 100))

        # Create GIF in the buffer
        frames[0].save(
            gif_buffer, format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=frame_duration, loop=0
        )
        gif_buffer.seek(0)  # Rewind to the start of the GIF buffer

        # Save GIF to temporary file for logging
        temp_gif_path = os.path.join(frames_dir, f"temp_{step}.gif")
        with open(temp_gif_path, 'wb') as f:
            f.write(gif_buffer.getvalue())

        # Log the GIF to backends
        for backend in self.backends:
            backend.log_video(tag, temp_gif_path, step)

        # Clean up temporary file
        try:
            os.remove(temp_gif_path)
        except OSError:
            pass

        gif_buffer.close()

    def should_log(self, step: int) -> bool:
        """Check if we should log at this step."""
        return step - self.last_gif_log >= self.config.log_interval

    def update_last_log_step(self, step: int) -> None:
        """Update the last logged step."""
        self.last_gif_log = step


class PerformanceLogger:
    """Handles FPS, throughput, and performance metrics."""

    def __init__(self, config: PerfConfig, backends: List[LoggingBackend]):
        self.config = config
        self.backends = backends
        self.fps_history = deque(maxlen=config.history_length)
        self.throughput_history = {}  # Per-policy throughput history

    def add_fps_sample(self, timestamp: float, total_steps: int) -> None:
        """Add a sample for FPS calculation."""
        self.fps_history.append((timestamp, total_steps))

    def add_throughput_sample(self, policy_id: int, timestamp: float, samples: int) -> None:
        """Add a sample for throughput calculation."""
        if policy_id not in self.throughput_history:
            self.throughput_history[policy_id] = deque(maxlen=self.config.history_length)
        self.throughput_history[policy_id].append((timestamp, samples))

    def get_fps_stats(self) -> List[float]:
        """Calculate FPS statistics for different intervals."""
        fps_stats = []
        for avg_interval in self.config.avg_intervals:
            fps_for_interval = float('nan')
            if len(self.fps_history) > 1:
                t1, x1 = self.fps_history[max(0, len(self.fps_history) - 1 - avg_interval)]
                t2, x2 = self.fps_history[-1]
                if t2 > t1:
                    fps_for_interval = (x2 - x1) / (t2 - t1)
            fps_stats.append(fps_for_interval)
        return fps_stats

    def get_throughput_stats(self) -> dict:
        """Calculate throughput statistics per policy."""
        throughput_stats = {}
        for policy_id, history in self.throughput_history.items():
            throughput_stats[policy_id] = float('nan')
            if len(history) > 1:
                t1, x1 = history[0]
                t2, x2 = history[-1]
                if t2 > t1:
                    throughput_stats[policy_id] = (x2 - x1) / (t2 - t1)
        return throughput_stats

    def log_performance_stats(self, stats: dict, step: int) -> None:
        """Log performance statistics."""
        if not self.config.enabled:
            return

        for key, value in stats.items():
            if not np.isnan(value):
                for backend in self.backends:
                    backend.log_scalar(f"perf/{key}", float(value), step)
