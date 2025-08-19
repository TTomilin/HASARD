from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import wandb
from PIL import Image
from tensorboardX import SummaryWriter

from sample_factory.algo.logging.interfaces import LoggingBackend


class TensorBoardBackend(LoggingBackend):
    """TensorBoard logging backend."""

    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(key, value, step)

    def log_image(self, key: str, image: np.ndarray, step: int) -> None:
        """Log an image to TensorBoard."""
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in the right format for TensorBoard (HWC or CHW)
        if image.ndim == 3:
            # If image is HWC, convert to CHW for TensorBoard
            if image.shape[2] in [1, 3, 4]:  # Channels last
                image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 2:
            # Add channel dimension for grayscale
            image = np.expand_dims(image, 0)
        
        self.writer.add_image(key, image, step)

    def log_video(self, key: str, video_path: str, step: int) -> None:
        """Log a video to TensorBoard."""
        # TensorBoard doesn't have direct video support like WandB
        # We could implement this by reading the video and adding frames
        # For now, we'll skip this or log a placeholder
        pass

    def flush(self) -> None:
        """Flush TensorBoard writer."""
        self.writer.flush()


class WandBBackend(LoggingBackend):
    """Weights & Biases logging backend."""

    def __init__(self):
        # WandB should already be initialized by the time this is created
        pass

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a scalar value to WandB."""
        wandb.log({key: value}, step=step)

    def log_image(self, key: str, image: np.ndarray, step: int) -> None:
        """Log an image to WandB."""
        # Handle different image formats
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image for WandB
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:  # CHW format
                image = np.transpose(image, (1, 2, 0))  # Convert to HWC
            if image.ndim == 3 and image.shape[2] == 1:  # Grayscale with channel
                image = image.squeeze(2)
            
            # Normalize if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            image = Image.fromarray(image)
        
        wandb.log({key: wandb.Image(image)}, step=step)

    def log_video(self, key: str, video_path: str, step: int) -> None:
        """Log a video to WandB."""
        if video_path.endswith('.gif'):
            # For GIF files, read as video
            with open(video_path, 'rb') as f:
                wandb.log({key: wandb.Video(f, format="gif")}, step=step)
        else:
            # For other video formats
            wandb.log({key: wandb.Video(video_path)}, step=step)

    def flush(self) -> None:
        """Flush WandB logs."""
        # WandB handles flushing automatically, but we can force it
        wandb.log({})


class MultiBackend(LoggingBackend):
    """Backend that logs to multiple backends simultaneously."""

    def __init__(self, backends: list[LoggingBackend]):
        self.backends = backends

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log scalar to all backends."""
        for backend in self.backends:
            backend.log_scalar(key, value, step)

    def log_image(self, key: str, image: np.ndarray, step: int) -> None:
        """Log image to all backends."""
        for backend in self.backends:
            backend.log_image(key, image, step)

    def log_video(self, key: str, video_path: str, step: int) -> None:
        """Log video to all backends."""
        for backend in self.backends:
            backend.log_video(key, video_path, step)

    def flush(self) -> None:
        """Flush all backends."""
        for backend in self.backends:
            backend.flush()