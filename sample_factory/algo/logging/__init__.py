"""
Modular logging system for Sample Factory.

This package provides a clean, modular logging system that separates concerns:
- LoggingBackend: Abstract interface for different logging systems (TensorBoard, WandB)
- StatsCollector: Interface for collecting stats from environments
- StatsProcessor: Handles aggregation and formatting of stats
- Specialized Loggers: HeatmapLogger, VideoLogger, PerformanceLogger
- LoggingManager: Central coordinator for all logging
"""

from sample_factory.algo.logging.interfaces import (
    LoggingConfig,
    HeatmapConfig,
    VideoConfig,
    PerfConfig,
    StatsConfig,
    LoggingBackend,
    StatsCollector,
)

from sample_factory.algo.logging.backends import (
    TensorBoardBackend,
    WandBBackend,
    MultiBackend,
)

from sample_factory.algo.logging.loggers import (
    HeatmapLogger,
    VideoLogger,
    PerformanceLogger,
)

from sample_factory.algo.logging.stats_processor import StatsProcessor

from sample_factory.algo.logging.logging_manager import LoggingManager

__all__ = [
    # Configuration classes
    'LoggingConfig',
    'HeatmapConfig',
    'VideoConfig',
    'PerfConfig',
    'StatsConfig',
    
    # Interfaces
    'LoggingBackend',
    'StatsCollector',
    
    # Backends
    'TensorBoardBackend',
    'WandBBackend',
    'MultiBackend',
    
    # Specialized loggers
    'HeatmapLogger',
    'VideoLogger',
    'PerformanceLogger',
    
    # Core components
    'StatsProcessor',
    'LoggingManager',
]