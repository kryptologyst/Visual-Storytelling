"""
Visual Storytelling: Advanced Computer Vision Project

This package provides state-of-the-art visual storytelling capabilities using
modern transformer-based architectures for generating coherent narratives from images.
"""

__version__ = "0.1.0"
__author__ = "AI Projects"
__email__ = "ai@example.com"

from .models import VisualStorytellingModel, StorytellingConfig
from .data import StorytellingDataset, StorytellingDataModule
from .utils import set_seed, get_device, compute_metrics
from .train import StorytellingTrainer
from .eval import StorytellingEvaluator

__all__ = [
    "VisualStorytellingModel",
    "StorytellingConfig", 
    "StorytellingDataset",
    "StorytellingDataModule",
    "set_seed",
    "get_device",
    "compute_metrics",
    "StorytellingTrainer",
    "StorytellingEvaluator",
]
