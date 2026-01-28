#!/usr/bin/env python3
"""
Quick start script for Visual Storytelling

This script demonstrates basic usage of the visual storytelling system.
"""

import sys
from pathlib import Path
import torch
from PIL import Image
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models import SimpleVisualStorytellingModel, StorytellingConfig
from src.utils import get_device, set_seed, setup_logging


def main():
    """Quick start demonstration."""
    # Setup logging
    setup_logging("INFO")
    logging.info("Visual Storytelling Quick Start")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Create model
    logging.info("Loading model...")
    config = StorytellingConfig()
    model = SimpleVisualStorytellingModel(config)
    model.to(device)
    model.eval()
    
    # Create a sample image (or load from file)
    logging.info("Creating sample image...")
    sample_image = Image.new('RGB', (224, 224), color=(255, 165, 0))  # Orange image
    
    # Generate story
    logging.info("Generating story...")
    with torch.no_grad():
        story = model.generate_story(sample_image)
    
    # Display results
    print("\n" + "="*50)
    print("VISUAL STORYTELLING DEMO")
    print("="*50)
    print(f"Device: {device}")
    print(f"Model: {config.text_model_name}")
    print(f"Generated Story: {story}")
    print("="*50)
    
    logging.info("Demo completed successfully!")


if __name__ == "__main__":
    main()
