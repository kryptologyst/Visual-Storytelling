#!/usr/bin/env python3
"""
Training script for Visual Storytelling

This script trains visual storytelling models with comprehensive configuration support.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models import VisualStorytellingModel, StorytellingConfig, SimpleVisualStorytellingModel
from src.data import StorytellingDataModule, StorytellingDataConfig
from src.train import StorytellingTrainer, TrainingConfig
from src.eval import StorytellingEvaluator, EvaluationConfig
from src.utils import set_seed, get_device, load_config, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Visual Storytelling Model")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="simple", 
                       choices=["simple", "advanced"],
                       help="Type of model to train")
    parser.add_argument("--vision_model", type=str, default="google/vit-base-patch16-224",
                       help="Vision model name")
    parser.add_argument("--text_model", type=str, default="gpt2",
                       help="Text model name")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Number of warmup steps")
    
    # Optimization arguments
    parser.add_argument("--optimizer", type=str, default="adamw",
                       choices=["adamw", "adam", "sgd"],
                       help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "linear", "constant"],
                       help="Learning rate scheduler")
    parser.add_argument("--use_amp", action="store_true",
                       help="Use automatic mixed precision")
    
    # Evaluation arguments
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation frequency in steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Checkpoint saving frequency in steps")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--logging_dir", type=str, default="logs",
                       help="Logging directory")
    
    # Device and reproducibility
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic training")
    
    # Logging and tracking
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="visual-storytelling",
                       help="Wandb project name")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Configuration file
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Configuration file path")
    
    return parser.parse_args()


def create_configs(args):
    """Create configuration objects from arguments."""
    
    # Load config file if provided
    if Path(args.config).exists():
        config_dict = load_config(args.config)
        
        # Override with command line arguments
        if hasattr(config_dict, 'model'):
            model_config = StorytellingConfig(**config_dict.model)
        else:
            model_config = StorytellingConfig()
        
        if hasattr(config_dict, 'data'):
            data_config = StorytellingDataConfig(**config_dict.data)
        else:
            data_config = StorytellingDataConfig()
        
        if hasattr(config_dict, 'training'):
            training_config = TrainingConfig(**config_dict.training)
        else:
            training_config = TrainingConfig()
        
        if hasattr(config_dict, 'evaluation'):
            eval_config = EvaluationConfig(**config_dict.evaluation)
        else:
            eval_config = EvaluationConfig()
    else:
        # Create configs from arguments
        model_config = StorytellingConfig(
            vision_model_name=args.vision_model,
            text_model_name=args.text_model
        )
        
        data_config = StorytellingDataConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        training_config = TrainingConfig(
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            use_amp=args.use_amp,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            output_dir=args.output_dir,
            logging_dir=args.logging_dir,
            device=args.device,
            seed=args.seed,
            deterministic=args.deterministic,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project
        )
        
        eval_config = EvaluationConfig()
    
    return model_config, data_config, training_config, eval_config


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info("Starting Visual Storytelling training")
    logging.info(f"Arguments: {args}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create configurations
    model_config, data_config, training_config, eval_config = create_configs(args)
    
    # Setup device
    if training_config.device == "auto":
        device = get_device()
    else:
        device = torch.device(training_config.device)
    
    logging.info(f"Using device: {device}")
    
    # Create data module
    logging.info("Setting up data module...")
    data_module = StorytellingDataModule(data_config)
    
    # Create model
    logging.info("Creating model...")
    if args.model_type == "simple":
        model = SimpleVisualStorytellingModel(model_config)
    else:
        model = VisualStorytellingModel(model_config)
    
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    
    # Create trainer
    logging.info("Setting up trainer...")
    trainer = StorytellingTrainer(
        model=model,
        data_module=data_module,
        config=training_config,
        eval_config=eval_config
    )
    
    # Start training
    logging.info("Starting training...")
    try:
        trainer.train()
        logging.info("Training completed successfully!")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise
    
    # Final evaluation
    logging.info("Running final evaluation...")
    test_dataloader = data_module.get_test_dataloader()
    final_metrics = trainer.evaluate(test_dataloader)
    
    logging.info("Final test metrics:")
    for metric, value in final_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")
    
    logging.info("Training script completed!")


if __name__ == "__main__":
    main()
