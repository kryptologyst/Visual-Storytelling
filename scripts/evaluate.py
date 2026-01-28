#!/usr/bin/env python3
"""
Evaluation script for Visual Storytelling

This script evaluates trained visual storytelling models on test datasets.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import json
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models import VisualStorytellingModel, StorytellingConfig, SimpleVisualStorytellingModel
from src.data import StorytellingDataModule, StorytellingDataConfig
from src.eval import StorytellingEvaluator, EvaluationConfig
from src.utils import set_seed, get_device, load_config, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Visual Storytelling Model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="simple",
                       choices=["simple", "advanced"],
                       help="Type of model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--test_split", type=str, default="test",
                       help="Test split name")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Generation arguments
    parser.add_argument("--max_length", type=int, default=150,
                       help="Maximum generation length")
    parser.add_argument("--num_beams", type=int, default=5,
                       help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    
    # Evaluation arguments
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output file for evaluation results")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save generated predictions")
    parser.add_argument("--predictions_file", type=str, default="predictions.json",
                       help="File to save predictions")
    
    # Device and reproducibility
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def load_model(model_path: str, model_type: str, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    logging.info(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    if model_type == "simple":
        model = SimpleVisualStorytellingModel(StorytellingConfig())
    else:
        model = VisualStorytellingModel(StorytellingConfig())
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logging.info("Model loaded successfully")
    return model


def evaluate_model(
    model: torch.nn.Module,
    data_module: StorytellingDataModule,
    args: argparse.Namespace,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate model on test dataset."""
    logging.info("Starting evaluation...")
    
    # Get test dataloader
    test_dataloader = data_module.get_test_dataloader()
    
    # Setup evaluator
    eval_config = EvaluationConfig()
    evaluator = StorytellingEvaluator(eval_config)
    
    # Generate predictions
    predictions = []
    references = []
    captions = []
    all_stories = []
    all_captions = []
    all_image_paths = []
    all_metadata = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            logging.info(f"Processing batch {batch_idx + 1}/{len(test_dataloader)}")
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate predictions
            generated_ids = model.generate(
                pixel_values=batch['pixel_values'],
                max_length=args.max_length,
                num_beams=args.num_beams,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )
            
            # Decode predictions
            for i in range(len(generated_ids)):
                pred_text = data_module.tokenizer.decode(
                    generated_ids[i], skip_special_tokens=True
                )
                predictions.append(pred_text)
                
                # Get reference story
                ref_text = batch['stories'][i]
                references.append([ref_text])
                
                # Get caption
                caption = batch['captions'][i] if batch['captions'][i] else ref_text
                captions.append(caption)
                
                # Store additional info
                all_stories.append(ref_text)
                all_captions.append(caption)
                all_image_paths.append(batch['image_paths'][i])
                all_metadata.append(batch['metadata'][i])
    
    logging.info(f"Generated {len(predictions)} predictions")
    
    # Compute evaluation metrics
    logging.info("Computing evaluation metrics...")
    metrics = evaluator.evaluate(predictions, references, captions)
    
    # Add generation parameters to results
    results = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "generation_params": {
            "max_length": args.max_length,
            "num_beams": args.num_beams,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k
        },
        "dataset_info": {
            "num_samples": len(predictions),
            "data_dir": args.data_dir,
            "test_split": args.test_split
        },
        "metrics": metrics
    }
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_data = []
        for i in range(len(predictions)):
            predictions_data.append({
                "image_path": all_image_paths[i],
                "predicted_story": predictions[i],
                "reference_story": all_stories[i],
                "caption": all_captions[i],
                "metadata": all_metadata[i]
            })
        
        predictions_file = Path(args.predictions_file)
        predictions_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        logging.info(f"Predictions saved to {predictions_file}")
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info("Starting Visual Storytelling evaluation")
    logging.info(f"Arguments: {args}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    logging.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, args.model_type, device)
    
    # Create data module
    logging.info("Setting up data module...")
    data_config = StorytellingDataConfig(
        data_dir=args.data_dir,
        test_split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    data_module = StorytellingDataModule(data_config)
    
    # Run evaluation
    results = evaluate_model(model, data_module, args, device)
    
    # Save results
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Evaluation results saved to {output_file}")
    
    # Print results
    logging.info("Evaluation Results:")
    logging.info("=" * 50)
    for metric, value in results["metrics"].items():
        logging.info(f"{metric:25}: {value:.4f}")
    
    logging.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
