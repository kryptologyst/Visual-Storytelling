"""
Training module for Visual Storytelling

This module provides training utilities, loss functions, and training loops
for visual storytelling models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Any, Union, Callable
import logging
from pathlib import Path
import time
from tqdm import tqdm
import json
from dataclasses import dataclass, asdict
import numpy as np

from ..models import VisualStorytellingModel, StorytellingConfig
from ..data import StorytellingDataModule, StorytellingDataConfig
from ..eval import StorytellingEvaluator, EvaluationConfig
from ..utils import get_device, set_seed, format_time, ensure_dir


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    num_epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, linear, constant
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    use_amp: bool = True
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    eval_strategy: str = "steps"  # steps, epoch
    
    # Checkpointing
    output_dir: str = "checkpoints"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "overall_score"
    greater_is_better: bool = True
    
    # Logging
    logging_steps: int = 100
    logging_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "visual-storytelling"
    
    # Device and reproducibility
    device: str = "auto"
    seed: int = 42
    deterministic: bool = True


class StorytellingTrainer:
    """Trainer for visual storytelling models."""
    
    def __init__(
        self,
        model: VisualStorytellingModel,
        data_module: StorytellingDataModule,
        config: TrainingConfig,
        eval_config: EvaluationConfig = None
    ):
        self.model = model
        self.data_module = data_module
        self.config = config
        self.eval_config = eval_config or EvaluationConfig()
        
        # Setup device
        if config.device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Setup evaluation
        self.evaluator = StorytellingEvaluator(self.eval_config)
        
        # Setup optimization
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('-inf') if config.greater_is_better else float('inf')
        self.training_history = []
        
        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.logging_dir = Path(config.logging_dir)
        ensure_dir(self.output_dir)
        ensure_dir(self.logging_dir)
        
        logging.info(f"Trainer initialized on device: {self.device}")
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler.lower() == "linear":
            total_steps = self.config.num_epochs * len(self.data_module.get_train_dataloader())
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps
            )
        elif self.config.scheduler.lower() == "constant":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _setup_logging(self):
        """Setup logging and tracking."""
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    config=asdict(self.config),
                    name=f"run_{int(time.time())}"
                )
                self.wandb = wandb
            except ImportError:
                logging.warning("wandb not available, skipping wandb logging")
                self.wandb = None
        else:
            self.wandb = None
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dict containing training metrics
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        if self.scaler:
            with autocast():
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
        else:
            outputs = self.model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
        
        loss = outputs['loss']
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Update scheduler
        if self.scheduler and self.config.scheduler.lower() != "cosine":
            self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dict containing evaluation metrics
        """
        self.model.eval()
        
        predictions = []
        references = []
        captions = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.scaler:
                    with autocast():
                        outputs = self.model(
                            pixel_values=batch['pixel_values'],
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                else:
                    outputs = self.model(
                        pixel_values=batch['pixel_values'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                
                total_loss += outputs['loss'].item()
                num_batches += 1
                
                # Generate predictions
                generated_ids = self.model.generate(
                    pixel_values=batch['pixel_values'],
                    max_length=self.model.config.max_length,
                    num_beams=self.model.config.num_beams
                )
                
                # Decode predictions and references
                for i in range(len(generated_ids)):
                    # Decode prediction
                    pred_text = self.data_module.tokenizer.decode(
                        generated_ids[i], skip_special_tokens=True
                    )
                    predictions.append(pred_text)
                    
                    # Use story as reference
                    ref_text = batch['stories'][i]
                    references.append([ref_text])
                    
                    # Use caption if available
                    caption = batch['captions'][i] if batch['captions'][i] else ref_text
                    captions.append(caption)
        
        # Compute evaluation metrics
        eval_loss = total_loss / num_batches
        metrics = self.evaluator.evaluate(predictions, references, captions)
        metrics['eval_loss'] = eval_loss
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            metrics: Current evaluation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': asdict(self.config)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint-{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logging.info(f"New best model saved with metric: {self.best_metric:.4f}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to stay within limit."""
        checkpoints = list(self.output_dir.glob("checkpoint-*.pt"))
        if len(checkpoints) > self.config.save_total_limit:
            # Sort by step number and remove oldest
            checkpoints.sort(key=lambda x: int(x.stem.split('-')[1]))
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                checkpoint.unlink()
    
    def train(self):
        """Main training loop."""
        logging.info("Starting training...")
        
        train_dataloader = self.data_module.get_train_dataloader()
        val_dataloader = self.data_module.get_val_dataloader()
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                train_loss += metrics['loss']
                train_steps += 1
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{metrics['learning_rate']:.2e}"
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    log_dict = {
                        'epoch': epoch,
                        'step': self.global_step,
                        'train_loss': metrics['loss'],
                        'learning_rate': metrics['learning_rate']
                    }
                    
                    if self.wandb:
                        self.wandb.log(log_dict)
                    
                    logging.info(f"Step {self.global_step}: {log_dict}")
                
                # Evaluation
                if (self.config.eval_strategy == "steps" and 
                    self.global_step % self.config.eval_steps == 0):
                    eval_metrics = self.evaluate(val_dataloader)
                    
                    # Check if this is the best model
                    current_metric = eval_metrics.get(self.config.metric_for_best_model, 0)
                    is_best = (current_metric > self.best_metric if self.config.greater_is_better 
                             else current_metric < self.best_metric)
                    
                    if is_best:
                        self.best_metric = current_metric
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(eval_metrics, is_best)
                    
                    # Log evaluation results
                    eval_log = {f"eval_{k}": v for k, v in eval_metrics.items()}
                    if self.wandb:
                        self.wandb.log(eval_log)
                    
                    logging.info(f"Evaluation at step {self.global_step}: {eval_metrics}")
            
            # End of epoch evaluation
            if self.config.eval_strategy == "epoch":
                eval_metrics = self.evaluate(val_dataloader)
                
                # Check if this is the best model
                current_metric = eval_metrics.get(self.config.metric_for_best_model, 0)
                is_best = (current_metric > self.best_metric if self.config.greater_is_better 
                         else current_metric < self.best_metric)
                
                if is_best:
                    self.best_metric = current_metric
                
                self.save_checkpoint(eval_metrics, is_best)
                
                logging.info(f"End of epoch {epoch+1} evaluation: {eval_metrics}")
            
            # Update scheduler for cosine annealing
            if self.scheduler and self.config.scheduler.lower() == "cosine":
                self.scheduler.step()
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_train_loss = train_loss / train_steps
            
            epoch_log = {
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'epoch_time': epoch_time
            }
            
            if self.wandb:
                self.wandb.log(epoch_log)
            
            logging.info(f"Epoch {epoch+1} completed in {format_time(epoch_time)}")
            logging.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Final evaluation and save
        final_eval_metrics = self.evaluate(val_dataloader)
        self.save_checkpoint(final_eval_metrics, False)
        
        total_time = time.time() - start_time
        logging.info(f"Training completed in {format_time(total_time)}")
        logging.info(f"Best {self.config.metric_for_best_model}: {self.best_metric:.4f}")
        
        if self.wandb:
            self.wandb.finish()
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        logging.info(f"Epoch: {self.epoch}, Step: {self.global_step}, Best metric: {self.best_metric:.4f}")
