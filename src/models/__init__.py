"""
Visual Storytelling Models

This module contains the core models for visual storytelling, including
vision encoders, text decoders, and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from dataclasses import dataclass
import logging


@dataclass
class StorytellingConfig:
    """Configuration for Visual Storytelling Model."""
    
    # Model architecture
    vision_model_name: str = "google/vit-base-patch16-224"
    text_model_name: str = "gpt2"
    cross_attention_layers: int = 6
    hidden_size: int = 768
    num_attention_heads: int = 12
    
    # Training parameters
    max_length: int = 150
    num_beams: int = 5
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Device and precision
    device: str = "auto"
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for vision-text interaction."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        """Forward pass of cross-attention.
        
        Args:
            text_features: Text features [batch_size, seq_len, hidden_size]
            vision_features: Vision features [batch_size, num_patches, hidden_size]
            
        Returns:
            torch.Tensor: Attended text features
        """
        batch_size, seq_len, _ = text_features.shape
        
        # Multi-head attention
        q = self.q_proj(text_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(vision_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(vision_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection and residual connection
        output = self.out_proj(attended)
        output = self.layer_norm(output + text_features)
        
        return output


class VisualStorytellingModel(nn.Module):
    """Advanced Visual Storytelling Model with cross-attention."""
    
    def __init__(self, config: StorytellingConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained models
        self.vision_encoder = AutoModel.from_pretrained(config.vision_model_name)
        self.text_decoder = AutoModel.from_pretrained(config.text_model_name)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout
            )
            for _ in range(config.cross_attention_layers)
        ])
        
        # Projection layers
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, config.hidden_size)
        self.text_proj = nn.Linear(self.text_decoder.config.hidden_size, config.hidden_size)
        
        # Output projection for generation
        self.output_proj = nn.Linear(config.hidden_size, self.text_decoder.config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for new layers."""
        for module in [self.vision_proj, self.text_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode vision features.
        
        Args:
            pixel_values: Input images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Vision features [batch_size, num_patches, hidden_size]
        """
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state  # [batch_size, num_patches, hidden_size]
        vision_features = self.vision_proj(vision_features)
        return self.dropout(vision_features)
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text features.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Text features [batch_size, seq_len, hidden_size]
        """
        text_outputs = self.text_decoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state
        text_features = self.text_proj(text_features)
        return self.dropout(text_features)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.
        
        Args:
            pixel_values: Input images
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for loss computation
            
        Returns:
            Dict containing logits and loss
        """
        # Encode vision and text
        vision_features = self.encode_vision(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)
        
        # Apply cross-attention layers
        for cross_attn in self.cross_attention_layers:
            text_features = cross_attn(text_features, vision_features)
        
        # Generate logits
        logits = self.output_proj(text_features)
        
        outputs = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs["loss"] = loss
        
        return outputs
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate story from image.
        
        Args:
            pixel_values: Input images
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        # Use config defaults if not provided
        max_length = max_length or self.config.max_length
        num_beams = num_beams or self.config.num_beams
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        
        # Encode vision features
        vision_features = self.encode_vision(pixel_values)
        
        # Initialize with start token
        if pad_token_id is None:
            pad_token_id = self.text_decoder.config.pad_token_id or 0
        if eos_token_id is None:
            eos_token_id = self.text_decoder.config.eos_token_id or 1
            
        input_ids = torch.full((batch_size, 1), pad_token_id, device=device, dtype=torch.long)
        
        # Generate tokens autoregressively
        for _ in range(max_length - 1):
            # Encode current text
            text_features = self.encode_text(input_ids)
            
            # Apply cross-attention
            for cross_attn in self.cross_attention_layers:
                text_features = cross_attn(text_features, vision_features)
            
            # Get next token logits
            next_token_logits = self.output_proj(text_features[:, -1, :])  # [batch_size, vocab_size]
            
            # Apply sampling strategy
            if do_sample:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS token
            if (next_token == eos_token_id).all():
                break
        
        return input_ids


class SimpleVisualStorytellingModel(nn.Module):
    """Simplified visual storytelling model using pre-trained components."""
    
    def __init__(self, config: StorytellingConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained vision-text model
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        outputs = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            labels=labels
        )
        return {"logits": outputs.logits, "loss": outputs.loss}
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate story from image."""
        max_length = max_length or self.config.max_length
        num_beams = num_beams or self.config.num_beams
        
        return self.model.generate(
            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
