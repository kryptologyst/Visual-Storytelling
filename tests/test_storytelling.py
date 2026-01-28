"""
Test suite for Visual Storytelling

This module contains unit tests for the visual storytelling components.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json
from PIL import Image

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import VisualStorytellingModel, StorytellingConfig, SimpleVisualStorytellingModel
from src.data import StorytellingDataset, StorytellingDataConfig, StorytellingDataModule
from src.eval import StorytellingEvaluator, EvaluationConfig
from src.utils import set_seed, get_device, count_parameters, clean_text


class TestStorytellingConfig:
    """Test StorytellingConfig class."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = StorytellingConfig()
        assert config.vision_model_name == "google/vit-base-patch16-224"
        assert config.text_model_name == "gpt2"
        assert config.hidden_size == 768
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid hidden_size
        with pytest.raises(ValueError):
            StorytellingConfig(hidden_size=769, num_attention_heads=12)


class TestVisualStorytellingModel:
    """Test VisualStorytellingModel class."""
    
    def setup_method(self):
        """Setup test method."""
        self.config = StorytellingConfig()
        self.device = torch.device("cpu")
    
    def test_model_creation(self):
        """Test model creation."""
        model = SimpleVisualStorytellingModel(self.config)
        assert isinstance(model, SimpleVisualStorytellingModel)
        assert count_parameters(model) > 0
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = SimpleVisualStorytellingModel(self.config)
        model.eval()
        
        # Create dummy inputs
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (batch_size, 10))
        attention_mask = torch.ones(batch_size, 10)
        labels = torch.randint(0, 1000, (batch_size, 10))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape[0] == batch_size
        assert outputs["loss"].item() >= 0
    
    def test_model_generation(self):
        """Test model generation."""
        model = SimpleVisualStorytellingModel(self.config)
        model.eval()
        
        # Create dummy image
        pixel_values = torch.randn(1, 3, 224, 224)
        
        # Generate story
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=50,
                num_beams=3
            )
        
        assert generated_ids.shape[0] == 1
        assert generated_ids.shape[1] <= 50


class TestStorytellingDataset:
    """Test StorytellingDataset class."""
    
    def setup_method(self):
        """Setup test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create dummy data
        self.create_dummy_data()
        
        self.config = StorytellingDataConfig(
            data_dir=str(self.temp_path),
            batch_size=2,
            num_workers=0
        )
    
    def create_dummy_data(self):
        """Create dummy data for testing."""
        # Create JSON data
        dummy_data = [
            {
                "image": "image1.jpg",
                "story": "A beautiful sunset over the mountains.",
                "caption": "Sunset mountains"
            },
            {
                "image": "image2.jpg", 
                "story": "Children playing in the park.",
                "caption": "Children playing"
            }
        ]
        
        with open(self.temp_path / "train.json", 'w') as f:
            json.dump(dummy_data, f)
        
        # Create dummy images
        for i in range(1, 3):
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            img.save(self.temp_path / f"image{i}.jpg")
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = StorytellingDataset(
            data_path=self.temp_path,
            split="train",
            config=self.config
        )
        
        assert len(dataset) == 2
        assert dataset.split == "train"
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        dataset = StorytellingDataset(
            data_path=self.temp_path,
            split="train",
            config=self.config
        )
        
        item = dataset[0]
        
        assert "pixel_values" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert "story" in item
        
        assert item["pixel_values"].shape == (3, 224, 224)
        assert item["input_ids"].shape[0] == self.config.max_text_length
    
    def test_synthetic_data(self):
        """Test synthetic data creation."""
        # Remove data files to trigger synthetic data creation
        (self.temp_path / "train.json").unlink()
        
        dataset = StorytellingDataset(
            data_path=self.temp_path,
            split="train",
            config=self.config
        )
        
        assert len(dataset) == 3  # Synthetic data has 3 samples
        assert all(item["metadata"].get("synthetic", False) for item in dataset.data)


class TestStorytellingDataModule:
    """Test StorytellingDataModule class."""
    
    def setup_method(self):
        """Setup test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create dummy data
        self.create_dummy_data()
        
        self.config = StorytellingDataConfig(
            data_dir=str(self.temp_path),
            batch_size=2,
            num_workers=0
        )
    
    def create_dummy_data(self):
        """Create dummy data for testing."""
        # Create JSON data for train and val splits
        dummy_data = [
            {
                "image": "image1.jpg",
                "story": "A beautiful sunset over the mountains.",
                "caption": "Sunset mountains"
            },
            {
                "image": "image2.jpg",
                "story": "Children playing in the park.",
                "caption": "Children playing"
            }
        ]
        
        for split in ["train", "val"]:
            with open(self.temp_path / f"{split}.json", 'w') as f:
                json.dump(dummy_data, f)
        
        # Create dummy images
        for i in range(1, 3):
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            img.save(self.temp_path / f"image{i}.jpg")
    
    def test_data_module_creation(self):
        """Test data module creation."""
        data_module = StorytellingDataModule(self.config)
        
        assert data_module.config == self.config
        assert data_module.tokenizer is not None
        assert data_module.processor is not None
    
    def test_dataloader_creation(self):
        """Test dataloader creation."""
        data_module = StorytellingDataModule(self.config)
        
        train_loader = data_module.get_train_dataloader()
        val_loader = data_module.get_val_dataloader()
        
        assert train_loader is not None
        assert val_loader is not None
        
        # Test batch iteration
        for batch in train_loader:
            assert "pixel_values" in batch
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            assert batch["pixel_values"].shape[0] <= self.config.batch_size
            break


class TestStorytellingEvaluator:
    """Test StorytellingEvaluator class."""
    
    def setup_method(self):
        """Setup test method."""
        self.config = EvaluationConfig()
        self.evaluator = StorytellingEvaluator(self.config)
    
    def test_evaluator_creation(self):
        """Test evaluator creation."""
        assert self.evaluator is not None
        assert self.evaluator.config == self.config
    
    def test_text_preprocessing(self):
        """Test text preprocessing."""
        text = "Hello, World! This is a test."
        
        processed = self.evaluator.preprocess_text(text)
        assert isinstance(processed, str)
        
        if self.config.lowercase:
            assert processed.islower()
    
    def test_text_tokenization(self):
        """Test text tokenization."""
        text = "Hello world this is a test"
        
        tokens = self.evaluator.tokenize_text(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_evaluation(self):
        """Test evaluation metrics computation."""
        predictions = [
            "A beautiful sunset over the mountains creates a peaceful atmosphere.",
            "Children playing in the park on a sunny day."
        ]
        references = [
            ["A beautiful sunset over the mountains creates a peaceful atmosphere."],
            ["Children playing in the park on a sunny day."]
        ]
        captions = [
            "Sunset over mountains",
            "Children playing"
        ]
        
        results = self.evaluator.evaluate(predictions, references, captions)
        
        assert isinstance(results, dict)
        assert "overall_score" in results
        
        # Check that metrics are reasonable
        for metric, value in results.items():
            assert isinstance(value, (int, float))
            assert 0 <= value <= 1 or metric in ["overall_score", "avg_sentence_length", "word_count"]


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Check that random numbers are deterministic
        rand1 = torch.randn(10)
        set_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        param_count = count_parameters(model)
        assert param_count == 55  # 10*5 + 5 bias
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "  Hello,   world!  "
        cleaned = clean_text(text)
        assert cleaned == "Hello, world!"
        
        # Test with special tokens
        text_with_tokens = "Hello <pad> world <eos>"
        cleaned = clean_text(text_with_tokens)
        assert "<pad>" not in cleaned
        assert "<eos>" not in cleaned


if __name__ == "__main__":
    pytest.main([__file__])
