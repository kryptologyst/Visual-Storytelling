"""
Gradio demo for Visual Storytelling

This module provides a Gradio interface for visual storytelling.
"""

import gradio as gr
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import time
from typing import Optional, Tuple, List
import logging
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import VisualStorytellingModel, StorytellingConfig, SimpleVisualStorytellingModel
from src.utils import get_device, set_seed, load_config
from src.eval import StorytellingEvaluator


# Global variables
model = None
device = None


def load_model(model_path: str, device_name: str = "auto") -> bool:
    """Load the visual storytelling model."""
    global model, device
    
    try:
        # Setup device
        if device_name == "auto":
            device = get_device()
        else:
            device = torch.device(device_name)
        
        # Load configuration
        config = StorytellingConfig()
        
        # Create model
        model = SimpleVisualStorytellingModel(config)
        
        # Load checkpoint if available
        checkpoint_path = Path(model_path)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        else:
            print(f"No checkpoint found at {model_path}, using pre-trained model")
        
        model.to(device)
        model.eval()
        
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False


def preprocess_image(image: Image.Image, target_size: int = 224) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize image
    image = image.resize((target_size, target_size))
    
    # Convert to tensor and normalize
    image_array = np.array(image) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor.unsqueeze(0)


def generate_story(image: Image.Image, max_length: int = 150, 
                  num_beams: int = 5, temperature: float = 1.0) -> Tuple[str, float]:
    """Generate story from image."""
    global model, device
    
    if model is None:
        return "Error: Model not loaded", 0.0
    
    try:
        # Preprocess image
        pixel_values = preprocess_image(image).to(device)
        
        # Generate story
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=temperature > 0
            )
        
        generation_time = time.time() - start_time
        
        # Decode story
        story = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return story, generation_time
    except Exception as e:
        return f"Error generating story: {e}", 0.0


def compute_story_metrics(story: str) -> dict:
    """Compute basic metrics for the generated story."""
    if not story or story.startswith("Error"):
        return {"Word Count": 0, "Sentence Count": 0, "Readability": "N/A"}
    
    words = story.split()
    sentences = story.split('.')
    
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Simple readability score
    if sentence_count > 0:
        avg_words_per_sentence = word_count / sentence_count
        if avg_words_per_sentence < 10:
            readability = "Easy"
        elif avg_words_per_sentence < 20:
            readability = "Medium"
        else:
            readability = "Complex"
    else:
        readability = "N/A"
    
    return {
        "Word Count": word_count,
        "Sentence Count": sentence_count,
        "Readability": readability
    }


def create_interface():
    """Create the Gradio interface."""
    
    # Load model
    model_loaded = load_model("checkpoints/best_model.pt")
    
    with gr.Blocks(
        title="Visual Storytelling",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .story-output {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #007bff;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 📖 Visual Storytelling Demo
        
        Transform images into compelling narratives using advanced computer vision and natural language processing.
        
        **Instructions:**
        1. Upload an image or use one of the example images
        2. Adjust the generation parameters if desired
        3. Click "Generate Story" to create a narrative
        4. View the generated story and evaluation metrics
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image input
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400
                )
                
                # Example images
                gr.Markdown("### Example Images")
                with gr.Row():
                    example_btn1 = gr.Button("Sunset", variant="secondary")
                    example_btn2 = gr.Button("Children", variant="secondary")
                    example_btn3 = gr.Button("Coffee Shop", variant="secondary")
                
                # Generation parameters
                gr.Markdown("### Generation Parameters")
                max_length = gr.Slider(
                    minimum=50, maximum=200, value=150, step=10,
                    label="Max Length"
                )
                num_beams = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Number of Beams"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                    label="Temperature"
                )
                
                # Generate button
                generate_btn = gr.Button(
                    "Generate Story", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Story output
                story_output = gr.Textbox(
                    label="Generated Story",
                    lines=10,
                    max_lines=15,
                    show_copy_button=True,
                    elem_classes=["story-output"]
                )
                
                # Generation info
                generation_time = gr.Textbox(
                    label="Generation Time",
                    interactive=False
                )
                
                # Metrics
                gr.Markdown("### Story Metrics")
                metrics_output = gr.JSON(
                    label="Metrics",
                    value={}
                )
                
                # Download button
                download_btn = gr.DownloadButton(
                    "Download Story",
                    variant="secondary"
                )
        
        # Example images (placeholder paths)
        example_images = [
            "assets/examples/sunset.jpg",
            "assets/examples/children.jpg",
            "assets/examples/coffee_shop.jpg"
        ]
        
        # Event handlers
        def generate_story_wrapper(image, max_len, beams, temp):
            """Wrapper for story generation."""
            if image is None:
                return "Please upload an image first.", "N/A", {}
            
            story, gen_time = generate_story(image, max_len, beams, temp)
            metrics = compute_story_metrics(story)
            
            return story, f"{gen_time:.2f} seconds", metrics
        
        def load_example_image(example_idx):
            """Load example image."""
            example_path = example_images[example_idx]
            if Path(example_path).exists():
                return Image.open(example_path)
            else:
                # Create a placeholder image
                return Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Connect events
        generate_btn.click(
            fn=generate_story_wrapper,
            inputs=[image_input, max_length, num_beams, temperature],
            outputs=[story_output, generation_time, metrics_output]
        )
        
        example_btn1.click(
            fn=lambda: load_example_image(0),
            outputs=image_input
        )
        
        example_btn2.click(
            fn=lambda: load_example_image(1),
            outputs=image_input
        )
        
        example_btn3.click(
            fn=lambda: load_example_image(2),
            outputs=image_input
        )
        
        # Download functionality
        def create_download_file(story):
            """Create downloadable text file."""
            if story and not story.startswith("Error"):
                return story
            return None
        
        story_output.change(
            fn=create_download_file,
            inputs=story_output,
            outputs=download_btn
        )
        
        # Model status
        if model_loaded:
            gr.Markdown("✅ Model loaded successfully")
        else:
            gr.Markdown("⚠️ Model loading failed - using pre-trained components")
        
        # Footer
        gr.Markdown("""
        ---
        **Visual Storytelling Demo** | Powered by Advanced Computer Vision
        
        This demo showcases state-of-the-art visual storytelling capabilities using transformer-based architectures.
        """)
    
    return demo


def main():
    """Main function to launch the Gradio demo."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create and launch interface
    demo = create_interface()
    
    # Launch with specific configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_tips=True
    )


if __name__ == "__main__":
    main()
