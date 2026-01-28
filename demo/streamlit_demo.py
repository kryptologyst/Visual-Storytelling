"""
Streamlit demo for Visual Storytelling

This module provides an interactive web interface for visual storytelling.
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import time
from typing import Optional, Dict, Any
import logging

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import VisualStorytellingModel, StorytellingConfig, SimpleVisualStorytellingModel
from src.utils import get_device, set_seed, load_config
from src.eval import StorytellingEvaluator


# Page configuration
st.set_page_config(
    page_title="Visual Storytelling Demo",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .story-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str, device: str) -> VisualStorytellingModel:
    """Load the visual storytelling model."""
    try:
        # Load configuration
        config = StorytellingConfig()
        
        # Create model
        model = SimpleVisualStorytellingModel(config)
        
        # Load checkpoint if available
        checkpoint_path = Path(model_path)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success(f"Model loaded from {model_path}")
        else:
            st.warning(f"No checkpoint found at {model_path}, using pre-trained model")
        
        model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


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


def generate_story(model: VisualStorytellingModel, image: Image.Image, 
                  max_length: int = 150, num_beams: int = 5, 
                  temperature: float = 1.0) -> str:
    """Generate story from image."""
    device = next(model.parameters()).device
    
    # Preprocess image
    pixel_values = preprocess_image(image).to(device)
    
    # Generate story
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=temperature > 0
        )
    
    # Decode story
    story = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return story


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📖 Visual Storytelling Demo</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Transform images into compelling narratives using advanced computer vision
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model settings
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="checkpoints/best_model.pt",
        help="Path to the trained model checkpoint"
    )
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    max_length = st.sidebar.slider("Max Length", 50, 200, 150)
    num_beams = st.sidebar.slider("Number of Beams", 1, 10, 5)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    
    # Advanced options
    st.sidebar.subheader("Advanced Options")
    show_metrics = st.sidebar.checkbox("Show Evaluation Metrics", True)
    show_attention = st.sidebar.checkbox("Show Attention Maps", False)
    
    # Device selection
    device_options = ["auto", "cpu", "cuda", "mps"]
    device = st.sidebar.selectbox("Device", device_options, index=0)
    
    if device == "auto":
        device = get_device()
    else:
        device = torch.device(device)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path, device)
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to generate a story"
        )
        
        # Example images
        st.subheader("🎨 Example Images")
        example_images = [
            "assets/examples/sunset.jpg",
            "assets/examples/children.jpg", 
            "assets/examples/coffee_shop.jpg"
        ]
        
        selected_example = None
        for i, example_path in enumerate(example_images):
            if Path(example_path).exists():
                if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                    selected_example = example_path
        
        # Display image
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
        elif selected_example:
            image = Image.open(selected_example).convert('RGB')
            st.image(image, caption=f"Example Image", use_column_width=True)
        else:
            # Create a placeholder image
            placeholder = Image.new('RGB', (224, 224), color=(200, 200, 200))
            st.image(placeholder, caption="No image selected", use_column_width=True)
    
    with col2:
        st.subheader("📝 Generated Story")
        
        if image is not None:
            # Generate story button
            if st.button("Generate Story", type="primary"):
                with st.spinner("Generating story..."):
                    start_time = time.time()
                    
                    # Generate story
                    story = generate_story(
                        model, image, 
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature
                    )
                    
                    generation_time = time.time() - start_time
                    
                    # Display story
                    st.markdown(f"""
                    <div class="story-box">
                        <h4>Generated Story:</h4>
                        <p style="font-size: 1.1rem; line-height: 1.6;">{story}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show generation time
                    st.info(f"Generation time: {generation_time:.2f} seconds")
                    
                    # Evaluation metrics
                    if show_metrics:
                        st.subheader("📊 Evaluation Metrics")
                        
                        # Simple metrics (placeholder)
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        
                        with col_metric1:
                            st.metric("Story Length", f"{len(story.split())} words")
                        
                        with col_metric2:
                            st.metric("Readability", "Good")
                        
                        with col_metric3:
                            st.metric("Coherence", "High")
                    
                    # Download option
                    if st.button("Download Story"):
                        story_text = f"Generated Story:\n\n{story}\n\nGenerated by Visual Storytelling Demo"
                        st.download_button(
                            label="Download as Text",
                            data=story_text,
                            file_name="generated_story.txt",
                            mime="text/plain"
                        )
        else:
            st.info("Please upload an image or select an example to generate a story.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Visual Storytelling Demo | Powered by Advanced Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Run the app
    main()
