"""Streamlit dashboard for the explainability framework."""

import streamlit as st
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_xai.models import load_pretrained_model
from hybrid_xai.framework import ExplainabilityFramework
from hybrid_xai.visualization import ComparativeVisualizer


def main():
    """Run Streamlit dashboard."""
    st.set_page_config(page_title="Hybrid XAI Framework", layout="wide")
    
    st.title("🤖 Hybrid Explainable AI Framework")
    st.markdown(
        """
        Interactive dashboard for comparing LIME, SHAP, and Grad-CAM explanations.
        """
    )
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["resnet18", "resnet50", "vgg16", "efficientnet_b0"]
    )
    
    num_classes = st.sidebar.slider("Number of Classes", 2, 1000, 10)
    
    methods = st.sidebar.multiselect(
        "Explanation Methods",
        ["lime", "shap", "gradcam"],
        default=["lime", "shap", "gradcam"]
    )
    
    # Main content
    st.header("Generate Explanations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    with col2:
        st.subheader("Settings")
        target_class = st.number_input("Target Class", 0, num_classes - 1, 0)
    
    if uploaded_file is not None:
        st.info("Loading model...")
        
        model = load_pretrained_model(
            model_name,
            num_classes=num_classes,
            pretrained=True
        )
        framework = ExplainabilityFramework(model)
        
        image = None
        try:
            from PIL import Image
            image = Image.open(uploaded_file)
            image_array = np.array(image) / 255.0
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return
        
        st.success("Model loaded!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Input Image", width=300)
        
        with col2:
            st.write("### Model Info")
            st.write(f"Model: {model_name}")
            st.write(f"Classes: {num_classes}")
            st.write(f"Methods: {', '.join(methods)}")
        
        if st.button("Generate Explanations"):
            with st.spinner("Generating explanations..."):
                try:
                    explanations = framework.explain(
                        image_array,
                        methods=methods,
                        target_class=target_class
                    )
                    
                    st.success("Explanations generated!")
                    
                    # Display explanations
                    for method, exp in explanations.items():
                        with st.expander(f"{method.upper()} Explanation", expanded=True):
                            st.write(f"Method: {exp.get('method', 'N/A')}")
                            
                            if "attributions" in exp:
                                st.write("Attribution shape: ", exp["attributions"].shape)
                    
                    # Comparative visualization
                    st.subheader("Comparative Analysis")
                    fig = ComparativeVisualizer.compare_explanations(
                        image_array,
                        explanations
                    )
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error generating explanations: {e}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Hybrid XAI Framework** v0.1.0")
    st.sidebar.markdown(
        "Combining LIME, SHAP, and Grad-CAM for interpretable deep learning."
    )


if __name__ == "__main__":
    main()
