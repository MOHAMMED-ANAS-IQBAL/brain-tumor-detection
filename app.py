import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMAGE_SIZE = 224
MODEL_PATH = 'best_model.pth'

# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pytorch_model():
    # 1. Check if the model file exists to prevent crashing
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file `{MODEL_PATH}` not found. Please ensure you have trained the model in the PyTorch notebook and placed `{MODEL_PATH}` in the same directory as this app.")
        st.stop()

    # 2. Initialize the base VGG16 model
    model = models.vgg16(weights=None)
    
    # 3. Rebuild the custom classifier matching your notebook
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(25088, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, len(CLASS_LABELS)) 
    )
    
    # 4. Load the trained weights (mapped to CPU so it runs on any machine)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        
    # 5. Disable inplace operations to prevent Grad-CAM errors
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False
            
    model.eval()
    return model

# ── PyTorch Grad-CAM Hook System ─────────────────────────────────────────────
class PyTorchGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.clone().detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone().detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        target = output[0, class_idx]
        target.backward()
        
        # Pool gradients across spatial dimensions
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        activations = self.activations[0].clone()
        activations = activations * pooled_gradients.view(-1, 1, 1)
            
        # Generate heatmap
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if np.max(heatmap) == 0:
            heatmap /= 1e-8
        else:
            heatmap /= np.max(heatmap)
            
        probs = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()[0]
        return heatmap, class_idx, probs

# ── Image Preprocessing ───────────────────────────────────────────────────────
def preprocess_image(image, apply_denoise=True):
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    arr = np.array(image)
    
    # Apply Gaussian Blur if selected (matching notebook behavior)
    if apply_denoise:
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
        image = Image.fromarray(arr)
        
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor, arr

# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("Configure preprocessing parameters.")
    apply_denoise = st.sidebar.checkbox("Apply Gaussian Denoising", value=True, help="Reduces noise in MRI scans before analysis.")
    st.sidebar.info("Ensure `best_model.pth` is placed in the same directory as this script.")

    # Main Content
    st.title("🧠 Brain Tumor Detection System")
    st.markdown("Upload an MRI scan to detect the presence of a brain tumor, visualize the affected region using Grad-CAM, and view class probability distributions.")

    uploaded_file = st.file_uploader("Upload an MRI Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display uploaded image
        image = Image.open(uploaded_file)
        
        st.markdown("---")
        
        # Create columns for better layout
        col_img, col_results = st.columns([1, 1.5])
        
        with col_img:
            st.subheader("Uploaded Scan")
            st.image(image, use_container_width=True)

        with col_results:
            with st.spinner("Analyzing scan and generating heatmaps..."):
                # Load Model
                model = load_pytorch_model()
                
                # Preprocess
                img_tensor, orig_img_arr = preprocess_image(image, apply_denoise=apply_denoise)
                
                # Setup Grad-CAM (attaching to the last conv layer of VGG16)
                target_layer = model.features[28]
                grad_cam = PyTorchGradCAM(model, target_layer)
                
                # Generate predictions and heatmap
                heatmap, pred_idx, probs = grad_cam.generate(img_tensor)
                
                confidence = probs[pred_idx]
                label = CLASS_LABELS[pred_idx]
                
                # Display Results Alert
                st.subheader("Analysis Result")
                if label == 'notumor':
                    st.success(f"**Diagnosis:** No Tumor Detected\n\n**Confidence:** {confidence*100:.2f}%")
                else:
                    st.error(f"**Diagnosis:** {label.capitalize()} Tumor Detected\n\n**Confidence:** {confidence*100:.2f}%")
                
                # Progress bars for all probabilities
                st.markdown("#### Probability Breakdown")
                for idx, (lbl, prob) in enumerate(zip(CLASS_LABELS, probs)):
                    st.progress(float(prob), text=f"{lbl.capitalize()}: {prob*100:.2f}%")

        # Grad-CAM Visualization Section
        st.markdown("---")
        st.subheader("🔍 Localization (Grad-CAM)")
        st.markdown("Heatmaps indicate the regions of the brain that the model focused on the most to make its decision.")
        
        # Prepare Overlay
        h, w = orig_img_arr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_img_arr, 0.6, heatmap_colored, 0.4, 0)
        
        # Display Visualizations side by side
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(orig_img_arr, caption="Original Preprocessed Image", use_container_width=True)
        with c2:
            st.image(heatmap_resized, caption="Activation Heatmap", use_container_width=True, clamp=True)
        with c3:
            st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

if __name__ == "__main__":
    main()