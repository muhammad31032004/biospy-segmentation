"""
Streamlit App -- Biopsy Image Segmentation Demo
================================================
Run: streamlit run streamlit_app.py

For research and demonstration purposes only. Not for clinical use.
"""

import os
import io
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gdown


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = 256
ENCODER = "efficientnet-b4"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_segmentation_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Google Drive public link file ID (replace YOUR_FILE_ID with actual ID)
GDRIVE_FILE_ID = "1mLhVOj35bT1cm1KZjrWorIzf1FaQydtx"


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                MODEL_PATH, quiet=False,
            )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def get_inference_transform(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_mask(model, image_np, device=DEVICE):
    """Run inference on a single RGB numpy image. Returns binary mask (0/255)."""
    orig_h, orig_w = image_np.shape[:2]
    tfm = get_inference_transform()
    augmented = tfm(image=image_np)
    inp = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Resize back to original size
    prob_resized = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary_mask = (prob_resized > 0.5).astype(np.uint8) * 255
    return binary_mask, prob_resized


def create_overlay(image, mask, color=(255, 0, 0), alpha=0.4):
    """Create an overlay visualization of the mask on the image."""
    overlay = image.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
    ).astype(np.uint8)

    # Draw contours
    contours, _ = cv2.findContours(
        (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Biopsy Segmentation AI",
        page_icon="🔬",
        layout="wide",
    )

    st.title("Biopsy Image Segmentation")
    st.caption("For research and demonstration purposes only. Not for clinical use.")

    # Sidebar
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Binarization Threshold", 0.1, 0.9, 0.5, 0.05)
    overlay_alpha = st.sidebar.slider("Overlay Transparency", 0.1, 0.8, 0.4, 0.05)
    overlay_color = st.sidebar.selectbox(
        "Overlay Color",
        ["Red", "Green", "Blue", "Yellow"],
        index=0,
    )
    color_map = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0),
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model:** UNet++ (EfficientNet-B4)")
    st.sidebar.markdown(f"**Device:** {DEVICE}")
    st.sidebar.markdown("**Input:** Biopsy image (any size)")
    st.sidebar.markdown("**Output:** Binary segmentation mask")

    # Load model
    model = load_model()

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a biopsy image",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Upload a medical biopsy image for segmentation analysis",
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        st.markdown("---")

        # Predict
        with st.spinner("Running segmentation model..."):
            binary_mask, prob_map = predict_mask(model, image_rgb)

        # Apply custom threshold
        binary_mask = (prob_map > threshold).astype(np.uint8) * 255

        # Create overlay
        overlay = create_overlay(
            image_rgb, binary_mask,
            color=color_map[overlay_color],
            alpha=overlay_alpha,
        )

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, use_container_width=True)

        with col2:
            st.subheader("Predicted Mask")
            st.image(binary_mask, use_container_width=True, clamp=True)

        with col3:
            st.subheader("Overlay")
            st.image(overlay, use_container_width=True)

        # Probability heatmap
        st.markdown("---")
        col4, col5 = st.columns(2)

        with col4:
            st.subheader("Probability Heatmap")
            heatmap = cv2.applyColorMap(
                (prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            st.image(heatmap_rgb, use_container_width=True)

        with col5:
            st.subheader("Statistics")
            fg_ratio = (binary_mask > 127).sum() / binary_mask.size
            st.metric("Foreground Ratio", f"{fg_ratio:.2%}")
            st.metric("Image Size", f"{image_rgb.shape[1]} x {image_rgb.shape[0]}")
            st.metric("Mean Probability", f"{prob_map.mean():.4f}")
            st.metric("Max Probability", f"{prob_map.max():.4f}")

        # Download button for mask
        st.markdown("---")
        mask_pil = Image.fromarray(binary_mask)
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        st.download_button(
            label="Download Predicted Mask",
            data=buf.getvalue(),
            file_name=f"mask_{uploaded_file.name.split('.')[0]}.png",
            mime="image/png",
        )

    else:
        st.info("Please upload a biopsy image to begin segmentation analysis.")

        # Show example if test images exist
        test_dir = os.path.join(
            os.path.dirname(__file__),
            "Segmentation-20260326T063949Z-1-001", "Segmentation", "testing", "images",
        )
        if os.path.exists(test_dir):
            st.markdown("---")
            st.subheader("Example from Test Set")
            test_files = sorted(os.listdir(test_dir))[:3]
            cols = st.columns(len(test_files))
            for col, fname in zip(cols, test_files):
                img = cv2.imread(os.path.join(test_dir, fname), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                col.image(img_rgb, caption=fname, use_container_width=True)

    st.markdown("---")
    st.caption("For research and demonstration purposes only. Not for clinical use.")


if __name__ == "__main__":
    main()
