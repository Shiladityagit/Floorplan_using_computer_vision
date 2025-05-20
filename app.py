import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  


import streamlit as st
from config import MODEL_PATH, IMAGE_PATH
from yolo.detector import run_detection
from builder.floorplan_3d import create_3d_model
from builder.exporter import export_scene
from yolo.visualizer import draw_boxes
from utils.pdf_generator import create_pdf
from PIL import Image
import io

# Set page layout
st.set_page_config(layout="wide")
st.title("🏠 YOLOv8 Floorplan to 3D Model")

# Instructions
st.markdown("""
### 📋 Instructions:
1. Upload a **clear floor plan image** in `.jpg` or `.png` format.
2. Preferably **resize the image to 640×640** for best results (as model was trained on that).
3. After upload, detection will run, and you'll get:
   - 📦 A downloadable `.glb` 3D model
   - 📝 A downloadable PDF report of 2 pages

### 🔍 Sample Floor Plan Images (640×640)
""")

# Example images
col1, col2, col3 = st.columns(3)
with col1:
    st.image("assets/example1.jpg", caption="Example 1", use_container_width=True)
with col2:
    st.image("assets/example2.jpg", caption="Example 2", use_container_width=True)
with col3:
    st.image("assets/example3.jpg", caption="Example 3", use_container_width=True)

# File upload
uploaded_file = st.file_uploader("📤 Upload Floor Plan Image", type=["jpg", "png"])
if uploaded_file:
    img_bytes = uploaded_file.read()
    image_path = "assets/test_image.jpg"
    with open(image_path, "wb") as f:
        f.write(img_bytes)

    # Run YOLOv8 detection
    detections, class_names, original_img = run_detection(image_path, MODEL_PATH)

    # Generate 3D model and PDF report
    scene = create_3d_model(detections, class_names)
    glb_path = export_scene(scene, "final_floorplan.glb")
    annotated_img = draw_boxes(original_img, detections, class_names)
    pdf_path = create_pdf(annotated_img, detections, class_names)

    # Display results
    st.success("✅ 3D Model and PDF generated.")
    st.download_button("⬇️ Download GLB", open(glb_path, "rb"), "final_floorplan.glb")
    st.download_button("⬇️ Download PDF Report", open(pdf_path, "rb"), "detection_summary.pdf")
