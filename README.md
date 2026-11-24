# 🏠 2D-to-3D Architectural Floorplan Reconstruction

An end-to-end computer vision and procedural 3D geometry pipeline that transforms 2D raster architectural floorplans into scaled, interactive **glTF/GLB** 3D models and automated spatial diagnostic reports.

🔗 **Live Interactive Application:** [Launch Streamlit App](https://3gqegbyglucte4kna9adxu.streamlit.app/)

---

## 🏗️ System Architecture & Pipeline

The system bridges deep learning object detection with computational geometry through a four-stage inference pipeline:

```text
[2D Floorplan Input] ──▶ [YOLOv8s Object Detection] ──▶ [Spatial Coordinate Normalization] ──▶ [Trimesh 3D Extrusion] ──▶ [.GLB Mesh & PDF Report]
     (640×640 RGB)         (Rooms, Doors, Utilities)         (2D Bounding Box Extraction)         (Walls, Slabs, Doorways)      (Downloadable Outputs)
```

1. **Architectural Element Detection (`yolo/`):** Utilizes a fine-tuned **YOLOv8s** model (`11.1M` parameters, `28.5 GFLOPs`) optimized for `640×640` inputs to detect structural zones and openings (`Bedroom`, `Living Room`, `Dining Room`, `Kitchen`, `Toilet`, `Balcony`, `Foyer`, `Study`, `Utility`, and `Door`).
2. **Coordinate Transformation (`utils/`):** Parses raw bounding box predictions `[x_min, y_min, x_max, y_max]` and normalizes pixel coordinates into proportional 3D world-space dimensions.
3. **Procedural 3D Mesh Generation (`builder/`):** Constructs a scene graph using **Trimesh**, generating a base foundation slab, extruding vertical wall structures (`3.0m` standard height, `0.2m` thickness), and positioning semantic door meshes (`2.5m` height) before compiling the scene into a standard binary **glTF 2.0 (`.glb`)** file.
4. **Automated Diagnostic Reporting:** Generates a multi-page **PDF report** pairing visual detection overlays with structured bounding-box coordinate summaries.

---

## 📈 Model Performance & Evaluation

Trained on an annotated architectural floorplan dataset (`249` training / `26` validation images at `640×640` resolution), the **YOLOv8s** detector converged at **Epoch 84** with early stopping. Pruning noisy furniture labels to focus on core structural elements improved overall **mAP@50 from 79.2% to 94.8%**:

| Metric | Score / Specification |
| :--- | :--- |
| **Architecture** | YOLOv8s (`11.1M` params · `28.5 GFLOPs`) |
| **mAP@50** | **94.8%** |
| **mAP@50-95** | **78.2%** |
| **Precision** | **91.8%** |
| **Recall** | **90.2%** |
| **Inference Speed** | **8.6 ms / image** (Tesla P100 GPU) |

---

## 📂 Repository Structure

```text
Floorplan_using_computer_vision/
├── Notebooks/          # Model training, dataset preprocessing, and evaluation experiments
├── assets/             # Pre-processed 640×640 benchmark floorplans for one-click testing
├── builder/            # Procedural 3D mesh generation and Trimesh scene assembly logic
├── utils/              # Image preprocessing, coordinate scaling, and PDF report utilities
├── yolo/               # Trained YOLOv8s model weights and inference wrappers
├── app.py              # Main Streamlit web application entry point
├── config.py           # Global hyperparameters, class mappings, and 3D extrusion constants
├── packages.txt        # System-level dependencies for cloud deployment
└── requirements.txt    # Python package dependencies
```

---

## ⚡ Quickstart & Local Installation

### Prerequisites
* Python 3.9 or higher
* `pip` package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shiladityagit/Floorplan_using_computer_vision.git
   cd Floorplan_using_computer_vision
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Usage & Outputs

1. **Input:** Upload a clear `.jpg` or `.png` floorplan image (or select one of the built-in `640×640` benchmark samples).
2. **Inference:** The pipeline automatically runs YOLOv8s detection and builds the 3D scene graph.
3. **Export:**
   * **`.glb` 3D Model:** Compatible with Blender, Three.js, Windows 3D Viewer, and web-based glTF viewers.
   * **2-Page PDF Report:** Contains the annotated detection overlay (Page 1) and extracted spatial coordinates (Page 2).

---

## 🗺️ Future Roadmap

* [ ] **Boolean Wall Cutouts & Merging:** Deduplicate shared interior walls and perform CSG boolean subtraction for realistic door and window openings.
* [ ] **Instance Segmentation (`YOLOv8-seg`):** Transition from axis-aligned bounding boxes to polygon masks to support non-rectangular and L-shaped room geometries.
* [ ] **OCR Dimension Parsing:** Extract textual room dimensions from floorplans to calibrate 3D meshes to exact real-world metric/imperial scales.
