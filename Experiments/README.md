# Pothole Detection Experiment (YOLO v1)

This module implements a specialized version of the YOLO v1 data pipeline, optimized specifically for single-class object detection (Potholes).

## 📂 Project Structure
```text
📁 YOLO_Object_Detection
├── 📁 Experiments
│   └── 📁 PotHoleDetection         # Main application directory
│       ├── 📁 data                 # Raw images (.jpg) and YOLO labels (.txt)
│       ├── 📄 dataset.py           # PyTorch Dataset class & Grid Encoder (S=7, C=1)
│       ├── 📄 generate_csv.py      # Creates train/test mapping for DataLoader
│       └── 📄 dataset_download.py  # Kaggle API integration script
└── 📁 assets                       # Visualization outputs (e.g., pothole_demo.png)
```

### 🛠 Engineering Optimizations
---
#### 1. The C=1 Tensor Optimization
Standard YOLO v1 implementations (Pascal VOC) use C=20 classes, resulting in a target tensor of shape $(7, 7, 30)$.

* **Standard Map:** [20 Classes, 2x(Confidence, x, y, w, h)]
  * Note: Training target only uses 1 box slot.
  * Since we are strictly detecting Potholes, we optimized the tensor to remove dead weight
* **New Shape:** $(7, 7, 6)$
* **Reduction:** 80% reduction in memory footprint per sample.

#### 2. The Target Tensor Layout
For every grid cell $(i, j)$, the target vector (depth 6) is structured as:

| Index | Value | Description |
| :--- | :--- | :--- |
| **0** | **1** | **Class Probability.** Always 1 (since Pothole is the only class). |
| **1** | **1** | **Confidence.** Presence of an object center in this cell. |
| **2** | **$x_{cell}$** | Center X relative to the grid cell (0.0 - 1.0). |
| **3** | **$y_{cell}$** | Center Y relative to the grid cell (0.0 - 1.0). |
| **4** | **$w_{img}$** | Width relative to the entire image. |
| **5** | **$h_{img}$** | Height relative to the entire image. |

#### 3. Coordinate Transformation Logic
The `dataset.py` script handles the complex math of converting global normalized coordinates (0-1) into grid-relative coordinates.

**The Math:**
$$Col_{index} = \lfloor x_{global} \times S \rfloor$$
$$x_{cell} = (x_{global} \times S) - Col_{index}$$

This ensures that the model learns to predict *offsets* from the grid corner, which is easier for Gradient Descent than predicting global positions.
