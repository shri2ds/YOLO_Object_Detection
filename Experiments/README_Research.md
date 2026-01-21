# ⚠️ Research Archive

**Note:** This document tracks my learning journey. For the latest usage instructions and production code, please see the [Main README](../../README.md).
---

# 👁️ YOLO_Object_Detection

> **Building Real-Time Vision Systems from First Principles to Production.**


This repository documents the end-to-end engineering journey of mastering Object Detection. Unlike standard classification tasks (ResNet), this module focuses on the complexity of **localization**: predicting not just *what* an object is, but exactly *where* it is in spatial coordinates.

---

## 📂 Project Structure & Progress

| Topic | Engineering Concept | Status |
| :--- | :--- | :--- |
| **BBox Mechanics** | **YOLO Format Visualization.** Implemented the math to convert normalized YOLO coordinates `(x_c, y_c, w, h)` into pixel-perfect OpenCV bounding boxes. | ✅ Done |
| **IoU Metric** | **Intersection over Union.** Implementing the core metric used to measure overlap between predicted and ground truth boxes. | ✅ Done  |
| **The Grid System** | **YOLO Architecture v1.** Built the "Mini-YOLO" architecture from scratch, implementing the $S \times S \times (B \times 5 + C)$ tensor output structure. | ✅ Done |
| **The Data Pipeline** | **Custom Dataset Engineering.** Built a robust data pipeline to convert raw images and text annotations into the complex $S \times S \times 30$ YOLO Target Tensor. | ✅ Done |
| **Loss Function** | **The Brain.** Implemented the custom Multi-Part Loss with Vectorized Masking (`Experiments/PotHoleDetection/lossfunction.py`) to handle Class Imbalance ($\lambda_{coord}=5, \lambda_{noobj}=0.5$). | ✅ Done |
| **Model Architecture** | **The Body.** Implementation of the YOLO CNN backbone with 24 convolutional layers. | ✅ Done |
| **Training Loop** | **The Engine Room.** Implementing the training loop, loss calculation, and backpropagation. |  ✅ Done |
| **Inference** | **Real-world Test.** Running the model on video or webcam feeds. | ⏳ Pending |

---

## 🛠️ Technical Implementation Details

### The Physics of Bounding Boxes
**Goal:** Build the visualization engine to debug future model predictions.
* **The Challenge:** Neural Networks output normalized numbers (0.0 to 1.0) representing the *center* of an object. OpenCV draws rectangles using *pixel corners*.
* **The Solution:** Implemented the transformation logic:
    $$x_{corner} = (x_{center} - \frac{w}{2}) \times ImageWidth$$
    $$y_{corner} = (y_{center} - \frac{h}{2}) \times ImageHeight$$
* **Outcome:** A script (`Visualise_BBox.py`) that generates synthetic ground truth data and renders pixel-perfect bounding boxes.

### Intersection over Union (IoU)
**Goal:** Implement the "Scorecard" for object detection.
* **The Challenge:** Accuracy (Correct/Total) works for classification but fails for detection. We need to measure *how well* a predicted box overlaps with the truth.
* **The Solution:** Implemented `inter_over_union(BOX1, BOX2, Format)`:
    $$IoU = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$
* **Corner Cases Handled:**
    * No overlap (Intersection = 0).
    * Division by Zero safety (`+ 1e-6`).
 
| Index | Concept | Variable | Slice Code | Why? |
| :--- | :--- | :--- | :--- | :--- |
| **0** | X Center | $x_c$ | `0:1` | We need the center to calculate corners. |
| **1** | Y Center | $y_c$ | `1:2` | We need the center to calculate corners. |
| **2** | **Width** | $w$ | **`2:3`** | We need width to find Left/Right edges. |
| **3** | **Height** | $h$ | **`3:4`** | We need height to find Top/Bottom edges. |

### The YOLO Architecture (The Grid)
**Goal:** Understand how to predict multiple objects efficiently without sliding windows.
* **The Concept:** Divorced the idea of "Scanning" the image. Instead, implemented the **Grid Logic**:
    * Split the image into a fixed $S \times S$ grid (e.g., $7 \times 7$).
    * **Rule:** The grid cell containing the object's *center* is responsible for detecting it.
* **The Tensor:** The model output is not a list, but a 3D Tensor of shape $(S, S, 30)$.
    * **$S \times S$:** The spatial grid locations.
    * **30 Channels:** 20 Classes + 2 Boxes $\times$ (4 coords + 1 confidence).
* **The Implementation:** Built a custom `nn.Module` sequence:
    * **Backbone:** Convolutional layers to downsample spatial dimensions ($448 \to 7$).
    * **Head:** Linear layers to map features to the specific $(7 \times 7 \times 30)$ tensor shape.
 
### The Data Pipeline (Pothole Detection)
**Goal:** Engineering a memory-efficient data pipeline for single-class detection.

<p align="center">
  <img src="Experiments/assets/pothole_demo.png" alt="Pothole Detection Result" width="600"/>
  <br>
  <em>Figure 1: Custom implementation of successfully localizing a pothole using the Grid System.</em>
</p>

* **The Challenge:** Standard YOLO pipelines rely on sparse tensors ($7 \times 7 \times 30$) designed for 20 classes, wasting memory for single-class tasks.
* **The Solution (C=1 Optimization):** I re-engineered the tensor architecture to be strictly purpose-built for Potholes, reducing tensor depth by **80%** (from 30 to 6 channels).
    * **Optimized Map:** `[Class_Prob(0), Confidence(1), x(2), y(3), w(4), h(5)]`
* **Grid Mechanics:** Implemented custom `__getitem__` logic to perform real-time, relative coordinate encoding ($x_{cell}, y_{cell}$) rather than relying on pre-computed targets.
* **Validation:** Built a robust "Round-Trip" visualizer (see Figure 1) to prove the tensor math is reversible and lossless.
---

## 🚀 Future Roadmap
* **Data Engineering:** Automated annotation conversion (COCO to YOLO) and augmentation pipelines.
* **Production Deployment:** Exporting trained models to **ONNX** for high-performance inference.

---
*Created by Shridhar Bhandar as part of the Deep Learning Engineering Track.*
