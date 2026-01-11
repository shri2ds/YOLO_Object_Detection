# 👁️ YOLO_Object_Detection

> **Building Real-Time Vision Systems from First Principles to Production.**

This repository documents the end-to-end engineering journey of mastering Object Detection. Unlike standard classification tasks (ResNet), this module focuses on the complexity of **localization**: predicting not just *what* an object is, but exactly *where* it is in spatial coordinates.

---

## 📂 Project Structure & Progress

| Topic | Engineering Concept | Status |
| :--- | :--- | :--- |
| **BBox Mechanics** | **YOLO Format Visualization.** Implemented the math to convert normalized YOLO coordinates `(x_c, y_c, w, h)` into pixel-perfect OpenCV bounding boxes. | ✅ Done |
| **IoU Metric** | **Intersection over Union.** Implementing the core metric used to measure overlap between predicted and ground truth boxes. | ⏳ Pending |
| **The Grid System** | **YOLO Architecture v1.** Built the "Mini-YOLO" architecture from scratch, implementing the $S \times S \times (B \times 5 + C)$ tensor output structure. | ✅ Done |
| **Custom Training** | **Fine-Tuning YOLO.** Training a custom model on a real-world dataset (e.g., PPE/Potholes). | ⏳ Pending |

---

## 🛠️ Technical Implementation Details

### The Physics of Bounding Boxes
**Goal:** Build the visualization engine to debug future model predictions.
* **The Challenge:** Neural Networks output normalized numbers (0.0 to 1.0) representing the *center* of an object. OpenCV draws rectangles using *pixel corners*.
* **The Solution:** Implemented the transformation logic:
    $$x_{corner} = (x_{center} - \frac{w}{2}) \times ImageWidth$$
    $$y_{corner} = (y_{center} - \frac{h}{2}) \times ImageHeight$$
* **Outcome:** A script (`Visualise_BBox.py`) that generates synthetic ground truth data and renders pixel-perfect bounding boxes.

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

---

## 🚀 Future Roadmap
* **Data Engineering:** Automated annotation conversion (COCO to YOLO) and augmentation pipelines.
* **Metric Implementation:** Manual implementation of Non-Maximum Suppression (NMS) to filter duplicate boxes.
* **Production Deployment:** Exporting trained models to **ONNX** for high-performance inference.

---
*Created by Shridhar Bhandar as part of the Deep Learning Engineering Track.*
