# 👁️ YOLO_Object_Detection

> **Building Real-Time Vision Systems from First Principles to Production.**

This repository documents the end-to-end engineering journey of mastering Object Detection. Unlike standard classification tasks (ResNet), this module focuses on the complexity of **localization**: predicting not just *what* an object is, but exactly *where* it is in spatial coordinates.

---

## 📂 Project Structure & Progress

| Topic | Engineering Concept | Status |
| :--- | :--- | :--- |
| **BBox Mechanics** | **YOLO Format Visualization.** Implemented the math to convert normalized YOLO coordinates `(x_c, y_c, w, h)` into pixel-perfect OpenCV bounding boxes. | ✅ Done |
| **IoU Metric** | **Intersection over Union.** Implementing the core metric used to measure overlap between predicted and ground truth boxes. | ⏳ Pending |
| **YOLO Architecture** | **The Grid System.** Understanding how YOLO divides images into grids to predict objects in parallel. | ⏳ Pending |
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

---

## 🚀 Future Roadmap
* **Data Engineering:** Automated annotation conversion (COCO to YOLO) and augmentation pipelines.
* **Metric Implementation:** Manual implementation of Non-Maximum Suppression (NMS) to filter duplicate boxes.
* **Production Deployment:** Exporting trained models to **ONNX** for high-performance inference.

---
*Created by Shridhar Bhandar as part of the Deep Learning Engineering Track.*
