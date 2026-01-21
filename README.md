# 🛣️ YOLOv1 Pothole Detection (Production)

A PyTorch implementation of YOLOv1 built for production deployment. This repository contains the modularized source code (`src`), training pipelines, and API serving infrastructure.

> **Note:** For research history, legacy notebooks, and debugging logs, see the [Experiments Directory](./Experiments/README_Research.md).

## 📂 Project Structure

The project follows a package-based architecture separating data, logic, and experiments.

```text
YOLO/
├── src/               # 🧠 Core Library
│   ├── model.py       # YOLOv1 Architecture (CNNBlock, Yolov1)
│   ├── loss.py        # Custom Loss (Coord, Object, NoObj, Class)
│   ├── dataset.py     # Pytorch Dataset & Transforms
│   └── utils/         # Metrics (IoU, NMS, mAP) & Visualization
├── data/              # 💾 Data Storage (Ignored by Git)
│   ├── images/        # Raw JPG/PNG images
│   ├── labels/        # YOLO format text labels
│   └── processed/     # CSV manifest files (train.csv, test.csv)
├── Experiments/       # ⚗️ Research Sandbox (Legacy code & Notebooks)
├── train.py           # 🚀 Training Entry Point
├── config.py          # ⚙️ Hyperparameters & Path Configuration
└── app/               # 🔌 FastAPI Service (Coming Soon)

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Adjust hyperparameters in `config.py` (Learning Rate, Epochs, Batch Size).

### 3. Training
To start training from scratch using the engine in src/:
```bash
python train.py
```

### 4. Inference (API)
```bash
uvicorn app.main:app --reload
```
---

## 📊 Performance
- **Model:** YOLOv1 (ResNet-like Backbone)
- **Input:** 448x448 RGB Images
- **Current mAP:** 66% (Preliminary result on Debug Subset)
- **Classes:** 1 (Pothole)

## 🛠️ Components
- **Dataset:** Custom Pothole Dataset (~1000 images)
- **Loss Function:** Multi-part loss (Localization penalty $\lambda_{coord}=5$)
- **Optimization:** Adam
