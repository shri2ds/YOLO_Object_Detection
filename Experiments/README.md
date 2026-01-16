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
│       ├── 📄 model.py             # YOLOv1 CNN Architecture (From Scratch)   
│       ├── 📄 dataset_download.py  # Kaggle API integration script
│       └── README.md               # Detailed Experiment Documentation
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

#### 4. The Loss Function Architecture (The Engine)
**Goal:** Penalize the model differently for 4 specific types of errors using a custom Multi-Part Loss function.

We cannot use standard `CrossEntropy` or `MSE` directly because the image is mostly empty background. If we treated all pixels equally, the model would learn to predict "No Object" everywhere and achieve 94% accuracy but 0% utility.

##### The 4-Part Loss Logic
We calculate the Total Loss ($L$) as the sum of four components:

$$L = \lambda_{coord} L_{coord} + L_{obj} + \lambda_{noobj} L_{noobj} + L_{class}$$

| Component | Math | Purpose |
| :--- | :--- | :--- |
| **1. Coordinate Loss** | $\sum \lambda_{coord} (x - \hat{x})^2 + (\sqrt{w} - \sqrt{\hat{w}})^2$ | Forces the box to fit the object tightly. **Note:** We use $\sqrt{w}$ to penalize errors in small boxes more heavily than large boxes. |
| **2. Object Loss** | $\sum (C_{true} - C_{pred})^2$ | Punishes the model if it says "No Object" when there IS a pothole. |
| **3. No-Object Loss** | $\sum \lambda_{noobj} (0 - C_{pred})^2$ | The "Silence" penalty. Punishes the 40+ empty grid cells if they try to hallucinate an object. |
| **4. Class Loss** | $\sum (p_c - \hat{p}_c)^2$ | Ensures correct classification (Trivial for Potholes as $C=1$, but architectural standard). |

##### **Key Engineering: Vectorized Masking**
Instead of using slow Python `for` loops to check every grid cell, we implemented **Tensor Broadcasting** to calculate loss in parallel.

* **The Challenge:** We only want to punish the *one* specific bounding box predictor (out of $B=2$) that is "Responsible" (Best IoU) for the object.
* **The Solution:**
    1.  **Calculated IoU** for both predictors vs Ground Truth.
    2.  **Created a Mask:** `best_box` tensor (Batch, 7, 7, 1) containing `0` or `1`.
    3.  **Broadcasted Multiplication:** Used this mask to zero-out the gradient for the "Loser" box, ensuring only the "Winner" learns from the coordinate error.

```python
# Engineering Snippet: Vectorized Selection
# (Batch, 7, 7, 1) * (Batch, 7, 7, 4) -> Zeroes out the loser's coordinates
box_predictions = exists_box * (
    best_box * box2_coords + (1 - best_box) * box1_coords
)
```

#### 5. The Model Architecture (The Body)
**Goal:** Implement the YOLO v1 CNN architecture from scratch, optimized for Pothole Detection.

Instead of defining 24 separate layers manually, we implemented a **Configuration-Driven Architecture**. The entire model structure is defined in a simple list, making it highly modular and easy to modify for experiments.

##### **A. The Configuration Pattern**

The architecture is defined in `model.py` using a list of tuples:
```python
# (kernel_size, filters, stride, padding)
architecture_config = [
    (7, 64, 2, 3),   # "Wide Angle" start (448 -> 224)
    "M",             # MaxPool (2x zoom out)
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # Repeated Blocks
    ...
]
```
- **Tuples**: Define standard Convolutional layers.
- **"M"**: Defines MaxPool layers (reducing spatial size by half).
- **Lists `[ ... ]`**: Define repeating blocks (e.g., repeat this sub-structure 4 times).

##### B. The Bottleneck Strategy (1x1 Convolutions)

To keep the model fast enough for real-time inference (45 FPS), we use **1x1 Convolutions** to reduce parameter count before performing heavy operations.

**The Problem:**
Running a `3 × 3` filter on 512 channels is computationally expensive (~2.3M params per layer).

**The Solution:**
- **Squeeze**: Use a `1 × 1` filter to reduce depth from **512 → 256**.
- **Process**: Run the `3 × 3` filter on the smaller depth of **256**.
- **Expand**: The network expands the features back up in later layers.

##### C. Final Output Shape

The model transforms the input image into our specific Grid Prediction Tensor:
```text
Input: (Batch, 3, 448, 448)  ──[ CNN ]──>  Output: (Batch, 7, 7, 11)
```
- `7 × 7`: The Split Grid (`S`).
- `11`: The Depth Vector (`C=1` Pothole Class + `B=2` Boxes `×` `5` coords).
