import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Create a Blank Image (Black Background)
# Height=400, Width=400, Channels=3 (RGB)
image_size = 400
image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

# 2. Define "Ground Truth" Boxes (YOLO Format: class, x_c, y_c, w, h)
# Normalized coordinates (0.0 to 1.0)
# Box 1: A small box in the center (Class 0)
# Box 2: A large box covering top-left (Class 1)
labels = [
    [0, 0.5, 0.5, 0.2, 0.2],
    [1, 0.25, 0.25, 0.4, 0.4]
]
class_names = ["Target", "Zone"]

# 3. The Visualization Logic
def draw_yolo_boxes(image, labels, names):
    h, w, _ = image.shape
    image_copy = image.copy()

    for box in labels:
        class_id, x_c, y_c, box_w, box_h = box

        # Un-normalize: Convert 0-1 to pixels
        # x_center * width of image
        x_pixel = int(x_c * w)
        y_pixel = int(y_c * h)
        w_pixel = int(box_w * w)
        h_pixel = int(box_h * h)

        # Calculate Top-Left Corner (for OpenCV drawing)
        # x_top_left = center - width/2
        x1 = int(x_pixel - w_pixel / 2)
        y1 = int(y_pixel - h_pixel / 2)
        x2 = int(x_pixel + w_pixel / 2)
        y2 = int(y_pixel + h_pixel / 2)

        # Draw Rectangle
        # Color: Green (0, 255, 0) for class 0, Red for class 1
        color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
        cv2.rectangle(image_copy, (x1,y1), (x2,y2), color, 2)

        # Add Label Text
        text = names[int(class_id)]
        cv2.putText(image_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image_copy

# Run Visualization
annotated_img = draw_yolo_boxes(image, labels, class_names)

# Display using Matplotlib (OpenCV uses BGR, Matplotlib uses RGB)
plt.imshow(annotated_img)
plt.title("YOLO Format Visualization")
plt.axis("off")
plt.show()
