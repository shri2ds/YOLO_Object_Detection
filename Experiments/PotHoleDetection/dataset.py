import kaggle
import torch
import os
import pandas as pd
from PIL import Image


class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        """
        Args:
            csv_file: A CSV with columns [img_filename, label_filename] (or we can scan folders)
            img_dir: Path to images
            label_dir: Path to text labels
            S: Grid size (7)
            B: Number of boxes (2)
            C: Number of classes (20)
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. Load Image and Label path
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])

        image = Image.open(img_path).convert("RGB")

        # 2. Parse Label File
        # Format: [class, x, y, w, h] (Line by line)
        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                # specific to PASCAL VOC / Standard YOLO format
                class_label, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in line.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, w, h])

        # 3. Apply Transformations (Augmentations) - OPTIONAL for now
        # if self.transform:
        #     image, boxes = self.transform(image, boxes)

        # 4. ENCODE TO GRID (The Hard Part)
        # We need a target tensor: [S, S, C + 5*B]
        # But wait! In the loss function, we only calculate loss for the "responsible" box.
        # So the target shape for training is simplified: [S, S, C + 5]
        # We store: [Class_Probs(C), x, y, w, h, confidence(1)]
        # Actually standard YOLOv1 Implementation target: [S, S, 25] if C=20.
        # Format per cell: [c_1...c_20, confidence, x, y, w, h]

        target = torch.zeros((self.S, self.S, self.C + 5))

        for box in boxes:
            class_label, x, y, w, h = box
            class_label = int(class_label)

            # i, j represents the ROW and COL of the grid cell
            # x, y are normalized (0-1).
            # If S=7 and x=0.5, then S*x = 3.5. Cell is index 3.
            i, j = int(self.S * y), int(self.S * x)

            # Calculates coordinates RELATIVE to the cell
            # If x=0.5 (Global), cell starts at 0.42 (3/7).
            # relative x = 3.5 - 3 = 0.5 (Center of the cell)
            x_cell = self.S * x - j
            y_cell = self.S * y - i

            # Width and Height are relative to the ENTIRE image, so we keep them as w, h
            # (Some implementations normalize w/h relative to cell, but standard YOLOv1 keeps image-relative)
            width_cell, height_cell = (
                w * self.S,
                h * self.S,
            )

            # Check if this cell already has an object (One object per cell limitation in YOLOv1)
            if target[i, j, 20] == 0:  # Index 20 is the Confidence Score
                # Set that there is an object
                target[i, j, 20] = 1

                # Set Box coordinates
                target[i, j, 21:25] = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                # Set One-Hot Encoding for Class
                target[i, j, class_label] = 1

        return image, target


# ... (Keep your YOLODataset class exactly as it is) ...

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # boxes is a list of [x, y, w, h, class]
    # These are normalized (0-1). We need to scale to image size.
    img_width, img_height = image.size[0], image.size[1]  # PIL image size

    for box in boxes:
        box_x, box_y, box_w, box_h = box

        # Convert Center (x,y) to Top-Left (x,y) for plotting
        # x = (x_c - w/2) * width
        lower_left_x = (box_x - box_w / 2) * img_width
        lower_left_y = (box_y - box_h / 2) * img_height

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (lower_left_x, lower_left_y),
            box_w * img_width,
            box_h * img_height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def convert_cell_boxes(predictions, S=7):
    """
    Converts the Grid Tensor (7, 7, 30) back to Global Coordinates (x, y, w, h).
    This reverses the logic in __getitem__.
    """
    # predictions shape: (7, 7, 30)
    # We only care about cells where Confidence (Index 20) == 1

    boxes = []

    # Iterate over grid cells
    for i in range(S):  # Rows
        for j in range(S):  # Cols
            # Check confidence score (Index 20)
            if predictions[i, j, 20] == 1:
                # We found an object!
                # Extract relative coordinates
                # Format: [C1..C20, Conf, x, y, w, h]
                x_cell = predictions[i, j, 21]
                y_cell = predictions[i, j, 22]
                w_cell = predictions[i, j, 23]
                h_cell = predictions[i, j, 24]

                # REVERSE THE MATH
                # x_cell = S * x - j  --->  x = (x_cell + j) / S
                x = (x_cell + j) / S
                y = (y_cell + i) / S

                # w, h were just stored directly (relative to image)
                # But in our code we stored them as w*S. Wait!
                # Let's check your __getitem__:
                # You did: width_cell = w * self.S
                # So we reverse: w = width_cell / S
                w = w_cell / S
                h = h_cell / S

                boxes.append([x, y, w, h])

    return boxes


if __name__ == "__main__":
    # 1. Setup paths
    # Ensure these point to where generate_csv.py put the files
    CSV_FILE = "data/train.csv"
    IMG_DIR = ""
    LABEL_DIR = ""

    # 2. Load Dataset
    dataset = YOLODataset(
        csv_file=CSV_FILE,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=7, B=2, C=20
    )

    print(f"Dataset Length: {len(dataset)}")

    # 3. Get one specific image (e.g., index 10)
    img, target = dataset[10]

    print(f"Image Shape: {img.size}")
    print(f"Target Tensor Shape: {target.shape}")  # Should be (7, 7, 25) or (7, 7, 30)

    # 4. Decode the Grid Tensor back to Boxes
    detected_boxes = convert_cell_boxes(target, S=7)

    print(f"Decoded {len(detected_boxes)} box(es) from the grid.")
    print(f"Box Coords: {detected_boxes}")

    # 5. Visualize
    plot_image(img, detected_boxes)