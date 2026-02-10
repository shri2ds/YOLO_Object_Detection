import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.model import YOLO_Architecture
from src.dataset import YOLODataset
from src.loss import YoloLoss
from src.utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    load_checkpoint,
    mean_average_precision,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_FILE = "checkpoint_best.pth.tar"

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Box format: [class_pred, prob_score, x_mid, y_mid, w, h]
    for box in boxes:
        box = box[2:]   # Drop class/score, keep coordinates
        assert len(box) == 4, "Got more values than [x, y, w, h]" 

        # Unpack relative coordinates (0..1)
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect_w =  box[2]
        rect_h =  box[3]

        # Draw Rectangle (Convert relative -> absolute pixels)
        rect = patches.Rectangle((upper_left_x * width, upper_left_y * height), rect_w * width, rect_h * height, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

    plt.show()

def main():
    # Setup
    model = YOLO_Architecture(split_size=7, num_of_boxes=2, num_of_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # Load Checkpoint
    print(f"Loading {CHECKPOINT_FILE}...")
    load_checkpoint(torch.load(CHECKPOINT_FILE, map_location=DEVICE), model, optimizer)
    model.eval()  # Turnoff the dropouts

    # Load Data (Just reuse train.csv for sanity check)
    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])
    dataset = YOLODataset(
        csv_file="data/processed/test.csv", img_dir="", label_dir="", transform=transform
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Inference Loop
    print("Starting Inference... ")
    for x, y in loader:
        x = x.to(DEVICE)

        with torch.no_grad():
            out = model(x)

            # Convert grid -> boxes
            bboxes = cellboxes_to_boxes(out)

            # Clean up boxes (NMS)
            bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.3, box_format="midpoint")   # Only keep boxes with >50% confidence

            # Plot
            plot_image(x[0].permute(1, 2, 0).to("cpu"), bboxes)
            input("Press Enter to see next image or exit...")


if __name__ == "__main__":
    main()     
    
