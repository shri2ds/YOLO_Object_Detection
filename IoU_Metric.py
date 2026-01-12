import torch

def inter_over_union(box_preds, box_labels, box_format="midpoint"):
    """
        Calculates IoU.

        Parameters:
        box_preds (tensor): Predictions of shape (BATCH_SIZE, 4)
        box_labels (tensor): Correct labels of shape (BATCH_SIZE, 4)
        box_format (str): "midpoint" (x,y,w,h) or "corners" (x1,y1,x2,y2)

        Returns:
        tensor: Intersection over union for all examples
    """

    # Convert to Corners (x1, y1, x2, y2) if needed

    if box_format == "midpoint":
        # x1 = x_c - w/2
        box1_x1 = box_preds[..., 0:1] - box_preds[..., 2:3] / 2
        box1_y1 = box_preds[..., 1:2] - box_preds[..., 3:4] / 2
        box1_x2 = box_preds[..., 0:1] + box_preds[..., 2:3] / 2
        box1_y2 = box_preds[..., 1:2] + box_preds[..., 3:4] / 2

        box2_x1 = box_labels[..., 0:1] - box_labels[..., 2:3] / 2
        box2_y1 = box_labels[..., 1:2] - box_labels[..., 3:4] / 2
        box2_x2 = box_labels[..., 0:1] + box_labels[..., 2:3] / 2
        box2_y2 = box_labels[..., 1:2] + box_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = box_preds[..., 0:1]
        box1_y1 = box_preds[..., 1:2]
        box1_x2 = box_preds[..., 2:3]
        box1_y2 = box_preds[..., 3:4]

        box2_x1 = box_labels[..., 0:1]
        box2_y1 = box_labels[..., 1:2]
        box2_x2 = box_labels[..., 2:3]
        box2_y2 = box_labels[..., 3:4]

    # Get the coordinates of the Intersection Box
    # Max of the top-left corners

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)

    # Min of the bottom-right corners
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Area of Intersection
    # clamp(0) ensures no negative area if they don't overlap
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Area of Union
    # Area = width * height
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection/(box1_area + box2_area - intersection + 1e-6)

if __name__ == "__main__":
    # Example: 2 predictions, 2 targets
    # Pred 1: Perfect match
    # Pred 2: No overlap

    # Format: (x_c, y_c, w, h) -> Normalized
    preds = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],  # Perfect Center
        [0.1, 0.1, 0.1, 0.1]  # Far top-left
    ])

    targets = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.9, 0.9, 0.1, 0.1]
    ])

    iou = inter_over_union(preds, targets, box_format="midpoint")
    print(f"IoU Scores:\n{iou}")

    if iou[0] > 0.99 and iou[1] < 0.01:
        print("✅ SUCCESS: IoU calculation is correct.")
    else:
        print("❌ FAIL: Check logic.")
