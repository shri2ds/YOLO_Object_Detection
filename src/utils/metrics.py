import torch
import numpy as np

from .box_ops import inter_over_union


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Does Non Max Suppression given a list of bounding boxes
    Parameters:
        bboxes: list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold: threshold where predicted bboxes is correct
        threshold: threshold to remove predicted bboxes (independent of IoU)
    """
    assert type(bboxes) == list

    # 1. Filter out boxes with low confidence (e.g., probability < 0.5)
    bboxes = [box for box in bboxes if box[1] > threshold]

    # 2. Sort by confidence (highest first)
    bboxes = sorted(bboxes, reverse=True, key=lambda x:x[1])
    bboxes_after_nms = []

    while bboxes:
        choosen_box = bboxes.pop(0)

        # 3. Filter out other boxes that overlap too much with the chosen box
        bboxes = [ box for box in bboxes if box[0] != choosen_box or
                   inter_over_union(torch.tensor(choosen_box[2:]), torch.tensor(box[2:]),
                                                box_format=box_format) < iou_threshold]

        bboxes_after_nms.append(choosen_box)

    return bboxes_after_nms

def cellboxes_to_boxes(out, S=7):
    """
    Converts the raw (batch, S, S, 30) tensor into a list of bounding boxes
    relative to the entire image, not just the cell.
    """
    device = out.device
    predictions = out.reshape(out.shape[0], S, S, 11)

    # Extract variables for both boxes
    # Box 1: Index 1..5
    confidence = predictions[..., 1].unsqueeze(3)
    bboxes1 = predictions[..., 2:6]

    # Box 2: Index 6..10
    confidence2 = predictions[..., 6].unsqueeze(3)
    bboxes2 = predictions[..., 7:11]

    # Keep the box with higher confidence
    scores =  torch.cat((confidence, confidence2), dim=-1)
    best_box = scores.argmax(-1).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + bboxes2 * best_box

    # Convert Cell coordinates (0..1) to Image coordinates (0..7)
    cell_indices = torch.arange(S).repeat(out.shape[0], S, 1).unsqueeze(-1).to(device)

    # X = (x_cell + cell_index) / S
    x = 1 / S * (best_boxes[..., 0:1] + cell_indices)
    # Y = (y_cell + row_index) / S
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))

    w_h = 1 / S * best_boxes[..., 2:4]

    # Concatenate Class + Score + X + Y + W + H
    converted_bboxes = torch.cat((predictions[..., 0:1], scores.max(-1)[0].unsqueeze(-1), x, y, w_h), dim=-1)

    return converted_bboxes.reshape(out.shape[0], S * S, -1).tolist()

def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=1
):
    """
    Calculates mAP based on the Intersection over Union (IoU) threshold.
    pred_boxes: list of [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    true_boxes: list of [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    """
    # List to store Average Precision for each class
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Filter for specific class
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        print(f"  [mAP Debug] Class {c}: {len(detections)} detections, {len(ground_truths)} ground truths")

        amount_bboxes = {}
        for gt in ground_truths:
            image_idx = gt[0]
            if image_idx not in amount_bboxes:
                amount_bboxes[image_idx] = 0
            amount_bboxes[image_idx] += 1

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        # If no ground truths for this class, skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = inter_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # Add AP for this class to the list
        average_precisions.append(torch.trapz(precisions, recalls))

    # --- SAFETY CHECK & RETURN ---
    if len(average_precisions) == 0:
        return torch.tensor(0.0)  # Safety for empty lists

    # THIS WAS LIKELY MISSING:
    return sum(average_precisions) / len(average_precisions)
