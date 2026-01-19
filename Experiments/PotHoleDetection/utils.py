import torch
import numpy as np
import os
import sys, 
# To import IoU_Metric functionality
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from IoU_Metric import inter_over_union


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
    
    

