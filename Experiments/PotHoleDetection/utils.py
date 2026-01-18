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

