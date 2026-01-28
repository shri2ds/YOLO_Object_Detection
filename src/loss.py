"""
YOLO v1 Custom Loss Function
Implementation of the Multi-Part Loss described in the original YOLO paper.
"""

import torch
import torch.nn as nn

from src.utils.box_ops import inter_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        # Architecture Hyperparameters
        self.S = S
        self.B = B
        self.C = C

        # The "Levers" (Lambdas)
        self.lambda_noobj = 0.5
        self.lambda_coord = 10

    def forward(self, predictions, target):
        """
        Args:
            predictions: (Batch, S, S, C + B*5) -> (Batch, 7, 7, 11)
            target: (Batch, S, S, C + 5) -> (Batch, 7, 7, 6)
        """
        # Reshape to ensure we have the grid structure
        # (Batch, 7, 7, 11)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Force all model outputs to be between 0 and 1
        predictions = torch.sigmoid(predictions)

        # =====================================================================
        # PHASE 1: PREPARATION & RESPONSIBILITY (Finding the Winner)
        # =====================================================================

        # EXTRACT BOXES (C=1 case)
        # Box 1: Indices 2,3,4,5
        box1_coords = predictions[..., self.C + 1 : self.C + 5]

        # FIX: Slice to keep dimension -> (Batch, 7, 7, 1)
        box1_conf   = predictions[..., self.C : self.C + 1]

        # Box 2: Indices 7,8,9,10
        box2_coords = predictions[..., self.C + 6 : self.C + 10]

        # FIX: Slice to keep dimension -> (Batch, 7, 7, 1)
        box2_conf   = predictions[..., self.C + 5 : self.C + 6]

        # Ground Truth: Indices 2,3,4,5
        true_box = target[..., 2:6]

        # CALCULATE IOU
        ious1 = inter_over_union(box1_coords, true_box)
        ious2 = inter_over_union(box2_coords, true_box)

        # Stack them: (Batch, S, S, 2)
        ious = torch.cat([ious1.unsqueeze(0), ious2.unsqueeze(0)], dim=0)

        # Find the winner (0 or 1)
        # best_box is indices (0 for box1, 1 for box2) -> Shape (Batch, 7, 7, 1)
        iou_maxes, best_box = torch.max(ious, dim=0)

        # EXISTS MASK: 1 if object exists in cell i,j, 0 otherwise
        # Shape: (Batch, S, S, 1)
        exists_box = target[..., 1].unsqueeze(3)

        # =====================================================================
        # PHASE 2: LOSS COMPONENTS
        # =====================================================================

        # --- 1. COORDINATE LOSS (Box Location) ---
        # We only punish the box that is "Responsible" (Best IoU)

        # Create a tensor of the PREDICTED boxes based on which one won
        # if best_box=0 -> box1, if best_box=1 -> box2
        box_predictions = exists_box * (
            (
                best_box * box2_coords + (1 - best_box) * box1_coords
            )
        )

        box_targets = exists_box * true_box

        # Square Root Logic (YOLO v1 Feature)
        # Take sqrt of w, h to penalize small errors in small boxes heavily
        # We add 1e-6 to avoid NaN (derivative of sqrt(0) is infinity)
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # Calculate box loss using winning box coordinates
        box_loss = self.lambda_coord * torch.sum(
            exists_box * (
                    (box_predictions[..., 0:1] - box_targets[..., 0:1]) ** 2 +
                    (box_predictions[..., 1:2] - box_targets[..., 1:2]) ** 2 +
                    (box_predictions[..., 2:3] - box_targets[..., 2:3]) ** 2 +
                    (box_predictions[..., 3:4] - box_targets[..., 3:4]) ** 2
            )
        )

        # Object loss: confidence of winning box vs target confidence (always 1)
        pred_conf = best_box * box2_conf + (1 - best_box) * box1_conf
        object_loss = torch.sum(exists_box * (pred_conf - target[..., 1:2]) ** 2)

        # No object loss: confidence in empty cells for both boxes + loser box in object cells
        no_object_loss = self.lambda_noobj * torch.sum(
            (1 - exists_box) * (box1_conf ** 2 + box2_conf ** 2)
        )
        
        # Penalize loser box in cells with objects
        # If best_box=0 (box1 wins), penalize box2; if best_box=1 (box2 wins), penalize box1
        no_object_loss += self.lambda_noobj * torch.sum(
            exists_box * (best_box * box1_conf ** 2 + (1 - best_box) * box2_conf ** 2)
        )

        # Class loss: predicted class vs target class
        class_loss = torch.sum(
            exists_box * (predictions[..., 0:1] - target[..., 0:1]) ** 2)

        return box_loss + object_loss + no_object_loss + class_loss

# --- UPDATED MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Define Architecture for Potholes
    S, B, C = 7, 2, 1

    # 1. Create Dummy Data
    # Prediction Shape: (Batch=2, S=7, S=7, Depth=11)
    predictions = torch.rand((2, S, S, C + B * 5))

    # Target Shape: (Batch=2, S=7, S=7, Depth=6)
    target = torch.rand((2, S, S, C + 5))

    # 2. Initialize Loss Function
    loss_fn = YoloLoss(S=S, B=B, C=C)

    # 3. Calculate Loss
    print("------------------------------------------------")
    print("🛠  Starting Loss Calculation Test...")
    print(f"   Architecture: S={S}, B={B}, C={C}")
    print(f"   Input Shapes: Pred={predictions.shape}, Target={target.shape}")

    total_loss = loss_fn(predictions, target)

    print("------------------------------------------------")
    # This line was crashing because total_loss was None.
    # Now it should be a Tensor.
    print(f"✅ FINAL LOSS CALCULATED: {total_loss.item()}")
    print("------------------------------------------------")
