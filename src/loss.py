"""
YOLO v1 Custom Loss Function
Implementation of the Multi-Part Loss described in the original YOLO paper.
"""

import torch
import torch.nn as nn

# --- IMPORT CHECK ---
try:
    from IoU_Metric import inter_over_union
except ImportError:
    # Fallback if the file is named differently
    from IoU_Metric import calculate_iou as inter_over_union


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
        self.lambda_coord = 5

    def forward(self, predictions, target):
        """
        Args:
            predictions: (Batch, S, S, C + B*5) -> (Batch, 7, 7, 11)
            target: (Batch, S, S, C + 5) -> (Batch, 7, 7, 6)
        """
        # Reshape to ensure we have the grid structure
        # (Batch, 7, 7, 11)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

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

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # --- 2. OBJECT LOSS (Confidence) ---
        # "You found the object, but how confident are you?"

        # Select the confidence of the winning box
        pred_box_conf = (
            best_box * box2_conf + (1 - best_box) * box1_conf
        )

        # Target Confidence is always 1 (if object exists)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box_conf),
            torch.flatten(exists_box * target[..., 1:2]),
        )

        # --- 3. NO OBJECT LOSS (The Silence) ---
        # "Punish the empty cells AND the loser box in the object cells"

        # Part A: Punish ALL boxes in Empty Cells
        # Flatten everything to (Batch * S * S, 1)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * box1_conf, start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 1:2], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * box2_conf, start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 1:2], start_dim=1),
        )

        # Part B: If an object exists, the "Loser" box is also No Object
        # If Best Box is 1 (Box 2 won), then Box 1 is a loser.
        no_object_loss += self.mse(
            torch.flatten(exists_box * best_box * box1_conf, start_dim=1),
            torch.flatten(torch.zeros_like(box1_conf), start_dim=1)
        )
        # If Best Box is 0 (Box 1 won), then Box 2 is a loser.
        no_object_loss += self.mse(
            torch.flatten(exists_box * (1 - best_box) * box2_conf, start_dim=1),
            torch.flatten(torch.zeros_like(box2_conf), start_dim=1)
        )

        # --- 4. CLASS LOSS ---
        # (N, S, S, C) -> (N*S*S, C)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2),
        )

        # =====================================================================
        # PHASE 3: TOTAL LOSS
        # =====================================================================

        loss = (
            self.lambda_coord * box_loss  # x, y, w, h
            + object_loss                 # Confidence (Winner)
            + self.lambda_noobj * no_object_loss # Confidence (Empty + Losers)
            + class_loss                  # Class Probability
        )

        return loss

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
