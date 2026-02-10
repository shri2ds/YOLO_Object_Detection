#!/usr/bin/env python3

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from src.model import YOLO_Architecture
from src.dataset import YOLODataset
from src.loss import YoloLoss
from src.utils import (
    save_checkpoint,
    load_checkpoint,
    cellboxes_to_boxes,
    non_max_suppression,
    mean_average_precision,
)

# --- Hyperparameters (from config.py) ---
LEARNING_RATE = config.LEARNING_RATE
DEVICE = config.DEVICE
BATCH_SIZE = config.BATCH_SIZE
WEIGHT_DECAY = config.WEIGHT_DECAY
EPOCHS = config.EPOCHS
NUM_WORKERS = config.NUM_WORKERS
PIN_MEMORY = config.PIN_MEMORY
LOAD_MODEL = config.LOAD_MODEL
LOAD_MODEL_FILE = config.CHECKPOINT_FILE

# --- Model Configuration ---
# Using custom YOLO_Architecture (no pretrained model)


# --- Image Transforms ---
# NO augmentation for aggressive overfitting
train_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x,y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # 1. Forward Pass
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        # 2. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. Update Progress Bar
        loop.set_postfix(loss=loss.item())

    epoch_mean_loss = sum(mean_loss) / len(mean_loss)
    print(f"Mean loss was {epoch_mean_loss}")
    return epoch_mean_loss

def evaluate_map(model, loader, iou_threshold=0.5, conf_threshold=0.01):
    """Evaluate mAP on a dataset"""
    model.eval()
    all_pred_boxes = []
    all_true_boxes = []
    total_raw_predictions = 0
    total_after_nms = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Evaluating mAP")):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Get predictions
            predictions = model(x)
            batch_size = x.shape[0]
            
            # Convert predictions to boxes
            bboxes = cellboxes_to_boxes(predictions)
            
            # Ground truth is already in grid format [batch, S, S, C+B*5]
            # Extract ground truth boxes directly from the grid
            S = y.shape[1]  # Grid size (7)
            
            for idx in range(batch_size):
                total_raw_predictions += len(bboxes[idx])
                
                # Apply NMS to predictions
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=iou_threshold,
                    threshold=conf_threshold,
                    box_format="midpoint",
                )
                
                total_after_nms += len(nms_boxes)
                
                # Add image index to each box (force class id = 0 since single-class)
                for box in nms_boxes:
                    image_idx = batch_idx * batch_size + idx
                    conf_score = box[1]
                    all_pred_boxes.append([
                        image_idx,
                        0,              # single pothole class
                        conf_score,
                        box[2], box[3], box[4], box[5]
                    ])
                
                # Extract ground truth boxes from grid
                for i in range(S):
                    for j in range(S):
                        cell = y[idx, i, j]
                        # Check if there's an object in this cell (class confidence > 0)
                        if cell[0] > 0:  # class probability
                            # Extract box coordinates (using first box)
                            confidence = cell[1]
                            if confidence > 0:
                                x_cell, y_cell, w, h = cell[2:6]
                                # Convert from cell-relative to image-relative
                                x_img = (j + x_cell) / S
                                y_img = (i + y_cell) / S
                                w_img = w
                                h_img = h
                                # Format: [train_idx, class, prob, x, y, w, h]
                                all_true_boxes.append([
                                    batch_idx * batch_size + idx,
                                    0,      # class (always 0 for single class)
                                    1.0,    # ground truth confidence
                                    x_img, y_img, w_img, h_img
                                ])
    
    model.train()
    
    # Debug info
    print(f"  [Debug] Raw predictions: {total_raw_predictions}, After NMS: {total_after_nms}, Final detections: {len(all_pred_boxes)}")
    print(f"  [Debug] Ground truths: {len(all_true_boxes)}")
    
    # Sample a few predictions to see confidence scores
    if len(all_pred_boxes) > 0:
        sample_preds = all_pred_boxes[:3]
        print(f"  [Debug] Sample predictions (first 3): {sample_preds}")
    
    # Calculate mAP
    if len(all_pred_boxes) == 0 or len(all_true_boxes) == 0:
        print(f"  [Debug] Returning 0.0 mAP (no predictions or ground truths)")
        return 0.0
    
    map_val = mean_average_precision(
        all_pred_boxes,
        all_true_boxes,
        iou_threshold=iou_threshold,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
    )
    
    return map_val.item() if torch.is_tensor(map_val) else map_val

def main():
    # Setup Model & Loss
    print("🏗️  Using Custom YOLO Architecture (Training from scratch)")
    print(f"📊 Hyperparameters: LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY}, Dropout=0.2")
    model = YOLO_Architecture(
        split_size=config.GRID_SIZE, 
        num_of_boxes=config.NUM_BOXES, 
        num_of_classes=config.NUM_CLASSES,
        dropout=0.2  # Proven dropout rate
    ).to(DEVICE)
    
    # Adam optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # ReduceLROnPlateau scheduler (proven configuration)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=15,
        min_lr=1e-6
    )
    
    loss_fn = YoloLoss()

    # --- LOAD CHECKPOINT (If True) ---
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # Setup Data Loaders - Use FULL training set
    train_dataset = YOLODataset(
        csv_file=config.TRAIN_CSV,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        transform=train_transform,
    )
    
    test_dataset = YOLODataset(
        csv_file=config.TEST_CSV,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        transform=test_transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
    
    # Start Training
    print(f"🚀 Training started on {DEVICE}...")
    print(f"📊 Dataset: {len(train_dataset)} images, Batch size: {BATCH_SIZE}")
    print(f"📈 Learning Rate: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        # Train for one epoch
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn)
        
        # Step the scheduler
        scheduler.step(epoch_loss)
        
        # Evaluate mAP every 10 epochs (DISABLED - will enable after loss < 100)
        # Early training has very low confidence scores, so mAP shows 0%
        # Focus on loss reduction first, then re-enable mAP evaluation
        # if (epoch + 1) % 10 == 0:
        #     print("\n📊 Evaluating mAP...")
        #     train_map = evaluate_map(model, train_loader, iou_threshold=0.5, conf_threshold=0.01)
        #     test_map = evaluate_map(model, test_loader, iou_threshold=0.5, conf_threshold=0.01)
        #     print(f"📈 Train mAP: {train_map * 100:.2f}% | Test mAP: {test_map * 100:.2f}%")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }
            save_checkpoint(checkpoint, filename="checkpoint_best.pth.tar")
            print(f"✅ Best model saved! Loss: {best_loss:.4f}")

        # --- SAVE CHECKPOINT ---
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": epoch_loss,
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
            print(f"✅ Checkpoint saved: checkpoint_epoch_{epoch+1}.pth.tar")
    
    # Final evaluation
    print("\n🎯 Final Evaluation...")
    train_map = evaluate_map(model, train_loader, iou_threshold=0.5, conf_threshold=0.01)
    test_map = evaluate_map(model, test_loader, iou_threshold=0.5, conf_threshold=0.01)
    print(f"\n📊 Final Results:")
    print(f"   Train mAP: {train_map * 100:.2f}%")
    print(f"   Test mAP: {test_map * 100:.2f}%")
    print(f"   Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
