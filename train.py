#!/usr/bin/env python3

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from src.model import YOLO_Architecture
from src.model_pretrained import YOLO_Pretrained
from src.dataset import YOLODataset
from src.loss import YoloLoss
from src.utils import (
    save_checkpoint,
    load_checkpoint,
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

# --- Transfer Learning Configuration ---
USE_PRETRAINED = True  # Set to True for ResNet-18 backbone, False for custom Darknet
DROPOUT_RATE = 0.0     # NO DROPOUT - overfitting mode to verify model can learn


# --- Image Transforms ---
# NO AUGMENTATION - overfitting mode (train = test, need to memorize exact images)
train_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),  # Only resize and convert to tensor, no randomness
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
        optimizer.step()  # No gradient clipping - allow full updates

        # 3. Update Progress Bar
        loop.set_postfix(loss=loss.item())

    epoch_mean_loss = sum(mean_loss) / len(mean_loss)
    print(f"Mean loss was {epoch_mean_loss}")
    return epoch_mean_loss

def main():
    # Setup Model & Loss
    if USE_PRETRAINED:
        print("📦 Using Pre-trained ResNet-18 backbone (Transfer Learning)")
        print(f"🛡️  Regularization: Dropout={DROPOUT_RATE}, Weight Decay={WEIGHT_DECAY}, Data Augmentation=Enabled")
        model = YOLO_Pretrained(
            split_size=config.GRID_SIZE, 
            num_of_boxes=config.NUM_BOXES, 
            num_of_classes=config.NUM_CLASSES,
            dropout=DROPOUT_RATE
        ).to(DEVICE)
    else:
        print("🏗️  Using Custom Darknet backbone (Training from scratch)")
        model = YOLO_Architecture(
            split_size=config.GRID_SIZE, 
            num_of_boxes=config.NUM_BOXES, 
            num_of_classes=config.NUM_CLASSES
        ).to(DEVICE)
    
    # Adam optimizer with NO weight decay for overfitting
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,  # 0.0 from config
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Very aggressive scheduler for overfitting mode
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,      # Reduce LR by 50%
        patience=5,      # Very aggressive: wait only 5 epochs
        min_lr=1e-7,     # Allow very low LR for deep overfitting
        threshold=0.005, # Require only 0.5% improvement
        cooldown=1       # Minimal cooldown
    )
    
    loss_fn = YoloLoss()

    # --- LOAD CHECKPOINT (If True) ---
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # Setup Data Loaders
    train_dataset = YOLODataset(
        csv_file=config.TRAIN_CSV,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        transform=train_transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,  # Don't drop the last incomplete batch
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
        # Save every 25 epochs
        if (epoch + 1) % 25 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch + 1}.pth.tar")
            print(f"✅ Checkpoint saved: checkpoint_epoch_{epoch + 1}.pth.tar")

if __name__ == "__main__":
    main()
