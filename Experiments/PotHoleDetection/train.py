import torch
import os
import sys
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import YOLODataset
from loss import YoloLoss
from utils import (
    save_checkpoint,
    load_checkpoint,
)

# --- Hyperparameters ---
LEARNING_RATE = 2e-5
DEVICE = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
BATCH_SIZE = 8 
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = False
LOAD_MODEL = False                 # Change it to true if we've a model file 
LOAD_MODEL_FILE = "exmaple.tar"    # Provide the name of the model   


# --- Image Transforms ---
# We resize to 448x448 as required by YOLO
transform = transforms.Compe([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

def train_fn(train_loader, model, optimizer, ls_fn):
    loop = tqdm(train_loader, leave=True)
    mean_ls = []

    for batch_idx, (x,y) in enumerate(loop):
        x, y  = x.to(DEVICE), y.to(DEVICE)
        
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

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

def main():
    # Setup Model & Loss
    model = Yolov1(split_size=7, num_boxes=2, num_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    # --- LOAD CHECKPOINT (If True) ---
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # Setup Data Loaders
    # Note: We use the same file for train/test for now to overfit first(To verify if our model is working as expected)
    train_dataset = YOLODataset(
        csv_file="data/train.csv",
        img_dir="",
        label_dir="",
        transform=transform,
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
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        # Train for one epoch
        train_fn(train_loader, model, optimizer, loss_fn)

        # --- SAVE CHECKPOINT ---
        # We save every 3 epochs to save disk space
        if epoch % 3 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch}.pth.tar")


if __name__ == "__main__":
    main()
