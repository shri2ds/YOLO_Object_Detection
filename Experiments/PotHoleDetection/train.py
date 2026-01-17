import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from Model import Yolov1
from dataset import YOLODataset
from loss import YoloLoss
from Calculate_IOU import inter_over_union

# --- Hyperparameters ---
LEARNING_RATE = 2e-5
BATCH_SIZE = 8 
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = False
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"

# --- Image Transforms ---
# We resize to 448x448 as required by YOLO
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x,y) in enumerate(loop):
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
    # A. Setup Model & Loss
    model = Yolov1(split_size=7, num_boxes=2, num_classes=1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    # B. Setup Data Loaders
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
    
    # C. Start Training
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
