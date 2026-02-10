"""Standalone script to compute mAP using the best saved checkpoint."""
import torch
from torch.utils.data import DataLoader

import config
from train import evaluate_map, train_transform, test_transform
from src.model import YOLO_Architecture
from src.dataset import YOLODataset

BEST_CHECKPOINT_PATH = "checkpoint_best.pth.tar"


def load_model_from_checkpoint():
    """Load YOLO_Architecture model weights from the best checkpoint."""
    model = YOLO_Architecture(
        split_size=config.GRID_SIZE,
        num_of_boxes=config.NUM_BOXES,
        num_of_classes=config.NUM_CLASSES,
        dropout=0.2,
    ).to(config.DEVICE)

    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build_loader(csv_path, transform):
    """Create a DataLoader for the given CSV and transform."""
    dataset = YOLODataset(
        csv_file=csv_path,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        transform=transform,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )


def main():
    print("📦 Loading best checkpoint for evaluation...")
    model = load_model_from_checkpoint()

    print("📂 Building data loaders...")
    train_loader = build_loader(config.TRAIN_CSV, train_transform)
    test_loader = build_loader(config.TEST_CSV, test_transform)

    print("\n📊 Evaluating Train mAP...")
    train_map = evaluate_map(model, train_loader, iou_threshold=0.5, conf_threshold=0.01)
    print(f"✅ Train mAP: {train_map * 100:.2f}%")

    print("\n📊 Evaluating Test mAP...")
    test_map = evaluate_map(model, test_loader, iou_threshold=0.5, conf_threshold=0.01)
    print(f"✅ Test mAP: {test_map * 100:.2f}%")


if __name__ == "__main__":
    main()
