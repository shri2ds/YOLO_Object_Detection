import torch
import os

# --- PATHS ---
# Base directory is the root of the project (where this file lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(DATA_DIR, "images")
LABEL_DIR = os.path.join(DATA_DIR, "labels")

# Points to the CSV files in processed/
TRAIN_CSV = os.path.join(BASE_DIR, "data/processed/train.csv")
TEST_CSV = os.path.join(BASE_DIR, "data/processed/test.csv")

# Checkpoint Path (Save model in root or a specific folder)
CHECKPOINT_FILE = os.path.join(BASE_DIR, "yolo_pothole.pth.tar")

# Hyperparameters
LEARNING_RATE = 2e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "yolo_pothole.pth.tar"

# Model Config
IMG_SIZE = 448
GRID_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 1
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
