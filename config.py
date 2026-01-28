import torch
import os

# --- PATHS ---
# Base directory is the root of the project (where this file lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
# CSV files contain full relative paths, so we use empty strings here
IMG_DIR = ""
LABEL_DIR = ""

# Points to the CSV files in processed/
TRAIN_CSV = os.path.join(DATA_DIR, "processed", "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "processed", "test.csv")

# Checkpoint Path (Save model in root or a specific folder)
CHECKPOINT_FILE = os.path.join(BASE_DIR, "yolo_pothole.pth.tar")

# Hyperparameters (OVERFITTING MODE - for sanity check)
LEARNING_RATE = 5e-4  # Very high LR for aggressive overfitting
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0  # NO weight decay - allow full overfitting
EPOCHS = 200  # Full training to reach loss ~10
NUM_WORKERS = 2
PIN_MEMORY = False  # Set to False for MPS compatibility
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
