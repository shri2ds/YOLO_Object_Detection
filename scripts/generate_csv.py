import os
import pandas as pd
import random
import glob


# Actual data locations
IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"
OUTPUT_DIR = "data/processed"


def generate_csv():
    print(f"🔍 Scanning '{IMAGE_DIR}' for images...")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Find all images inside the folder
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) + \
                  glob.glob(os.path.join(IMAGE_DIR, "*.png"))

    data = []
    missing_labels = 0

    print(f"   Found {len(image_paths)} images. Linking labels...")

    for img_path in image_paths:
        # Get filename without extension
        basename = os.path.basename(img_path)
        file_root = os.path.splitext(basename)[0]
        label_filename = file_root + ".txt"
        
        # Check if corresponding label exists in LABEL_DIR
        label_path = os.path.join(LABEL_DIR, label_filename)
        
        if os.path.exists(label_path):
            # Store paths relative to project root
            data.append([img_path, label_path])
        else:
            missing_labels += 1
            if missing_labels < 3:  # Print first few errors only
                print(f"⚠️ Label missing for: {img_path}")

    # Validation
    if len(data) == 0:
        print("❌ Error: No labels found. Check folder names.")
        return

    # Shuffle and Split
    random.seed(42)
    random.shuffle(data)

    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # Save CSVs to processed directory
    train_csv_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_csv_path = os.path.join(OUTPUT_DIR, "test.csv")
    
    pd.DataFrame(train_data, columns=['img', 'label']).to_csv(train_csv_path, index=False)
    pd.DataFrame(test_data, columns=['img', 'label']).to_csv(test_csv_path, index=False)

    print(f"\n✅ SUCCESS!")
    print(f"   - Total matched: {len(data)}")
    print(f"   - Train samples: {len(train_data)}")
    print(f"   - Test samples: {len(test_data)}")
    print(f"   - Saved to '{OUTPUT_DIR}/' directory")


if __name__ == "__main__":
    generate_csv()