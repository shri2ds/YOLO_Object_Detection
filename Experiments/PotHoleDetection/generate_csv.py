import os
import pandas as pd
import random
import glob


DATASET_DIR = "data/final_pot_holes"


def generate_csv():
    print(f"🔍 Scanning '{DATASET_DIR}' for images...")

    # 1. Find all images inside the folder
    # This matches 'final_pot_holes/*.jpg'
    image_paths = glob.glob(os.path.join(DATASET_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(DATASET_DIR, "*.jpeg")) + \
                  glob.glob(os.path.join(DATASET_DIR, "*.png"))

    data = []
    missing_labels = 0

    print(f"   Found {len(image_paths)} images. Linking labels...")

    for img_path in image_paths:
        # img_path example: "final_pot_holes/3.jpg"

        # Get filename without extension (e.g., "3")
        basename = os.path.basename(img_path)
        file_root = os.path.splitext(basename)[0]

        label_filename = file_root + ".txt"

        # WE CHECK TWO LOCATIONS:
        # 1. Same folder as image (Standard)
        path_option_1 = os.path.join(os.path.dirname(img_path), label_filename)
        # 2. Inside 'labels' subfolder (Your Structure)
        path_option_2 = os.path.join(os.path.dirname(img_path), "labels", label_filename)

        final_label_path = None

        if os.path.exists(path_option_1):
            final_label_path = path_option_1
        elif os.path.exists(path_option_2):
            final_label_path = path_option_2

        if final_label_path:
            # We save the path relative to the 'data' folder
            data.append([img_path, final_label_path])
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

    # Save CSVs
    pd.DataFrame(train_data, columns=['img', 'label']).to_csv("data/train.csv", index=False)
    pd.DataFrame(test_data, columns=['img', 'label']).to_csv("data/test.csv", index=False)

    print(f"\n✅ SUCCESS!")
    print(f"   - Total matched: {len(data)}")
    print(f"   - Train samples: {len(train_data)}")
    print(f"   - Test samples: {len(test_data)}")
    print(f"   - Saved 'train.csv' and 'test.csv' in 'data' folder.")


if __name__ == "__main__":
    generate_csv()