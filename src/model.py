import torch
import torch.nn as nn

class YOLO_Architecture(nn.Module):
    def __init__(self, split_size=7, num_of_classes=20, num_of_boxes=2, dropout=0.2):
        super(YOLO_Architecture, self).__init__()
        self.S = split_size
        self.B = num_of_boxes
        self.C = num_of_classes
        self.dropout_rate = dropout

        # 1. The Backbone (Feature Extractor)
        # In production, this would be ResNet50 or Darknet53.
        # We build a simple one to see the dimension reduction.
        self.features = nn.Sequential(
            # Block 1: Input 448x448 -> 224x224
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # 112X112

            # Block 2: 112 -> 56
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), #56X56

                # Block 3: 56 -> 28
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 28x28

                # Block 4: 28 -> 14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 14x14

                # Block 5: 14 -> 7 (The Final Grid Size)
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # Output: 7x7
        )

        # 2. The Head (Detection Layer)
        # We flatten the 7x7x1024 features and map them to the S*S*(B*5+C) output
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fcs(x)

        # Reshape to the Grid Format: [Batch, S, S, (C + B*5)]
        # This is the "Magic Shape" where spatial meaning is restored
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        return x


if __name__ == "__main__":
    # Simulate a batch of 2 images
    BATCH_SIZE = 2
    model = YOLO_Architecture(split_size=7, num_of_boxes=2, num_of_classes=20)

    # Input: 448x448 RGB images
    dummy_input = torch.randn(BATCH_SIZE, 3, 448, 448)
    output = model(dummy_input)

    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")

    expected_depth = 20 + 2 * 5  # 30
    print(f"\nVerification:")
    print(f"Grid Size: {output.shape[1]}x{output.shape[2]} (Expected 7x7)")
    print(f"Output Depth: {output.shape[3]} (Expected {expected_depth})")

    if output.shape == (BATCH_SIZE, 7, 7, 30):
        print("\n✅ SUCCESS: The Architecture produces the correct Tensor Grid.")
    else:
        print("\n❌ FAIL: Dimension mismatch.")


