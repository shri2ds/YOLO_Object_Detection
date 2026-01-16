"""
YOLO v1 Architecture Implementation
"""

import torch
import torch.nn as nn

# (kernel_size, filters, stride, padding)
# "M" = MaxPool2d
architecture_config = [
    (7, 64, 2, 3),   # Conv 1: 448 -> 224
    "M",             # Pool 1: 224 -> 112
    (3, 192, 1, 1),
    "M",             # Pool 2: 112 -> 56
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",             # Pool 3: 56 -> 28
    # List: [(layer1), (layer2), repeat_times]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",             # Pool 4: 28 -> 14
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1), # Conv: 14 -> 7 (Stride 2 reduces size)
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        # bias=False because BatchNorm essentially cancels out the bias of Conv
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyRelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels

        # The Darknet Backbone
        self.darknet = self._create_conv_layers(self.architecture)

        # The Fully Connected Head
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        # Flatten: (Batch, 1024, 7, 7) -> (Batch, 1024*7*7)
        x = torch.flatten(x, start_dim=1)
        return self.fcs(x)

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) ==  tuple:
                layers += [ CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]) ]
                in_channels = x[1]  #   Update for next layer
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [ CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]) ]
                    layers += [CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3] ) ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            # Original paper used 4096, but we reduce to 496 to save GPU RAM and because Potholes are simpler than ImageNet
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )


if __name__ == "__main__":
    split_size = 7
    num_boxes = 2
    num_classes = 1  # Potholes

    # Create the model
    model = Yolov1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)

    # Dummy Input: Batch of 2 images, 3 channels, 448x448
    x = torch.randn((2, 3, 448, 448))

    # Forward Pass
    out = model(x)

    # Reshape output to grid for checking
    out = out.reshape(out.shape[0], split_size, split_size, num_classes + num_boxes * 5)

    print(f"✅ Model Loaded Successfully.")
    print(f"   Input Shape: {x.shape}")
    print(f"   Output Shape: {out.shape}")
    print(f"   Expected:    torch.Size([2, 7, 7, 11])")
