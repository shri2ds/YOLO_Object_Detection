"""
YOLO v1 with Pre-trained ResNet-18 Backbone (Transfer Learning)
Replaces custom Darknet with ImageNet pre-trained ResNet for faster convergence.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class YOLO_Pretrained(nn.Module):
    def __init__(self, split_size=7, num_of_boxes=2, num_of_classes=1, dropout=0.3):
        super(YOLO_Pretrained, self).__init__()
        
        self.S = split_size
        self.B = num_of_boxes
        self.C = num_of_classes
        
        # Load pre-trained ResNet-18 (lighter than ResNet-50, faster on Mac)
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final FC layer and avgpool
        # ResNet-18 output: (Batch, 512, 14, 14) after layer4
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Additional conv layers to reduce spatial size to 7x7
        # (Batch, 512, 14, 14) -> (Batch, 1024, 7, 7)
        # Added dropout for regularization
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(dropout * 0.5),
        )
        
        # Enhanced detection head with gradual compression
        # 50,176 → 4096 → 1024 → 512 → 539 (smoother transformation)
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, self.S * self.S * (self.C + self.B * 5)),
        )
        
        # Freeze early layers of ResNet (optional, for faster training)
        # Uncomment to freeze first 2 layers
        # for param in list(self.backbone.parameters())[:20]:
        #     param.requires_grad = False
    
    def forward(self, x):
        # Extract features with pre-trained backbone
        x = self.backbone(x)
        
        # Additional convolutions
        x = self.conv_layers(x)
        
        # Detection head
        x = self.fcs(x)
        
        return x


if __name__ == "__main__":
    split_size = 7
    num_boxes = 2
    num_classes = 1
    
    # Create model with pre-trained weights
    model = YOLO_Pretrained(split_size=split_size, num_of_boxes=num_boxes, num_of_classes=num_classes)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Pre-trained Model Loaded Successfully.")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn((2, 3, 448, 448))
    out = model(x)
    out = out.reshape(out.shape[0], split_size, split_size, num_classes + num_boxes * 5)
    
    print(f"   Input Shape: {x.shape}")
    print(f"   Output Shape: {out.shape}")
    print(f"   Expected:    torch.Size([2, 7, 7, 11])")
