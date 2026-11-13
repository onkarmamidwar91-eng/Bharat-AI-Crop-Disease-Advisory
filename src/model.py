import torch
import torch.nn as nn
from torchvision import models

def build_edge_model(num_classes=38, variant="large", pretrained=True):
    """
    Builds an edge-optimized MobileNetV3 model.
    variant: 'small' (2.5M params, ultra-compact) or 'large' (4.2M params, field-adapted).
    Target runtime: Arm Cortex-A76 (Raspberry Pi 5) / Android via ExecuTorch or PyTorch.
    """
    if variant == "large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    else:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[3].in_features
    dropout = 0.3 if variant == "large" else 0.2
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model
