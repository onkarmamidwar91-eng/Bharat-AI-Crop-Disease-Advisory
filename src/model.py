import torch
import torch.nn as nn
from torchvision import models

def build_edge_model(num_classes=38, pretrained=True):
    """
    Builds an edge-optimized MobileNetV3-Small model.
    MobileNetV3-Small features hard-swish activations and SE (Squeeze-and-Excitation) blocks,
    delivering high accuracy with minimal memory footprint (~2.5M params, ~9.5MB file size).
    Ideal for Arm-based SoCs, Raspberry Pi 5, and Android devices.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace final classification head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )

    return model
