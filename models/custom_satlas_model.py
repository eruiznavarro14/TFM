import torch
import torch.nn as nn
from satlaspretrain_models import Weights

def get_custom_satlas_model(num_classes=10, freeze_backbone=True):
    wm = Weights()

    # Cargar backbone Swin de Satlas sin cabeza
    full_model = wm.get_pretrained_model(
        "Sentinel2_SwinB_SI_RGB",
        fpn=False,
        head=None
    )
    backbone = full_model.backbone

    # Definir un wrapper que extrae la última escala y la pasa por la head
    class Wrapper(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.backbone = backbone
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.classifier = nn.Linear(1024, num_classes)

        def forward(self, x):
            features = self.backbone(x)  # List of 4 scales
            x = features[-1]             # Usamos la última: [B, 1024, H/32, W/32]
            x = self.pool(x)             # [B, 1024, 1, 1]
            x = self.flatten(x)          # [B, 1024]
            x = self.classifier(x)       # [B, num_classes]
            return x

    model = Wrapper(backbone, num_classes)

    # Congelamos el backbone
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model