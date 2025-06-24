import torch.nn as nn
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

def get_swinV2_model(num_classes: int, pretrained: bool = True):

    if pretrained:
        weights = Swin_V2_B_Weights.DEFAULT
        model = swin_v2_b(weights=weights)
    else:
        model = swin_v2_b(weights=None)

    # Obtener el número de características de entrada de la cabeza original
    in_features = model.head.in_features

    # Sustituir la cabeza de clasificación
    model.head = nn.Linear(in_features, num_classes)

    # Congelar todos los parámetros excepto la nueva cabeza
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    return model
