import timm
import torch.nn as nn

def create_model(model_name, num_classes, pretrained=True):
    if model_name == "vit":  # ViT base
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes
        )
    elif model_name == "swin":  # Swin V1 base
        model = timm.create_model(
            "swin_base_patch4_window7_224", pretrained=pretrained, num_classes=num_classes
        )
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado")
    
    # Congelar todos los parámetros
    for name, param in model.named_parameters():
        if "head" not in name:  # solo dejamos entrenable la cabeza
            param.requires_grad = False


    return model
