from satlaspretrain_models import Weights, Head

def get_satlas_model(num_classes=10, freeze_backbone=True):
    wm = Weights()
    model = wm.get_pretrained_model(
        "Sentinel2_SwinB_SI_RGB",
        fpn=False,
        head=Head.CLASSIFY,
        num_categories=num_classes
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False

    return model