import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_eurosat_dataloaders(data_dir, batch_size=32, model_type="vit"):
    image_size = 224

    if model_type in ["vit", "swin", "swin2"]:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    elif model_type in ["satlas", "custom_satlas"]:
        image_size_satlas = 224
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size_satlas, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor()  # Transforma [H x W x C] a [C x H x W] y normaliza a [0, 1]
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((image_size_satlas, image_size_satlas)),
            transforms.ToTensor()
        ])

    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    # Datasets y DataLoaders
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader

