import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from PIL import Image

import torch

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

def get_augmented_train_loader(data_dir, model_type="vit", batch_size=8):
    image_size = 224

    if model_type in ["vit", "swin"]:
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
    elif model_type == "satlas":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor()
        ])
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    train_dir = os.path.join(data_dir, "train")
    dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, dataset.classes

def denormalize(img_tensor):
    # Solo para modelos que usan normalización ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return img_tensor * std + mean

def visualize_batch(loader, class_names, model_type="vit"):
    images, labels = next(iter(loader))

    if  model_type in ["vit", "swin"]:
        images = denormalize(images)  # solo para usar vit o swin

    grid = make_grid(images, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    titles = [class_names[label] for label in labels]
    print("Clases mostradas:", titles)
    plt.title("Ejemplos con aumentos del set de entrenamiento")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Eurosat_split_dataset")
    parser.add_argument("--model_type", type=str, default="vit", choices=["vit", "swin", "satlas"])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    loader, class_names = get_augmented_train_loader(
        data_dir=args.data_dir,
        model_type=args.model_type,
        batch_size=args.batch_size
    )
    visualize_batch(loader, class_names, model_type= args.model_type)

if __name__ == "__main__":
    main()
