import os
import random
import torch
import torch.nn as nn
import argparse
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import make_grid
from datasets.eurosat_loader import get_eurosat_dataloaders
from models.vit_swin import create_model
from models.custom_satlas_model import get_custom_satlas_model
from models.swinV2 import get_swinV2_model
from utils import compute_metrics, plot_confusion_matrix, save_metrics_to_file


# 🔒 Establecer semillas
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

def visualize_predictions(model, dataloader, device, class_names, num_images=8, save_path=None):
    model.eval()

    all_images = []
    all_preds = []
    all_labels = []

    # 🔁 Recolectar todas las imágenes y etiquetas
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for img, pred, label in zip(images, preds, labels):
                all_images.append(img.cpu())
                all_preds.append(pred.cpu())
                all_labels.append(label.cpu())

    # 🔀 Mezclar aleatoriamente
    combined = list(zip(all_images, all_preds, all_labels))
    random.shuffle(combined)
    all_images, all_preds, all_labels = zip(*combined)

    # Seleccionar solo las primeras `num_images`
    selected_images = list(all_images[:num_images]) # Convertimos en lista
    selected_preds = all_preds[:num_images]
    selected_labels = all_labels[:num_images]

    # Crear grid
    grid_img = make_grid(selected_images, nrow=4, normalize=True)

    # Mostrar imágenes
    plt.figure(figsize=(12, 6))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")

    # Título con etiquetas
    title_lines = [
        f"Pred: {class_names[pred.item()]} | True: {class_names[true.item()]}"
        for pred, true in zip(selected_preds, selected_labels)
    ]
    full_title = "\n".join(title_lines)
    plt.title(full_title, fontsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"🖼 Imagen guardada en {save_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["vit", "swin", "swin2", "custom_satlas"])
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--data_dir", type=str, default="Eurosat_split_dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 📦 Cargar modelo
    if args.model_type == "custom_satlas":
        model = get_custom_satlas_model(num_classes=10).to(device)
    elif args.model_type == "swin2":
        model = get_swinV2_model(num_classes=10).to(device)
    else:
        model = create_model(args.model_type, num_classes=10).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"✅ Modelo {args.model_type.upper()} cargado desde {args.model_path}")

    # 📂 Cargar datos
    _, _, test_loader = get_eurosat_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        model_type=args.model_type
    )

    # 🎯 Evaluación
    accuracy = evaluate(model, test_loader, device)
    print(f"\n🎯 Accuracy en el conjunto TEST: {accuracy:.4f}")

    # 🖼 Visualización
    print("\n🖼 Mostrando predicciones sobre imágenes variadas del test set...")
    visualize_predictions(model, test_loader, device, class_names=CLASSES, num_images=8, save_path="pred_grid.png")

    # 📊 Métricas y matriz de confusión
    metrics = compute_metrics(model, test_loader, device)
    plot_confusion_matrix(metrics['all_labels'], metrics['all_preds'], normalize=True)
    save_metrics_to_file(metrics, filename=f"{args.model_type}_metrics.txt")

if __name__ == "__main__":
    main()
