import os
# os.environ["OMP_NUM_THREADS"] = "1"
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.eurosat_loader import get_eurosat_dataloaders
from models.vit_swin import create_model
from models.swinV2 import get_swinV2_model
from models.custom_satlas_model import get_custom_satlas_model   


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience, save_path):
    best_val_acc = 0.0
    epochs_no_improve = 0

    lr_scheduler_patience = 3
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=lr_scheduler_patience)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f"\n🌀 Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc="Entrenando"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # Scheduler step 
        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"✅ Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"🔍 Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            print("💾 Mejor modelo encontrado, guardando...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"⏹ Early stopping: no mejora en {patience} epochs")
            break
        
        torch.cuda.empty_cache()
        gc.collect()

    return history

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss / total, correct / total

def plot_history(history, output_dir):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label="Train Acc")
    plt.plot(history['val_acc'], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "training_plot.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["vit", "swin", "swin2", "custom_satlas"])
    parser.add_argument("--data_dir", type=str, default="Eurosat_split_dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--plot_dir", type=str, default="plots")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model selection
    if args.model_type == "custom_satlas":
        model = get_custom_satlas_model(num_classes=10).to(device)
    elif args.model_type == "swin2":
        model = get_swinV2_model(num_classes=10).to(device)
    elif args.model_type in ["vit", "swin"]:
        model = create_model(args.model_type, num_classes=10).to(device)
    else:
        raise ValueError(f"Modelo '{args.model_type}' no reconocido.")

    # Dataloaders
    train_loader, val_loader, _ = get_eurosat_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        model_type=args.model_type
    )


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)


    print(f"\n🚀 Entrenando modelo {args.model_type.upper()} en {device}...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        save_path=args.save_path
    )

    print(f"\n📊 Guardando gráfica en {args.plot_dir}")
    plot_history(history, args.plot_dir)

if __name__ == "__main__":

    main()
