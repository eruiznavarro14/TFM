import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

def compute_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)

    print("📊 Clasification Report:")
    print(report)

    return {
        "accuracy": acc,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "all_preds": all_preds,
        "all_labels": all_labels
    }

def plot_confusion_matrix(y_true, y_pred, class_names=CLASSES, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", square=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Matriz de Confusión" + (" (Normalizada)" if normalize else ""))
    plt.tight_layout()
    plt.show()

def save_metrics_to_file(metrics_dict, filename="metrics.txt"):
    with open(filename, "w") as f:
        f.write(f"Accuracy: {metrics_dict['accuracy']:.4f}\n\n")
        f.write("Precision por clase:\n")
        for cls, prec in zip(CLASSES, metrics_dict['precision_per_class']):
            f.write(f"{cls}: {prec:.4f}\n")
        f.write("\nRecall por clase:\n")
        for cls, rec in zip(CLASSES, metrics_dict['recall_per_class']):
            f.write(f"{cls}: {rec:.4f}\n")
        f.write("\nF1-score por clase:\n")
        for cls, f1 in zip(CLASSES, metrics_dict['f1_per_class']):
            f.write(f"{cls}: {f1:.4f}\n")
