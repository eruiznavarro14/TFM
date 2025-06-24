import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import Counter
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

def count_images_per_class(path):
    dataset = ImageFolder(path)
    class_counts = Counter([label for _, label in dataset])
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    for label_idx, count in class_counts.items():
        print(f"{idx_to_class[label_idx]}: {count}")

def show_samples(path, num_per_class=5):
    dataset = ImageFolder(path, transform=transforms.ToTensor())
    class_to_imgs = {}
    for img, label in dataset:
        if label not in class_to_imgs:
            class_to_imgs[label] = []
        if len(class_to_imgs[label]) < num_per_class:
            class_to_imgs[label].append(img.permute(1, 2, 0))  # CHW → HWC

    for label, imgs in class_to_imgs.items():
        fig, axs = plt.subplots(1, num_per_class, figsize=(15, 3))
        for i, img in enumerate(imgs):
            axs[i].imshow(img)
            axs[i].axis('off')
        plt.suptitle(dataset.classes[label])
        plt.show()

if __name__ == "__main__":
    path = "Eurosat_split_dataset/train"
    count_images_per_class(path)
    show_samples(path)
