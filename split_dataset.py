import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

def split_dataset(
    original_dir='EuroSAT_RGB', 
    output_dir='Eurosat_split_dataset', 
    train_ratio=0.7, 
    val_ratio=0.15,
    seed=42
):
    random.seed(seed)

    if not os.path.exists(original_dir):
        raise FileNotFoundError(f"No se encontró la carpeta: {original_dir}")

    class_names = sorted([cls for cls in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, cls))])

    splits = ['train', 'val', 'test']
    split_counts = {split: defaultdict(int) for split in splits}

    for split in splits:
        for cls in class_names:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    print("Dividiendo imágenes por clase...")

    for cls in tqdm(class_names):
        cls_dir = os.path.join(original_dir, cls)
        images = [img for img in os.listdir(cls_dir) if img.endswith('.jpg') or img.endswith('.png')]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val

        partitions = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, split_imgs in partitions.items():
            for img in split_imgs:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(output_dir, split, cls, img)
                shutil.copy(src, dst)
                split_counts[split][cls] += 1

    print("\n✅ Dataset dividido correctamente.")
    print("📊 Distribución final por clase y partición:\n")

    for split in splits:
        print(f"📁 {split.upper()}:")
        total = 0
        for cls in class_names:
            count = split_counts[split][cls]
            total += count
            print(f"   - {cls:<20}: {count} imágenes")
        print(f"   Total {split}: {total} imágenes\n")

if __name__ == "__main__":
    split_dataset()
