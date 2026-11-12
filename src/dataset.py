import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PlantVillageDataset(Dataset):
    """
    Edge-ready Dataset loader for PlantVillage with custom sampling and augmentations.
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    """
    Heavy field-condition data augmentations to maximize model generalization and accuracy:
    - RandomResizedCrop: Invariant to leaf distance/zoom
    - RandomHorizontalFlip & RandomRotation: Invariant to handheld angle
    - ColorJitter: Invariant to harsh sunlight, shade, and golden hour lighting
    - GaussianBlur: Invariant to slight camera shake/blur
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def prepare_dataloaders(dataset_dir=r"D:\plantvillage dataset\color",
                        max_samples_per_class=100,
                        train_ratio=0.8,
                        batch_size=32,
                        seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    for cls_name in classes:
        cls_dir = os.path.join(dataset_dir, cls_name)
        img_names = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(img_names)

        # Cap samples per class to ensure fast, high-quality balanced training
        if max_samples_per_class and len(img_names) > max_samples_per_class:
            img_names = img_names[:max_samples_per_class]

        split_idx = int(len(img_names) * train_ratio)
        train_files = img_names[:split_idx]
        val_files = img_names[split_idx:]

        cls_idx = class_to_idx[cls_name]
        for f in train_files:
            train_paths.append(os.path.join(cls_dir, f))
            train_labels.append(cls_idx)

        for f in val_files:
            val_paths.append(os.path.join(cls_dir, f))
            val_labels.append(cls_idx)

    train_transform, val_transform = get_transforms()

    train_dataset = PlantVillageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PlantVillageDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, classes, class_to_idx
