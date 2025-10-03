import os
import torch
from torchvision import datasets, transforms

def get_tinyimagenet_loaders(data_dir="./tiny-imagenet-200", batch_size=128, num_workers=4):
    """
    Tiny-ImageNet loaders (train / val).
    Directory must contain 'train/' and 'val/' folders in ImageFolder format.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_set = datasets.ImageFolder(train_dir, transform=transform_train)
    val_set = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader

