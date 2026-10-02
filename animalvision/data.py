import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from animalvision.config import Config

def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_dataloaders():
    train_transform, val_transform = get_data_transforms()

    if Config.DATASET_NAME == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=Config.DATA_DIR, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=Config.DATA_DIR, train=False, download=True, transform=val_transform)
        
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_sub, val_sub = random_split(train_dataset, [train_size, val_size])
    else:
        full_dataset = datasets.ImageFolder(root=os.path.join(Config.DATA_DIR, "train"), transform=train_transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_sub, val_sub = random_split(full_dataset, [train_size, val_size])
        test_dataset = val_sub

    train_loader = DataLoader(train_sub, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_sub, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    return train_loader, val_loader, test_loader