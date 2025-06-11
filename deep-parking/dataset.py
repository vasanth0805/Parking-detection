import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class ParkingDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Read the split file
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        self.samples = []
        for line in lines:
            img_path, label = line.strip().split()
            self.samples.append((img_path, int(label)))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_path)
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transform(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
