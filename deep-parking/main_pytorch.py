import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from model import ParkingNet
from dataset import ParkingDataset, get_transform

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_dir):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

def main():
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_epochs = 6
    learning_rate = 0.0008
    
    # Create save directory
    save_dir = 'runs_pytorch'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model
    model = ParkingNet(num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Load PKLot dataset
    train_dataset = ParkingDataset(
        root_dir='splits/PKLot/config.root_folder',
        split_file='splits/PKLot/train.txt',
        transform=get_transform(is_training=True)
    )
    
    val_dataset = ParkingDataset(
        root_dir='splits/PKLot/config.root_folder',
        split_file='splits/PKLot/val.txt',
        transform=get_transform(is_training=False)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_dir)

if __name__ == '__main__':
    main()
