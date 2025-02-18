import os
import yaml
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.cuda.amp as amp  # For mixed precision training

class ImageClassificationDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # Preload all image paths
        self.image_paths = [os.path.join(self.img_dir, file) for file in self.data['file']]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Use numpy to read image - faster than PIL
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = torch.tensor(self.data.iloc[idx]['label']).long()
        
        return image, label

def get_train_transforms(config):
    return A.Compose([
        A.Resize(config['data']['image_size'], config['data']['image_size']),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(
            mean=config['augmentation']['train']['normalize']['mean'],
            std=config['augmentation']['train']['normalize']['std']
        ),
        ToTensorV2(),
    ])

def get_model(config):
    if config['model']['name'] == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=config['model']['pretrained'])
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, config['model']['num_classes'])
    return model

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Mixed precision training
        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    
    return train_loss/len(loader), 100.*correct/total

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device and optimization flags
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    # Create datasets
    train_dataset = ImageClassificationDataset(
        config['data']['train_csv'],
        config['data']['train_dir'],
        transform=get_train_transforms(config)
    )
    
    # Use multiple workers and pin memory for faster data loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=4,  # Adjust based on CPU cores
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Create model and move to GPU
    model = get_model(config)
    model = model.to(device)
    
    # Enable mixed precision training
    scaler = amp.GradScaler()
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['train']['scheduler']['T_max']
    )
    
    # Training loop
    for epoch in range(config['train']['num_epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        print(f'Epoch: {epoch+1}/{config["train"]["num_epochs"]}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'GPU Memory: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB')
        
        scheduler.step()
        
        # Save model
        if not os.path.exists(config['output']['model_save_path']):
            os.makedirs(config['output']['model_save_path'])
            
        torch.save(model.state_dict(), 
                  os.path.join(config['output']['model_save_path'], f'model_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    main()