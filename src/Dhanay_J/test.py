import os
import yaml
import torch
import pandas as pd
import numpy as np
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        df = pd.read_csv('./data/test.csv')
        self.image_files = [img for img in df['file']]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, img_name

def get_test_transforms(config):
    return A.Compose([
        A.Resize(config['data']['image_size'], config['data']['image_size']),
        A.Normalize(
            mean=config['augmentation']['test']['normalize']['mean'],
            std=config['augmentation']['test']['normalize']['std']
        ),
        ToTensorV2(),
    ])

def get_model(config):
    if config['model']['name'] == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=config['model']['pretrained'])
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, config['model']['num_classes'])
    return model

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset
    test_dataset = TestDataset(
        config['data']['test_dir'],
        transform=get_test_transforms(config)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Load model
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(
        os.path.join(config['output']['model_save_path'], 'model_epoch_120.pth')
    ))
    model.eval()
    
    # Predictions
    predictions = []
    image_files = []
    
    with torch.no_grad():
        for images, img_names in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            predictions.extend(predicted.cpu().numpy())
            image_files.extend(img_names)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        # 'file': image_files,
        'label': predictions
    })
    
    if not os.path.exists(config['output']['submission_path']):
        os.makedirs(config['output']['submission_path'])
        
    submission.to_csv(
        os.path.join(config['output']['submission_path'], 'submission.csv'),
        index=False
    )

if __name__ == '__main__':
    main()