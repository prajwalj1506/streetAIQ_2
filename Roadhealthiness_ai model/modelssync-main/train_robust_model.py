import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import copy
import ssl

# Bypass macOS SSL certificate verification errors for PyTorch model downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Paths
DATA_DIR = "/Users/vijaykarthik/RoadHealthAImodel/Images/Images"
CSV_FILE = "/Users/vijaykarthik/RoadHealthAImodel/metadata.csv"
MODEL_SAVE_PATH = "/Users/vijaykarthik/RoadHealthAImodel/robust_street_model.pth"

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

class RoadDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['filename'])
        # Handle alpha channels or grayscale by converting to RGB
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            # Fallback for corrupted images
            print(f"Warning: Could not open {img_name}: {e}")
            image = Image.new("RGB", (224, 224), (128, 128, 128))
            
        label = int(self.dataframe.iloc[idx]['label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_model():
    print("--------------------------------------------------")
    print("Starting Transfer Learning with ResNet-18...")
    print("--------------------------------------------------")
    
    # 1. Load Data
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    print(f"Total images found in dataset: {len(df)}")
    
    # Split into train and validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"Training on {len(train_df)} images, Validating on {len(val_df)} images.")
    
    # 2. Aggressive Data Augmentation for Train, Standard for Val
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = RoadDataset(train_df, DATA_DIR, transform=train_transforms)
    val_dataset = RoadDataset(val_df, DATA_DIR, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Load Pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze earlier layers (optional, but good for small datasets to prevent overfitting)
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final Fully Connected layer
    num_ftrs = model.fc.in_features
    # Binary classification: Clean (0) vs Dirty (1)
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Optimize only the final layers
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:.4f}')
    
    # Load best weights and save
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
