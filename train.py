import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from PIL import Image
import json
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configuration
class Config:
    # Paths
    train_data_path = 'output/train/data.json'
    train_img_dir = 'output/train/imgs'
    test_data_path = 'output/test/data.json'
    test_img_dir = 'output/test/imgs'
    
    # Model settings
    model_names = ['efficientnet_b3', 'convnext_base', 'swinv2_base_window12_192_22k']
    img_size = 384
    num_classes = 7
    
    # Training settings
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_folds = 5
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Class names
    class_names = ['ear-left', 'ear-right', 'nose-left', 'nose-right', 'throat', 'vc-closed', 'vc-open']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

config = Config()

# Advanced augmentation for training with medical-specific improvements
def get_train_transforms():
    """Anatomy-aware augmentation - removed VerticalFlip, added medical-specific augmentations"""
    return A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.HorizontalFlip(p=0.5),
        # REMOVED: A.VerticalFlip(p=0.3),  # Not anatomically correct for medical images
        A.Rotate(limit=15, p=0.4),  # Reduced rotation - anatomical orientation matters
        
        # Medical imaging specific augmentations
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),  # Different imaging devices
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),  # Enhance anatomical details
        A.Equalize(p=0.2),  # Enhance contrast for subtle features
        
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),  # Camera sensor noise
        
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        A.CLAHE(clip_limit=4.0, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=15, p=0.4),
        A.CoarseDropout(max_holes=4, max_height=24, max_width=24, p=0.2),  # Reduced
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

# Dataset class
class MedicalImageDataset(Dataset):
    def __init__(self, data_list, img_dir, transform=None):
        self.data_list = data_list
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = os.path.join(self.img_dir, item['path'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = config.class_to_idx[item['anatomical_region']]
        
        return image, label

# Model with multiple architectures
class EnsembleModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.model(x)

# Label smoothing loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Training function
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        running_loss += loss.item()
        
        pbar.set_postfix({'loss': running_loss/len(pbar), 'acc': 100.*correct/total})
    
    if scheduler is not None:
        scheduler.step()
    
    return running_loss / len(dataloader), 100. * correct / total

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': running_loss/len(pbar), 'acc': 100.*correct/total})
    
    acc = 100. * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return running_loss / len(dataloader), acc, f1

# Helper function to get class weights for weighted sampling
def get_class_weights(train_data, label_map):
    """Calculate class weights for imbalanced dataset"""
    class_counts = {}
    for item in train_data:
        label = label_map[item['anatomical_region']]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total = len(train_data)
    weights = {label: total / count for label, count in class_counts.items()}
    sample_weights = [weights[label_map[item['anatomical_region']]] for item in train_data]
    return torch.DoubleTensor(sample_weights)

# Helper function to get loss weights
def get_loss_weights(train_data, label_map, device):
    """Calculate class weights for weighted loss"""
    class_counts = torch.zeros(len(label_map))
    for item in train_data:
        label = label_map[item['anatomical_region']]
        class_counts[label] += 1
    
    # Inverse frequency weights
    class_weights = len(train_data) / (len(label_map) * class_counts)
    return class_weights.to(device)

# Main training function
def train_model(model_name, train_data, val_data, fold):
    print(f'\n{"="*50}')
    print(f'Training {model_name} - Fold {fold}')
    print(f'{"="*50}')
    
    # Create datasets
    train_dataset = MedicalImageDataset(train_data, config.train_img_dir, get_train_transforms())
    val_dataset = MedicalImageDataset(val_data, config.train_img_dir, get_val_transforms())
    
    # IMPROVEMENT: Weighted sampling for class imbalance
    sample_weights = get_class_weights(train_data, config.class_to_idx)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = EnsembleModel(model_name, config.num_classes).to(config.device)
    
    # IMPROVEMENT: Weighted loss function for class imbalance
    class_weights = get_loss_weights(train_data, config.class_to_idx, config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, 
                           weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda")
    
    best_acc = 0.0
    best_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, scheduler, config.device, scaler)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, config.device)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model_name}_fold{fold}.pth')
            print(f'✓ Saved best model with Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return best_acc, best_f1

# Main execution
def main():
    print(f'Using device: {config.device}')
    if config.device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # Load training data
    with open(config.train_data_path, 'r') as f:
        train_data = json.load(f)
    
    print(f'\nTotal training samples: {len(train_data)}')
    
    # Count samples per class
    class_counts = {}
    for item in train_data:
        label = item['anatomical_region']
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print('\nClass distribution:')
    for class_name, count in sorted(class_counts.items()):
        print(f'  {class_name}: {count}')
    
    # Stratified K-Fold cross-validation
    labels = [item['anatomical_region'] for item in train_data]
    label_indices = [config.class_to_idx[label] for label in labels]
    
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)
    
    results = {model_name: [] for model_name in config.model_names}
    
    for model_name in config.model_names:
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, label_indices)):
            train_fold = [train_data[i] for i in train_idx]
            val_fold = [train_data[i] for i in val_idx]
            
            best_acc, best_f1 = train_model(model_name, train_fold, val_fold, fold)
            fold_results.append({'acc': best_acc, 'f1': best_f1})
        
        results[model_name] = fold_results
        
        # Print fold results
        avg_acc = np.mean([r['acc'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        print(f'\n{model_name} - Average Validation Accuracy: {avg_acc:.2f}%')
        print(f'{model_name} - Average Validation F1: {avg_f1:.4f}')
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n' + '='*50)
    print('Training completed!')
    print('='*50)

if __name__ == '__main__':
    main()
