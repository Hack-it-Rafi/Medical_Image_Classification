import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from PIL import Image
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
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
    torch.backends.cudnn.deterministic = False  # Changed for speed
    torch.backends.cudnn.benchmark = True  # Optimized for speed

set_seed(42)

# OPTIMIZED Configuration
class Config:
    # Paths
    train_data_path = 'output/train/data.json'
    train_img_dir = 'output/train/imgs'
    
    # Model settings - FASTER MODEL
    model_name = 'efficientnet_b0'  # Single lightweight model instead of 3 heavy ones
    img_size = 224  # Reduced from 384 (3x faster)
    num_classes = 7
    
    # Training settings - OPTIMIZED FOR SPEED
    batch_size = 32  # Increased from 16 (2x faster)
    gradient_accumulation_steps = 2  # Simulates batch size of 64
    num_epochs = 30  # Reduced from 50
    learning_rate = 2e-4  # Slightly higher for faster convergence
    weight_decay = 1e-5
    num_folds = 3  # Reduced from 5 (40% faster)
    
    # Performance settings
    num_workers = 8  # Increased for faster data loading
    prefetch_factor = 3  # Prefetch batches
    persistent_workers = True  # Reuse workers
    
    # Early stopping
    patience = 7  # Reduced from 10
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Class names
    class_names = ['ear-left', 'ear-right', 'nose-left', 'nose-right', 'throat', 'vc-closed', 'vc-open']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

config = Config()

# SIMPLIFIED augmentation - much faster
def get_train_transforms():
    """Lightweight augmentation for faster training - NO FLIPS for left/right preservation"""
    return A.Compose([
        A.Resize(config.img_size, config.img_size),
        # REMOVED: A.HorizontalFlip(p=0.5),  # Would incorrectly swap ear-left/right, nose-left/right!
        A.Rotate(limit=10, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# Optimized Dataset with caching
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
        
        # Optimized loading
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = config.class_to_idx[item['anatomical_region']]
        return image, label

# Lightweight model
class MedicalModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.model(x)

# Fast training function with gradient accumulation
def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Mixed precision training
        with torch.amp.autocast(device="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels) / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        running_loss += loss.item() * config.gradient_accumulation_steps
        
        pbar.set_postfix({'loss': running_loss/(batch_idx+1), 'acc': 100.*correct/total})
    
    return running_loss / len(dataloader), 100. * correct / total

# Fast validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100. * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return running_loss / len(dataloader), acc, f1

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

def get_loss_weights(train_data, label_map, device):
    """Calculate class weights for weighted loss"""
    class_counts = torch.zeros(len(label_map))
    for item in train_data:
        label = label_map[item['anatomical_region']]
        class_counts[label] += 1
    
    class_weights = len(train_data) / (len(label_map) * class_counts)
    return class_weights.to(device)

# Optimized training function
def train_model(train_data, val_data, fold):
    print(f'\n{"="*50}')
    print(f'Training {config.model_name} - Fold {fold+1}/{config.num_folds}')
    print(f'{"="*50}')
    
    # Create datasets
    train_dataset = MedicalImageDataset(train_data, config.train_img_dir, get_train_transforms())
    val_dataset = MedicalImageDataset(val_data, config.train_img_dir, get_val_transforms())
    
    # Weighted sampling
    sample_weights = get_class_weights(train_data, config.class_to_idx)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor
    )
    
    # Create model
    model = MedicalModel(config.model_name, config.num_classes).to(config.device)
    
    # Compile model for faster execution (PyTorch 2.0+) - Skip on Windows
    if hasattr(torch, 'compile'):
        try:
            import platform
            if platform.system() != 'Windows':
                model = torch.compile(model)
                print("✓ Model compiled for faster execution")
            else:
                print("⚠ Model compilation skipped (Windows - Triton not supported)")
        except Exception as e:
            print(f"⚠ Model compilation not available: {e}")
    
    # Weighted loss
    class_weights = get_loss_weights(train_data, config.class_to_idx, config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Optimizer with higher learning rate
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Simpler scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler(device="cuda")
    
    best_acc = 0.0
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.device, scaler)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, config.device)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{config.model_name}_fold{fold}.pth')
            print(f'✓ Saved best model with Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f'⚡ Early stopping at epoch {epoch+1}')
            break
    
    return best_acc, best_f1

# Main execution
def main():
    print('='*60)
    print('FAST TRAINING MODE - OPTIMIZED FOR SPEED')
    print('='*60)
    print(f'Using device: {config.device}')
    
    if config.device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    print(f'\nOptimizations:')
    print(f'  • Model: {config.model_name} (lightweight)')
    print(f'  • Image size: {config.img_size}x{config.img_size} (reduced)')
    print(f'  • Batch size: {config.batch_size} (increased)')
    print(f'  • Effective batch: {config.batch_size * config.gradient_accumulation_steps} (gradient accumulation)')
    print(f'  • Folds: {config.num_folds} (reduced)')
    print(f'  • Max epochs: {config.num_epochs} (reduced)')
    print(f'  • Workers: {config.num_workers} (optimized)')
    
    # Load training data
    with open(config.train_data_path, 'r') as f:
        train_data = json.load(f)
    
    print(f'\nTotal training samples: {len(train_data)}')
    
    # Class distribution
    class_counts = {}
    for item in train_data:
        label = item['anatomical_region']
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print('\nClass distribution:')
    for class_name, count in sorted(class_counts.items()):
        print(f'  {class_name}: {count}')
    
    # Stratified K-Fold
    labels = [item['anatomical_region'] for item in train_data]
    label_indices = [config.class_to_idx[label] for label in labels]
    
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    import time
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, label_indices)):
        train_fold = [train_data[i] for i in train_idx]
        val_fold = [train_data[i] for i in val_idx]
        
        best_acc, best_f1 = train_model(train_fold, val_fold, fold)
        fold_results.append({'acc': best_acc, 'f1': best_f1})
    
    elapsed_time = time.time() - start_time
    
    # Results summary
    avg_acc = np.mean([r['acc'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    
    print('\n' + '='*60)
    print('TRAINING COMPLETED!')
    print('='*60)
    print(f'\n{config.model_name} Results:')
    print(f'  Average Validation Accuracy: {avg_acc:.2f}%')
    print(f'  Average Validation F1 Score: {avg_f1:.4f}')
    print(f'\nTotal training time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)')
    print(f'Average time per fold: {elapsed_time/config.num_folds/60:.1f} minutes')
    
    # Save results
    results = {
        'model': config.model_name,
        'img_size': config.img_size,
        'batch_size': config.batch_size,
        'num_folds': config.num_folds,
        'avg_accuracy': float(avg_acc),
        'avg_f1': float(avg_f1),
        'fold_results': fold_results,
        'training_time_minutes': elapsed_time/60
    }
    
    with open('fast_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n✓ Results saved to fast_training_results.json')
    print('='*60)

if __name__ == '__main__':
    main()
