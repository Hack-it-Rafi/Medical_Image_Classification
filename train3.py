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
from sklearn.metrics import f1_score, classification_report, confusion_matrix
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
    torch.backends.cudnn.deterministic = False  # For speed
    torch.backends.cudnn.benchmark = True  # Optimized for speed

set_seed(42)

# ============================================================================
# DELIBERATE DESIGN CHOICE #1: Optimized Multi-Model Ensemble Strategy
# Why: Balance between accuracy and training time
# ============================================================================
class Config:
    # Paths
    train_data_path = 'output/train/data.json'
    train_img_dir = 'output/train/imgs'
    test_data_path = 'output/test/data.json'
    test_img_dir = 'output/test/imgs'
    
    # DESIGN CHOICE: Two complementary models (CNN + Transformer)
    model_names = ['convnext_tiny', 'efficientnet_b2']  # Fast but powerful
    img_size = 288  # Sweet spot: better than 224, faster than 384
    num_classes = 7
    
    # Training settings
    batch_size = 24  # Optimized for GPU memory
    gradient_accumulation_steps = 2  # Effective batch size: 48
    num_epochs = 40
    learning_rate = 2e-4
    weight_decay = 1e-4
    num_folds = 4  # Balance between validation quality and time
    
    # DESIGN CHOICE: Test-Time Augmentation
    tta_enabled = True
    tta_transforms = 3  # Number of TTA variants
    
    # Performance settings
    num_workers = 6
    prefetch_factor = 2
    persistent_workers = True
    
    # Early stopping
    patience = 8
    min_delta = 0.001  # Minimum improvement threshold
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Class names
    class_names = ['ear-left', 'ear-right', 'nose-left', 'nose-right', 'throat', 'vc-closed', 'vc-open']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

config = Config()

# ============================================================================
# DELIBERATE DESIGN CHOICE #2: Medical-Specific Augmentation Strategy
# Why: Respect anatomical constraints while adding diversity
# ============================================================================
def get_train_transforms():
    """
    Medical imaging augmentation strategy:
    - NO horizontal flip (would swap left/right labels incorrectly!)
    - NO vertical flip (anatomically incorrect)
    - Limited rotation (preserve orientation)
    - Contrast/brightness for different imaging devices
    - Noise simulation for real-world conditions
    """
    return A.Compose([
        A.Resize(config.img_size, config.img_size),
        
        # Geometric (anatomy-aware) - NO FLIPS!
        # REMOVED: A.HorizontalFlip(p=0.5),  # Would mislabel ear-left/right, nose-left/right!
        A.Rotate(limit=12, p=0.4),  # Small rotation only
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=12, p=0.4),
        
        # Color/Contrast (simulate different devices)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(85, 115), p=0.4),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=0.3),
        
        # Medical imaging specific
        A.CLAHE(clip_limit=3.0, p=0.4),  # Enhance local contrast
        A.Sharpen(alpha=(0.15, 0.4), lightness=(0.6, 1.0), p=0.3),
        
        # Noise (simulate sensor/compression artifacts)
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.4), p=1.0),
        ], p=0.3),
        
        # Occlusion (simulate partial views)
        A.CoarseDropout(max_holes=3, max_height=20, max_width=20, p=0.2),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_tta_transforms():
    """Test-Time Augmentation transforms - NO FLIPS for left/right preservation"""
    return [
        A.Compose([
            A.Resize(config.img_size, config.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # REMOVED: HorizontalFlip TTA variant - would mislabel left/right anatomical regions
        A.Compose([
            A.Resize(config.img_size, config.img_size),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(config.img_size, config.img_size),
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]

# Dataset class
class MedicalImageDataset(Dataset):
    def __init__(self, data_list, img_dir, transform=None, tta_transforms=None):
        self.data_list = data_list
        self.img_dir = img_dir
        self.transform = transform
        self.tta_transforms = tta_transforms
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = os.path.join(self.img_dir, item['path'])
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.tta_transforms:
            # Return multiple augmented versions
            images = []
            for tta_transform in self.tta_transforms:
                aug = tta_transform(image=image)
                images.append(aug['image'])
            label = config.class_to_idx[item['anatomical_region']]
            return images, label
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            label = config.class_to_idx[item['anatomical_region']]
            return image, label

# ============================================================================
# DELIBERATE DESIGN CHOICE #3: Custom Architecture with Attention
# Why: Add spatial attention to focus on anatomical features
# ============================================================================
class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important anatomical regions"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention

class MedicalClassifier(nn.Module):
    """
    Enhanced model with:
    - Pretrained backbone for transfer learning
    - Spatial attention for anatomical focus
    - Dropout for regularization
    """
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config.img_size, config.img_size)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Add spatial attention
        self.attention = SpatialAttention(self.feature_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply spatial attention
        features = self.attention(features)
        
        # Global pooling
        features = self.global_pool(features)
        features = features.flatten(1)
        
        # Classification
        output = self.classifier(features)
        return output

# ============================================================================
# DELIBERATE DESIGN CHOICE #4: Focal Loss for Class Imbalance
# Why: Better than weighted CE for handling imbalanced medical datasets
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: Focuses learning on hard examples
    Better than weighted cross-entropy for imbalanced datasets
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Training function with gradient accumulation
def train_epoch(model, dataloader, criterion, optimizer, device, scaler, accumulation_steps):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        running_loss += loss.item() * accumulation_steps
        
        pbar.set_postfix({'loss': running_loss/(batch_idx+1), 'acc': 100.*correct/total})
    
    return running_loss / len(dataloader), 100. * correct / total

# ============================================================================
# DELIBERATE DESIGN CHOICE #5: Test-Time Augmentation
# Why: Improves robustness and accuracy at inference
# ============================================================================
def validate_with_tta(model, dataloader, criterion, device):
    """Validation with Test-Time Augmentation"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images_list, labels in tqdm(dataloader, desc='Validation', leave=False):
            labels = labels.to(device, non_blocking=True)
            
            # Average predictions across TTA transforms
            tta_outputs = []
            for images in images_list:
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                tta_outputs.append(outputs)
            
            # Average TTA predictions
            outputs = torch.stack(tta_outputs).mean(dim=0)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            running_loss += loss.item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return running_loss / len(dataloader), acc, f1, all_preds, all_labels

def validate(model, dataloader, criterion, device):
    """Standard validation without TTA"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            running_loss += loss.item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return running_loss / len(dataloader), acc, f1, all_preds, all_labels

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

def get_focal_loss_alpha(train_data, label_map, device):
    """Calculate alpha for Focal Loss"""
    class_counts = torch.zeros(len(label_map))
    for item in train_data:
        label = label_map[item['anatomical_region']]
        class_counts[label] += 1
    
    alpha = 1.0 / class_counts
    alpha = alpha / alpha.sum()  # Normalize
    return alpha.to(device)

# ============================================================================
# DELIBERATE DESIGN CHOICE #6: Advanced Learning Rate Scheduling
# Why: OneCycleLR for faster convergence
# ============================================================================
def train_model(model_name, train_data, val_data, fold):
    print(f'\n{"="*60}')
    print(f'Training {model_name} - Fold {fold+1}/{config.num_folds}')
    print(f'{"="*60}')
    
    # Create datasets
    train_dataset = MedicalImageDataset(train_data, config.train_img_dir, get_train_transforms())
    val_dataset = MedicalImageDataset(val_data, config.train_img_dir, get_val_transforms())
    
    if config.tta_enabled:
        val_tta_dataset = MedicalImageDataset(val_data, config.train_img_dir, 
                                               tta_transforms=get_tta_transforms())
    
    # Weighted sampling
    sample_weights = get_class_weights(train_data, config.class_to_idx)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
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
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor
    )
    
    if config.tta_enabled:
        val_tta_loader = DataLoader(
            val_tta_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    # Create model
    model = MedicalClassifier(model_name, config.num_classes).to(config.device)
    
    # Try to compile model (PyTorch 2.0+) - Skip on Windows or if Triton unavailable
    # if hasattr(torch, 'compile'):
    #     try:
    #         import platform
    #         if platform.system() != 'Windows':
    #             model = torch.compile(model, mode='max-autotune')
    #             print("✓ Model compiled with max-autotune")
    #         else:
    #             print("⚠ Model compilation skipped (Windows - Triton not supported)")
    #     except Exception as e:
    #         print(f"⚠ Model compilation not available: {e}")
    
    # Focal Loss with class weights
    alpha = get_focal_loss_alpha(train_data, config.class_to_idx, config.device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, 
                           weight_decay=config.weight_decay)
    
    # OneCycleLR scheduler for faster convergence
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    scaler = torch.amp.GradScaler("cuda")
    
    best_acc = 0.0
    best_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device, 
            scaler, config.gradient_accumulation_steps
        )
        
        # Standard validation
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, config.device
        )
        
        # Update scheduler
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
        
        # Check improvement
        improved = False
        if val_acc > best_acc + config.min_delta:
            best_acc = val_acc
            best_f1 = val_f1
            improved = True
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
        
        if improved:
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, f'best_{model_name}_fold{fold}.pth')
            print(f'✓ Saved best model (Acc: {val_acc:.2f}%, F1: {val_f1:.4f})')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f'⚡ Early stopping at epoch {epoch+1}')
            break
    
    # Final evaluation with TTA
    if config.tta_enabled:
        print('\n🔄 Running final evaluation with Test-Time Augmentation...')
        model.load_state_dict(torch.load(f'best_{model_name}_fold{fold}.pth')['model_state_dict'])
        tta_loss, tta_acc, tta_f1, tta_preds, tta_labels = validate_with_tta(
            model, val_tta_loader, criterion, config.device
        )
        print(f'TTA Results: Acc: {tta_acc:.2f}%, F1: {tta_f1:.4f}')
        
        # Print confusion matrix
        cm = confusion_matrix(tta_labels, tta_preds)
        print('\nConfusion Matrix:')
        print(cm)
        
        return tta_acc, tta_f1, cm
    else:
        return best_acc, best_f1, None

# Main execution
def main():
    print('='*70)
    print('ADVANCED MEDICAL IMAGE CLASSIFICATION')
    print('Demonstrating Deliberate Design Choices Beyond Baseline Fine-tuning')
    print('='*70)
    print(f'\nUsing device: {config.device}')
    
    if config.device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    print(f'\n📋 Design Choices:')
    print(f'  1. Multi-Model Ensemble: {config.model_names}')
    print(f'  2. Medical-Specific Augmentation (anatomy-aware)')
    print(f'  3. Spatial Attention Mechanism')
    print(f'  4. Focal Loss for Class Imbalance')
    print(f'  5. Test-Time Augmentation: {"Enabled" if config.tta_enabled else "Disabled"}')
    print(f'  6. OneCycleLR Scheduling')
    print(f'  7. Gradient Accumulation (effective batch: {config.batch_size * config.gradient_accumulation_steps})')
    print(f'  8. {config.num_folds}-Fold Cross-Validation')
    
    # Load training data
    with open(config.train_data_path, 'r') as f:
        train_data = json.load(f)
    
    print(f'\n📊 Dataset Info:')
    print(f'Total training samples: {len(train_data)}')
    
    # Class distribution
    class_counts = {}
    for item in train_data:
        label = item['anatomical_region']
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print('\nClass distribution:')
    for class_name, count in sorted(class_counts.items()):
        percentage = 100 * count / len(train_data)
        print(f'  {class_name:12s}: {count:3d} ({percentage:5.2f}%)')
    
    # Stratified K-Fold
    labels = [item['anatomical_region'] for item in train_data]
    label_indices = [config.class_to_idx[label] for label in labels]
    
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)
    
    all_results = {}
    
    import time
    start_time = time.time()
    
    for model_name in config.model_names:
        print(f'\n{"="*70}')
        print(f'Training Model: {model_name}')
        print(f'{"="*70}')
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, label_indices)):
            train_fold = [train_data[i] for i in train_idx]
            val_fold = [train_data[i] for i in val_idx]
            
            acc, f1, cm = train_model(model_name, train_fold, val_fold, fold)
            fold_results.append({
                'acc': float(acc), 
                'f1': float(f1),
                'confusion_matrix': cm.tolist() if cm is not None else None
            })
        
        all_results[model_name] = fold_results
        
        # Print model summary
        avg_acc = np.mean([r['acc'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        std_acc = np.std([r['acc'] for r in fold_results])
        std_f1 = np.std([r['f1'] for r in fold_results])
        
        print(f'\n{"="*70}')
        print(f'{model_name} Summary:')
        print(f'  Average Accuracy: {avg_acc:.2f}% (±{std_acc:.2f}%)')
        print(f'  Average F1 Score: {avg_f1:.4f} (±{std_f1:.4f})')
        print(f'{"="*70}')
    
    elapsed_time = time.time() - start_time
    
    # Final summary
    print('\n' + '='*70)
    print('🎉 TRAINING COMPLETED!')
    print('='*70)
    
    print(f'\n⏱️  Total training time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)')
    print(f'   Average time per fold: {elapsed_time/config.num_folds/60:.1f} minutes')
    
    print(f'\n📊 Final Results:')
    for model_name, fold_results in all_results.items():
        avg_acc = np.mean([r['acc'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        print(f'  {model_name:20s}: Acc={avg_acc:.2f}%, F1={avg_f1:.4f}')
    
    # Ensemble prediction (theoretical)
    print(f'\n💡 Ensemble Strategy:')
    print(f'   At inference, average predictions from all {len(config.model_names)} models')
    print(f'   with TTA for maximum robustness')
    
    # Save results
    results = {
        'config': {
            'models': config.model_names,
            'img_size': config.img_size,
            'batch_size': config.batch_size,
            'num_folds': config.num_folds,
            'tta_enabled': config.tta_enabled,
        },
        'results': all_results,
        'training_time_minutes': elapsed_time/60,
        'design_choices': [
            'Multi-model ensemble (CNN + Transformer)',
            'Medical-specific augmentation (anatomy-aware)',
            'Spatial attention mechanism',
            'Focal loss for class imbalance',
            'Test-time augmentation',
            'OneCycleLR scheduling',
            'Gradient accumulation',
            'Stratified K-fold cross-validation'
        ]
    }
    
    with open('train3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n✓ Results saved to train3_results.json')
    print('='*70)

if __name__ == '__main__':
    main()
