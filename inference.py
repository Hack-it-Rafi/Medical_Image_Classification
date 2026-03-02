import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Paths
    test_data_path = 'output/test/data.json'
    test_img_dir = 'output/test/imgs'
    output_path = 'submission.json'
    
    # Model settings
    model_names = ['efficientnet_b3', 'convnext_base', 'swinv2_base_window12_192_22k']
    img_size = 384
    num_classes = 7
    batch_size = 32
    num_folds = 5
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Class names
    class_names = ['ear-left', 'ear-right', 'nose-left', 'nose-right', 'throat', 'vc-closed', 'vc-open']
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}

config = Config()

# Test-time augmentation transforms
def get_tta_transforms():
    transforms_list = []
    
    # Original
    transforms_list.append(A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    # Horizontal flip
    transforms_list.append(A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    # Vertical flip
    transforms_list.append(A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    # Both flips
    transforms_list.append(A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    return transforms_list

# Test dataset
class TestDataset(Dataset):
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
        
        # Apply transform
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, item['path']

# Model definition
class EnsembleModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.model(x)

# Inference with TTA
def predict_with_tta(model, image, device):
    model.eval()
    tta_transforms = get_tta_transforms()
    predictions = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            # Apply TTA transform
            img_array = image.cpu().numpy().transpose(1, 2, 0)
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = img_array * std + mean
            img_array = (img_array * 255).astype(np.uint8)
            
            # Apply transform and predict
            augmented = transform(image=img_array)
            img_tensor = augmented['image'].unsqueeze(0).to(device)
            
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy()[0])
    
    # Average predictions
    avg_probs = np.mean(predictions, axis=0)
    return avg_probs

# Main inference function
def inference():
    print(f'Using device: {config.device}')
    if config.device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Load test data
    with open(config.test_data_path, 'r') as f:
        test_data = json.load(f)
    
    print(f'\nTotal test samples: {len(test_data)}')
    
    # Prepare results
    results = []
    
    # Process each test image
    for item in tqdm(test_data, desc='Processing test images'):
        img_path = os.path.join(config.test_img_dir, item['path'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        all_predictions = []
        
        # Ensemble over all models and folds
        for model_name in config.model_names:
            for fold in range(config.num_folds):
                model_path = f'best_{model_name}_fold{fold}.pth'
                
                if not os.path.exists(model_path):
                    print(f'Warning: Model not found: {model_path}')
                    continue
                
                # Load model
                model = EnsembleModel(model_name, config.num_classes).to(config.device)
                model.load_state_dict(torch.load(model_path, map_location=config.device))
                model.eval()
                
                # Apply TTA transforms
                tta_transforms = get_tta_transforms()
                fold_predictions = []
                
                with torch.no_grad():
                    for transform in tta_transforms:
                        augmented = transform(image=image)
                        img_tensor = augmented['image'].unsqueeze(0).to(config.device)
                        
                        output = model(img_tensor)
                        probs = torch.softmax(output, dim=1)
                        fold_predictions.append(probs.cpu().numpy()[0])
                
                # Average TTA predictions for this fold
                avg_fold_pred = np.mean(fold_predictions, axis=0)
                all_predictions.append(avg_fold_pred)
        
        # Average all predictions (ensemble)
        if all_predictions:
            final_probs = np.mean(all_predictions, axis=0)
            predicted_class = np.argmax(final_probs)
            predicted_label = config.idx_to_class[predicted_class]
            confidence = final_probs[predicted_class]
        else:
            # Fallback if no models found
            predicted_label = item['anatomical_region']
            confidence = 0.0
        
        results.append({
            'path': item['path'],
            'anatomical_region': predicted_label,
            'confidence': float(confidence)
        })
    
    # Save results
    with open(config.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n✓ Predictions saved to {config.output_path}')
    
    # Print statistics
    pred_counts = Counter([r['anatomical_region'] for r in results])
    print('\nPrediction distribution:')
    for class_name, count in sorted(pred_counts.items()):
        print(f'  {class_name}: {count}')
    
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f'\nAverage confidence: {avg_confidence:.4f}')

if __name__ == '__main__':
    inference()
