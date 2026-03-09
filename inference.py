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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Paths
    test_data_path = 'output/test/data.json'
    test_img_dir = 'output/test/imgs'
    output_path = 'submission.json'
    
    # Model settings - MUST MATCH train.py!
    model_names = ['convnext_tiny', 'efficientnet_b3']  # Changed to match train.py
    img_size = 384
    num_classes = 7
    batch_size = 32
    num_folds = 4  # Changed to match train.py
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Class names
    class_names = ['ear-left', 'ear-right', 'nose-left', 'nose-right', 'throat', 'vc-closed', 'vc-open']
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}

config = Config()

# Test-time augmentation transforms - NO FLIPS to preserve left/right labels
def get_tta_transforms():
    """Test-time augmentation transforms - anatomically correct (NO HORIZONTAL FLIP!)"""
    transforms_list = []
    
    # Original
    transforms_list.append(A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    # REMOVED: Horizontal flip - would mislabel ear-left/right, nose-left/right!
    
    # Slight rotation
    transforms_list.append(A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Rotate(limit=10, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    # Brightness adjustment
    transforms_list.append(A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    # CLAHE for contrast enhancement
    transforms_list.append(A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.CLAHE(clip_limit=2.0, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    # REMOVED: Horizontal flip + rotation - would mislabel left/right regions!
    
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
# def predict_with_tta(model, image, device):
#     model.eval()
#     tta_transforms = get_tta_transforms()
#     predictions = []
    
#     with torch.no_grad():
#         for transform in tta_transforms:
#             # Apply TTA transform
#             img_array = image.cpu().numpy().transpose(1, 2, 0)
#             # Denormalize
#             mean = np.array([0.485, 0.456, 0.406])
#             std = np.array([0.229, 0.224, 0.225])
#             img_array = img_array * std + mean
#             img_array = (img_array * 255).astype(np.uint8)
            
#             # Apply transform and predict
#             augmented = transform(image=img_array)
#             img_tensor = augmented['image'].unsqueeze(0).to(device)
            
#             output = model(img_tensor)
#             probs = torch.softmax(output, dim=1)
#             predictions.append(probs.cpu().numpy()[0])
    
#     # Average predictions
#     avg_probs = np.mean(predictions, axis=0)
#     return avg_probs

# IMPROVED: Main inference function with ensemble and TTA
def inference():
    print(f'Using device: {config.device}')
    if config.device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Load test data
    with open(config.test_data_path, 'r') as f:
        test_data = json.load(f)
    
    print(f'\nTotal test samples: {len(test_data)}')
    print(f'Using Ensemble of {len(config.model_names)} architectures × {config.num_folds} folds = {len(config.model_names) * config.num_folds} models')
    print(f'Using Test-Time Augmentation with 4 variations per model (NO horizontal flip to preserve left/right)')
    
    # ========================================
    # LOAD ALL MODELS ONCE (EFFICIENT!)
    # ========================================
    print('\n⏳ Loading all models...')
    loaded_models = []
    model_info = []  # Track which model each belongs to
    
    for model_name in config.model_names:
        for fold in range(config.num_folds):
            model_path = f'best_{model_name}_fold{fold}.pth'
            
            if not os.path.exists(model_path):
                print(f'⚠️  Warning: {model_path} not found, skipping...')
                continue
            
            # Load model
            model = EnsembleModel(model_name, config.num_classes).to(config.device)
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            model.eval()
            loaded_models.append(model)
            model_info.append({'name': model_name, 'fold': fold})
            print(f'✓ Loaded {model_path}')
    
    print(f'\n✓ Successfully loaded {len(loaded_models)} models')
    
    if len(loaded_models) == 0:
        raise RuntimeError('No models found! Please check model paths.')
    
    # Get TTA transforms
    tta_transforms = get_tta_transforms()
    
    # Prepare results
    results = []
    all_confidences = []
    all_predictions = []  # Store predicted labels
    all_ground_truth = []  # Store true labels
    all_predictions_proba = []  # Store probability predictions for AUROC
    
    # Track predictions per model architecture
    model_predictions = {name: [] for name in config.model_names}
    model_confidences = {name: [] for name in config.model_names}
    model_predictions_proba = {name: [] for name in config.model_names}  # Store probabilities per model
    
    # ========================================
    # PROCESS EACH TEST IMAGE (EFFICIENT!)
    # ========================================
    print(f'\n🔄 Processing {len(test_data)} test images...\n')
    
    for item in tqdm(test_data, desc='Processing test images'):
        img_path = os.path.join(config.test_img_dir, item['path'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Store ground truth label
        true_label = item['anatomical_region']
        all_ground_truth.append(true_label)
        
        all_predictions_prob = []
        
        # Track predictions per model type
        model_type_predictions = {name: [] for name in config.model_names}
        
        # Use all loaded models (NO RELOADING!)
        for idx, model in enumerate(loaded_models):
            fold_predictions = []
            
            with torch.no_grad():
                for transform in tta_transforms:
                    augmented = transform(image=image)
                    img_tensor = augmented['image'].unsqueeze(0).to(config.device)
                    
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    fold_predictions.append(probs.cpu().numpy()[0])
            
            # Average TTA predictions for this model
            avg_fold_pred = np.mean(fold_predictions, axis=0)
            all_predictions_prob.append(avg_fold_pred)
            
            # Track by model type
            model_name = model_info[idx]['name']
            model_type_predictions[model_name].append(avg_fold_pred)
        
        # Get predictions for each model architecture separately
        for model_name in config.model_names:
            if model_type_predictions[model_name]:
                model_avg_probs = np.mean(model_type_predictions[model_name], axis=0)
                predicted_class = np.argmax(model_avg_probs)
                predicted_label = config.idx_to_class[predicted_class]
                model_predictions[model_name].append(predicted_label)
                model_confidences[model_name].append(float(model_avg_probs[predicted_class]))
        
        # Average all predictions (ensemble)
        if all_predictions_prob:
            final_probs = np.mean(all_predictions_prob, axis=0)
            predicted_class = np.argmax(final_probs)
            predicted_label = config.idx_to_class[predicted_class]
            confidence = final_probs[predicted_class]
            all_predictions_proba.append(final_probs)  # Store final probabilities
        else:
            # Fallback if no models found
            predicted_label = 'throat'  # Default to most uncertain class
            confidence = 0.0
        
        # Store predicted label
        all_predictions.append(predicted_label)
        
        results.append({
            'path': item['path'],
            'anatomical_region': predicted_label,
            'confidence': float(confidence),
            'ground_truth': true_label  # Include ground truth in results
        })
        all_confidences.append(confidence)
    
    # Clean up models after inference
    print('\n🧹 Cleaning up models...')
    for model in loaded_models:
        del model
    torch.cuda.empty_cache()
    print('✓ Memory cleaned')
    
    # Save results as JSON
    with open(config.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save as CSV for submission
    results_df = pd.DataFrame({
        'image_path': [r['path'] for r in results],
        'anatomical_region': [r['anatomical_region'] for r in results]
    })
    results_df.to_csv('submission.csv', index=False)
    
    print(f'\n✓ Predictions saved to {config.output_path}')
    print(f'✓ Submission CSV saved to submission.csv')
    
    # Print statistics
    pred_counts = Counter([r['anatomical_region'] for r in results])
    print('\nPrediction distribution:')
    for class_name, count in sorted(pred_counts.items()):
        print(f'  {class_name}: {count}')
    
    avg_confidence = np.mean(all_confidences)
    min_confidence = np.min(all_confidences)
    max_confidence = np.max(all_confidences)
    low_confidence_count = sum(1 for c in all_confidences if c < 0.8)
    
    print(f'\nConfidence statistics:')
    print(f'  Average: {avg_confidence:.4f}')
    print(f'  Min: {min_confidence:.4f}')
    print(f'  Max: {max_confidence:.4f}')
    print(f'  Low confidence (<0.8): {low_confidence_count} samples')
    
    # ========================================
    # COMPUTE TEST METRICS (REQUIRED)
    # ========================================
    print('\n' + '='*60)
    print('TEST SET EVALUATION METRICS')
    print('='*60)
    
    # Overall metrics
    overall_accuracy = accuracy_score(all_ground_truth, all_predictions)
    weighted_precision = precision_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
    weighted_recall = recall_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
    weighted_f1 = f1_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
    
    # Calculate AUROC score (multi-class)
    # Binarize the ground truth labels for multi-class AUROC calculation
    y_true_binarized = label_binarize(all_ground_truth, classes=config.class_names)
    all_predictions_proba_array = np.array(all_predictions_proba)
    
    try:
        # Calculate macro and weighted AUROC
        auroc_macro = roc_auc_score(y_true_binarized, all_predictions_proba_array, average='macro', multi_class='ovr')
        auroc_weighted = roc_auc_score(y_true_binarized, all_predictions_proba_array, average='weighted', multi_class='ovr')
        
        # Calculate per-class AUROC
        per_class_auroc = []
        for i in range(len(config.class_names)):
            try:
                auroc_class = roc_auc_score(y_true_binarized[:, i], all_predictions_proba_array[:, i])
                per_class_auroc.append(auroc_class)
            except:
                per_class_auroc.append(0.0)  # Handle case where class is not present
    except Exception as e:
        print(f'\n⚠️  Warning: Could not calculate AUROC: {e}')
        auroc_macro = 0.0
        auroc_weighted = 0.0
        per_class_auroc = [0.0] * len(config.class_names)
    
    print(f'\n📊 Overall Metrics:')
    print(f'  Accuracy:           {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)')
    print(f'  Weighted Precision: {weighted_precision:.4f}')
    print(f'  Weighted Recall:    {weighted_recall:.4f}')
    print(f'  Weighted F1-Score:  {weighted_f1:.4f}')
    print(f'  AUROC (Macro):      {auroc_macro:.4f}')
    print(f'  AUROC (Weighted):   {auroc_weighted:.4f}')
    
    # Per-class metrics
    print(f'\n📋 Per-Class Metrics:')
    print(f'{"Class":<15} {"Precision":<12} {"Recall":<12} {"F1-Score":<12} {"AUROC":<12} {"Support":<10}')
    print('-' * 77)
    
    per_class_precision = precision_score(all_ground_truth, all_predictions, average=None, labels=config.class_names, zero_division=0)
    per_class_recall = recall_score(all_ground_truth, all_predictions, average=None, labels=config.class_names, zero_division=0)
    per_class_f1 = f1_score(all_ground_truth, all_predictions, average=None, labels=config.class_names, zero_division=0)
    
    # Count support for each class
    class_support = {cls: all_ground_truth.count(cls) for cls in config.class_names}
    
    for i, class_name in enumerate(config.class_names):
        print(f'{class_name:<15} {per_class_precision[i]:<12.4f} {per_class_recall[i]:<12.4f} {per_class_f1[i]:<12.4f} {per_class_auroc[i]:<12.4f} {class_support[class_name]:<10}')
    
    # Detailed classification report
    print('\n' + '='*60)
    print('DETAILED CLASSIFICATION REPORT')
    print('='*60)
    print(classification_report(all_ground_truth, all_predictions, 
                                target_names=config.class_names, 
                                digits=4, 
                                zero_division=0))
    
    # ========================================
    # CONFUSION MATRIX
    # ========================================
    print('\n' + '='*60)
    print('CONFUSION MATRIX')
    print('='*60)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_ground_truth, all_predictions, labels=config.class_names)
    
    # Print confusion matrix as text
    print('\nConfusion Matrix (rows=actual, columns=predicted):')
    print(f'{"":>15}', end='')
    for class_name in config.class_names:
        print(f'{class_name[:10]:>12}', end='')
    print()
    print('-' * (15 + 12 * len(config.class_names)))
    
    for i, class_name in enumerate(config.class_names):
        print(f'{class_name:<15}', end='')
        for j in range(len(config.class_names)):
            print(f'{cm[i][j]:>12}', end='')
        print()
    
    # Create and save confusion matrix visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.class_names, 
                yticklabels=config.class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print('\n✓ Confusion matrix plot saved to confusion_matrix.png')
    
    # Create normalized confusion matrix (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=config.class_names, 
                yticklabels=config.class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title('Normalized Confusion Matrix - Test Set (Percentages)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    print('✓ Normalized confusion matrix plot saved to confusion_matrix_normalized.png')
    
    # Save metrics to file
    metrics_dict = {
        'overall_metrics': {
            'accuracy': float(overall_accuracy),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'auroc_macro': float(auroc_macro),
            'auroc_weighted': float(auroc_weighted)
        },
        'per_class_metrics': {},
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist()
    }
    
    for i, class_name in enumerate(config.class_names):
        metrics_dict['per_class_metrics'][class_name] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1_score': float(per_class_f1[i]),
            'auroc': float(per_class_auroc[i]),
            'support': int(class_support[class_name])
        }
    
    # ========================================
    # MODEL COMPARISON SECTION
    # ========================================
    print('\n' + '='*60)
    print('MODEL ARCHITECTURE COMPARISON')
    print('='*60)
    
    model_comparison = {}
    
    for model_name in config.model_names:
        if not model_predictions[model_name]:
            continue
            
        # Calculate metrics for this model
        model_acc = accuracy_score(all_ground_truth, model_predictions[model_name])
        model_prec = precision_score(all_ground_truth, model_predictions[model_name], 
                                     average='weighted', zero_division=0)
        model_rec = recall_score(all_ground_truth, model_predictions[model_name], 
                                average='weighted', zero_division=0)
        model_f1 = f1_score(all_ground_truth, model_predictions[model_name], 
                           average='weighted', zero_division=0)
        
        # Calculate per-class metrics
        model_per_class_prec = precision_score(all_ground_truth, model_predictions[model_name], 
                                               average=None, labels=config.class_names, zero_division=0)
        model_per_class_rec = recall_score(all_ground_truth, model_predictions[model_name], 
                                          average=None, labels=config.class_names, zero_division=0)
        model_per_class_f1 = f1_score(all_ground_truth, model_predictions[model_name], 
                                     average=None, labels=config.class_names, zero_division=0)
        
        # Confidence statistics
        avg_conf = np.mean(model_confidences[model_name])
        
        # Store in comparison dict
        model_comparison[model_name] = {
            'accuracy': float(model_acc),
            'weighted_precision': float(model_prec),
            'weighted_recall': float(model_rec),
            'weighted_f1': float(model_f1),
            'avg_confidence': float(avg_conf),
            'per_class_precision': model_per_class_prec.tolist(),
            'per_class_recall': model_per_class_rec.tolist(),
            'per_class_f1': model_per_class_f1.tolist()
        }
        
        print(f'\n📊 {model_name.upper()}:')
        print(f'  Accuracy:           {model_acc:.4f} ({model_acc*100:.2f}%)')
        print(f'  Weighted Precision: {model_prec:.4f}')
        print(f'  Weighted Recall:    {model_rec:.4f}')
        print(f'  Weighted F1-Score:  {model_f1:.4f}')
        print(f'  Avg Confidence:     {avg_conf:.4f}')
        
        # Print per-class F1 scores
        print(f'\n  Per-Class F1-Scores:')
        for i, class_name in enumerate(config.class_names):
            print(f'    {class_name:<15}: {model_per_class_f1[i]:.4f}')
    
    # Comparison summary
    print('\n' + '-'*60)
    print('COMPARISON SUMMARY:')
    print('-'*60)
    
    if len(model_comparison) == 2:
        model_names_list = list(model_comparison.keys())
        model1, model2 = model_names_list[0], model_names_list[1]
        
        acc_diff = model_comparison[model1]['accuracy'] - model_comparison[model2]['accuracy']
        f1_diff = model_comparison[model1]['weighted_f1'] - model_comparison[model2]['weighted_f1']
        conf_diff = model_comparison[model1]['avg_confidence'] - model_comparison[model2]['avg_confidence']
        
        print(f'\n{model1} vs {model2}:')
        print(f'  Accuracy Difference:    {acc_diff:+.4f} ({abs(acc_diff)*100:.2f}%)')
        print(f'  F1-Score Difference:    {f1_diff:+.4f}')
        print(f'  Confidence Difference:  {conf_diff:+.4f}')
        
        if abs(acc_diff) > 0.01:
            better_model = model1 if acc_diff > 0 else model2
            print(f'\n  🏆 Best Model by Accuracy: {better_model}')
        else:
            print(f'\n  ⚖️  Models perform similarly (difference < 1%)')
        
        # Agreement analysis
        agreements = sum(1 for i in range(len(all_ground_truth)) 
                        if model_predictions[model1][i] == model_predictions[model2][i])
        agreement_rate = agreements / len(all_ground_truth)
        
        both_correct = sum(1 for i in range(len(all_ground_truth))
                          if model_predictions[model1][i] == all_ground_truth[i] 
                          and model_predictions[model2][i] == all_ground_truth[i])
        both_wrong = sum(1 for i in range(len(all_ground_truth))
                        if model_predictions[model1][i] != all_ground_truth[i] 
                        and model_predictions[model2][i] != all_ground_truth[i])
        one_correct = len(all_ground_truth) - both_correct - both_wrong
        
        print(f'\n  Model Agreement:')
        print(f'    Agreement Rate:     {agreement_rate:.2%}')
        print(f'    Both Correct:       {both_correct} ({both_correct/len(all_ground_truth)*100:.1f}%)')
        print(f'    Both Wrong:         {both_wrong} ({both_wrong/len(all_ground_truth)*100:.1f}%)')
        print(f'    One Correct:        {one_correct} ({one_correct/len(all_ground_truth)*100:.1f}%)')
        
        # Per-class comparison
        print(f'\n  Per-Class F1-Score Comparison:')
        print(f'  {"Class":<15} {model1[:10]:<12} {model2[:10]:<12} {"Difference":<12}')
        print('  ' + '-'*54)
        
        for i, class_name in enumerate(config.class_names):
            f1_1 = model_comparison[model1]['per_class_f1'][i]
            f1_2 = model_comparison[model2]['per_class_f1'][i]
            diff = f1_1 - f1_2
            marker = '✓' if diff > 0.05 else ('✗' if diff < -0.05 else '≈')
            print(f'  {class_name:<15} {f1_1:<12.4f} {f1_2:<12.4f} {diff:+.4f} {marker}')
    
    # Create comparison visualization
    if len(model_comparison) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall Metrics Comparison
        metrics = ['accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for idx, model_name in enumerate(model_comparison.keys()):
            values = [model_comparison[model_name][m] for m in metrics]
            axes[0, 0].bar(x + idx*width, values, width, label=model_name, alpha=0.8)
        
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Overall Performance Comparison')
        axes[0, 0].set_xticks(x + width/2)
        axes[0, 0].set_xticklabels(metric_names)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 1.0])
        
        # 2. Per-Class F1-Score Comparison
        x_classes = np.arange(len(config.class_names))
        
        for idx, model_name in enumerate(model_comparison.keys()):
            f1_scores = model_comparison[model_name]['per_class_f1']
            axes[0, 1].bar(x_classes + idx*width, f1_scores, width, label=model_name, alpha=0.8)
        
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_title('Per-Class F1-Score Comparison')
        axes[0, 1].set_xticks(x_classes + width/2)
        axes[0, 1].set_xticklabels(config.class_names, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].set_ylim([0, 1.0])
        
        # 3. Confidence Distribution
        for model_name in model_comparison.keys():
            axes[1, 0].hist(model_confidences[model_name], bins=20, alpha=0.6, label=model_name, edgecolor='black')
        
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Confidence Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Agreement Visualization (if 2 models)
        if len(model_comparison) == 2:
            agreement_data = [both_correct, both_wrong, one_correct]
            agreement_labels = ['Both Correct', 'Both Wrong', 'One Correct']
            colors = ['#2ecc71', '#e74c3c', '#f39c12']
            
            axes[1, 1].pie(agreement_data, labels=agreement_labels, autopct='%1.1f%%',
                          colors=colors, startangle=90)
            axes[1, 1].set_title('Model Agreement Analysis')
        else:
            axes[1, 1].text(0.5, 0.5, 'Agreement analysis\nrequires 2 models', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print(f'\n✓ Model comparison plot saved to model_comparison.png')
    
    # Add model comparison to metrics dict
    metrics_dict['model_comparison'] = model_comparison
    
    with open('test_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f'\n✓ Test metrics (with model comparison) saved to test_metrics.json')
    print('='*60)

if __name__ == '__main__':
    inference()
