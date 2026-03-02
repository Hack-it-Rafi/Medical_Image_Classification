# Medical Image Classification for Anatomical Regions

## Overview

This solution classifies endoscopy images into 7 anatomical regions:

- ear-left, ear-right
- nose-left, nose-right
- throat
- vc-open (vocal cords open), vc-closed (vocal cords closed)

## Key Features for High Accuracy

### 1. **Advanced Model Architecture**

- **Ensemble of 3 state-of-the-art models:**
  - EfficientNet-B3: Efficient and accurate CNN
  - ConvNeXt-Base: Modern ConvNet architecture
  - SwinV2-Base: Vision Transformer for capturing global features

### 2. **Training Techniques**

- **Transfer Learning**: Pre-trained on ImageNet for better feature extraction
- **5-Fold Stratified Cross-Validation**: Ensures robust model performance
- **Mixed Precision Training**: Faster training with CUDA
- **Label Smoothing**: Prevents overfitting (smoothing=0.1)
- **Data Augmentation**:
  - Geometric: Rotation, flip, shift, scale
  - Color: Brightness, contrast adjustments
  - Noise: Gaussian noise, blur
  - CLAHE: Histogram equalization for medical images
  - CoarseDropout: Regularization

### 3. **Optimization**

- **AdamW Optimizer**: Better weight decay regularization
- **Cosine Annealing with Warm Restarts**: Dynamic learning rate
- **Early Stopping**: Prevents overfitting (patience=10 epochs)
- **Gradient Scaling**: Stable mixed precision training

### 4. **Inference Optimization**

- **Test-Time Augmentation (TTA)**: 4 augmentations per image
- **Ensemble Voting**: Average predictions from all models × all folds × TTA
- **Total predictions per image**: 3 models × 5 folds × 4 TTA = 60 predictions averaged

## Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Virtual environment activated

### Installation

Packages are already installed:

- PyTorch (with CUDA support)
- timm (PyTorch Image Models)
- albumentations (advanced augmentations)
- scikit-learn, numpy, Pillow, tqdm

## Usage

### Training

```bash
python train.py
```

This will:

- Train 3 different model architectures
- Use 5-fold cross-validation
- Save best models for each fold: `best_{model_name}_fold{i}.pth`
- Save training results to `training_results.json`
- Expected training time: 4-8 hours on GPU (depends on hardware)

### Inference

```bash
python inference.py
```

This will:

- Load all trained models
- Apply test-time augmentation
- Generate ensemble predictions
- Save results to `submission.json`

## Expected Performance

### Target Metrics

- **Accuracy**: 95%+ (with ensemble)
- **F1-Score**: 0.95+ (weighted)
- **Per-class accuracy**: High across all 7 classes

### Why This Approach Works

1. **Medical Image Specificity**: CLAHE and specialized augmentations help with endoscopy images
2. **Ensemble Diversity**: Different architectures capture different features
3. **Robust Validation**: 5-fold CV ensures generalization
4. **TTA**: Multiple views of each test image improve predictions
5. **Label Smoothing**: Better calibrated confidence scores

## File Structure

```
BUET_TEST/
├── train.py              # Main training script
├── inference.py          # Inference with ensemble + TTA
├── README.md            # This file
├── output/
│   ├── train/
│   │   ├── data.json
│   │   └── imgs/
│   └── test/
│       ├── data.json
│       └── imgs/
├── best_*.pth           # Saved model weights
├── training_results.json # CV results
└── submission.json      # Final predictions
```

## Tips for Maximum Accuracy

1. **Let training complete**: Don't interrupt early stopping
2. **Monitor GPU memory**: Reduce batch_size if OOM errors occur
3. **Check validation accuracy**: Should be >90% per fold
4. **Ensemble is key**: Use all models and folds for inference
5. **CUDA acceleration**: Ensure GPU is being used

## Troubleshooting

### Out of Memory

Reduce batch_size in Config class (try 8 or 4)

### Slow Training

- Ensure CUDA is available: Check GPU usage
- Reduce num_workers if CPU bottleneck

### Low Accuracy

- Check data augmentation isn't too aggressive
- Verify image paths are correct
- Ensure balanced class distribution

## Technical Details

### Model Parameters

- Image size: 384×384 (higher resolution for detail)
- Batch size: 16 (adjust based on GPU memory)
- Learning rate: 1e-4 (with cosine annealing)
- Weight decay: 1e-5
- Max epochs: 50 (with early stopping)

### Hardware Recommendations

- **Minimum**: NVIDIA GPU with 8GB VRAM
- **Recommended**: NVIDIA GPU with 12GB+ VRAM (RTX 3080, V100, A100)
- RAM: 16GB+
- Storage: 10GB free space
