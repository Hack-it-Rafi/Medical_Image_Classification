# Medical Image Classification for Anatomical Regions

## Overview

This solution classifies endoscopy images into 7 anatomical regions:

- ear-left, ear-right
- nose-left, nose-right
- throat
- vc-open (vocal cords open), vc-closed (vocal cords closed)

## Key Features for High Accuracy

### 1. **Advanced Model Architecture**

- **Ensemble of 2 state-of-the-art models:**
  - ConvNeXt-Tiny: Modern ConvNet architecture with excellent efficiency
  - EfficientNet-B3: Efficient and accurate CNN

### 2. **Training Techniques**

- **Transfer Learning**: Pre-trained on ImageNet for better feature extraction
- **4-Fold Stratified Cross-Validation**: Ensures robust model performance
- **Mixed Precision Training**: Faster training with CUDA
- **Weighted Loss & Sampling**: Handles class imbalance effectively
- **Label Smoothing**: Prevents overfitting (smoothing=0.1)
- **Medical-Specific Augmentation** (NO horizontal/vertical flips to preserve left/right anatomical labels):
  - Small rotations (±15°): Anatomical orientation matters
  - Medical imaging adjustments: Gamma, sharpening, CLAHE
  - Color: Brightness, contrast, HSV adjustments
  - Noise: Gaussian, ISO noise (simulates camera sensors)
  - Regularization: CoarseDropout

### 3. **Optimization**

- **AdamW Optimizer**: Better weight decay regularization
- **Cosine Annealing with Warm Restarts**: Dynamic learning rate
- **Early Stopping**: Prevents overfitting (patience=10 epochs)
- **Gradient Scaling**: Stable mixed precision training

### 4. **Inference Optimization**

- **Test-Time Augmentation (TTA)**: 4 augmentations per image (NO flips for anatomical correctness)
- **Ensemble Voting**: Average predictions from all models × all folds × TTA
- **Total predictions per image**: 2 models × 4 folds × 4 TTA = 32 predictions averaged
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUROC, Confusion Matrix
- **Model Comparison**: Side-by-side architecture performance analysis

## Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Jupyter Notebook or JupyterLab
- Virtual environment activated

### Installation

Required packages:

- PyTorch (with CUDA support)
- timm (PyTorch Image Models)
- albumentations (advanced augmentations)
- scikit-learn, numpy, Pillow, pandas
- tqdm (notebook version)
- matplotlib, seaborn (for visualizations)

## Usage

### Using Jupyter Notebook (Recommended)

1. **Open the combined notebook**:

   ```bash
   jupyter notebook combined.ipynb
   ```

   or

   ```bash
   jupyter lab combined.ipynb
   ```

2. **Run cells in order**:

   - **Training Section**: Cells 1-12 (imports, config, model training)
   - **Inference Section**: Cells 13-16 (load models, predict, evaluate)

3. **Important Notes for Notebooks**:
   - The notebook automatically sets `num_workers=0` for DataLoader to prevent multiprocessing hangs
   - Uses `tqdm.notebook` for proper progress bars in Jupyter
   - If training appears stuck, check the cell output - it may be loading/processing silently
   - You can interrupt training with the stop button and resume from saved checkpoints

### Alternative: Command Line (Legacy)

Training:

```bash
python train.py
```

Inference:

```bash
python inference.py
```

## Training Process

### Resume Training Feature

The notebook supports resuming training from a specific model and fold:

```python
# In the main() function, modify these variables:
resume_model = None  # Set to model name (e.g., 'convnext_tiny') or None to start fresh
resume_from_fold = 0  # Set fold number to resume from
```

This will:

- Train 2 different model architectures
- Use 4-fold cross-validation
- Save best models for each fold: `best_{model_name}_fold{i}.pth`
- Save training results to `training_results.json`
- Expected training time: 3-6 hours on GPU (depends on hardware)

### Troubleshooting Training Stuck Issue

**If training appears stuck at "Epoch 1/40" with "Training: 0%":**

1. **In Jupyter Notebooks**: This is usually caused by DataLoader multiprocessing

   - ✅ **FIXED**: The notebook now uses `num_workers=0` by default
   - Check that tqdm shows as `tqdm.notebook` not regular `tqdm`

2. **Check GPU Activity**:

   - Open terminal and run: `nvidia-smi` (watch for GPU utilization)
   - If GPU is active, training is working (just slow on first epoch)

3. **First Epoch is Always Slowest**:

   - Model initialization and first forward pass take time
   - Expect 2-3 minutes for first batch on slower GPUs
   - Subsequent epochs will be much faster

4. **Reduce Batch Size if Needed**:
   - If stuck with no GPU activity, you may have OOM
   - Reduce `batch_size` from 16 to 8 or 4 in Config

## Inference & Evaluation

The inference section provides comprehensive evaluation:

### Output Files

1. **submission.json**: Predictions with confidence scores and ground truth
2. **submission.csv**: Simple format for competition submission
3. **test_metrics.json**: Detailed metrics including:

   - Overall: Accuracy, Precision, Recall, F1, AUROC
   - Per-class: All metrics + support count
   - Model comparison: Individual model performances
   - Confusion matrices (raw and normalized)

4. **Visualizations**:
   - `confusion_matrix.png`: Raw counts
   - `confusion_matrix_normalized.png`: Percentage-based
   - `model_comparison.png`: 4-panel comparison (if 2+ models)

### Metrics Reported

- **Overall Metrics**: Accuracy, Weighted Precision/Recall/F1, AUROC (Macro & Weighted)
- **Per-Class Metrics**: Precision, Recall, F1, AUROC, Support for each anatomical region
- **Confusion Matrix**: Both raw counts and normalized percentages
- **Model Comparison**: Side-by-side performance of different architectures
- **Agreement Analysis**: How often models agree/disagree (if 2 models)

## Expected Performance

### Target Metrics

- **Accuracy**: 95%+ (with ensemble)
- **F1-Score**: 0.95+ (weighted)
- **AUROC**: 0.95+ (macro average)
- **Per-class accuracy**: High across all 7 classes

### Why This Approach Works

1. **Medical Image Specificity**: CLAHE and specialized augmentations help with endoscopy images
2. **Anatomical Correctness**: NO horizontal/vertical flips preserves left/right labels
3. **Ensemble Diversity**: Different architectures capture different features
4. **Robust Validation**: 4-fold CV ensures generalization
5. **TTA**: Multiple views of each test image improve predictions
6. **Class Imbalance Handling**: Weighted sampling and weighted loss
7. **Label Smoothing**: Better calibrated confidence scores

## File Structure

```
BUET_TEST/
├── combined.ipynb        # Main notebook (training + inference)
├── train.py             # Standalone training script
├── inference.py         # Standalone inference script
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
├── submission.json      # Final predictions
├── submission.csv       # CSV format predictions
├── test_metrics.json    # Detailed evaluation metrics
├── confusion_matrix*.png # Confusion matrix visualizations
└── model_comparison.png # Model performance comparison
```

## Tips for Maximum Accuracy

1. **Use Jupyter Notebook**: Better for interactive development and debugging
2. **Let training complete**: Don't interrupt early stopping
3. **Monitor GPU memory**: Reduce batch_size if OOM errors occur
4. **Check validation accuracy**: Should be >85% per fold
5. **Ensemble is key**: Use all models and folds for inference
6. **CUDA acceleration**: Ensure GPU is being used
7. **Review confusion matrix**: Identify problematic class pairs

## Troubleshooting

### Training Stuck at "Epoch 1/40"

- ✅ **FIXED in notebook**: Uses `num_workers=0` to avoid multiprocessing issues
- Wait 2-3 minutes - first epoch initialization is slow
- Check GPU activity with `nvidia-smi`
- If using standalone script, manually set `num_workers=0` in DataLoader

### Out of Memory

Reduce batch_size in Config class:

```python
batch_size = 8  # or 4 for very limited VRAM
```

### Slow Training

- Ensure CUDA is available: Check `torch.cuda.is_available()`
- Reduce num_workers if CPU bottleneck (already 0 in notebook)
- Close other GPU-intensive applications

### Low Accuracy

- Check data augmentation isn't too aggressive
- Verify image paths are correct
- Ensure balanced class distribution (check training output)
- Review per-class metrics to identify problematic classes

### Progress Bar Not Updating in Notebook

- Ensure using `tqdm.notebook` not regular `tqdm`
- Try restarting kernel and running all cells fresh
- Check Jupyter extensions are up to date

## Technical Details

### Model Parameters

- Image size: 384×384 (higher resolution for medical detail)
- Batch size: 16 (adjust based on GPU memory)
- Learning rate: 1e-4 (with cosine annealing)
- Weight decay: 1e-5
- Max epochs: 40 (with early stopping at 10 epochs patience)
- Number of models: 2 architectures × 4 folds = 8 models

### Hardware Recommendations

- **Minimum**: NVIDIA GPU with 4GB VRAM (RTX 3050)
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 3070)
- **Optimal**: NVIDIA GPU with 12GB+ VRAM (RTX 3080, V100, A100)
- RAM: 16GB+
- Storage: 10GB free space

### Medical Imaging Best Practices

1. **NO Horizontal/Vertical Flips**: Would swap ear-left ↔ ear-right, nose-left ↔ nose-right
2. **Limited Rotation**: Only ±15° to preserve anatomical orientation
3. **Medical-Specific Preprocessing**: CLAHE, sharpening, gamma adjustment
4. **Weighted Loss**: Handles class imbalance (throat: 81 samples vs nose-right: 325 samples)
5. **Test-Time Augmentation**: Anatomically safe transformations only

## Results Interpretation

After inference, review:

1. **Overall accuracy**: Should be >90%
2. **Per-class F1-scores**: Identify weak classes
3. **Confusion matrix**: Which pairs are confused (e.g., ear-left vs ear-right)
4. **Model comparison**: Which architecture performs best
5. **Confidence distribution**: Low confidence samples may need review

## Citation

If you use this code, please consider citing:

- ConvNeXt: Liu et al., "A ConvNet for the 2020s"
- EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling"
- Albumentations: Buslaev et al., "Albumentations: Fast and Flexible Image Augmentations"
