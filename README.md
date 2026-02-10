# ECG Classification - Multi-Modal Deep Learning

A comprehensive deep learning solution for ECG classification using signals and GAF/MTF images with multiple model architectures.

## Project Overview

This project implements multi-modal ECG classification using:
- **Signal Processing**: 1D CNN, RNN (GRU/LSTM) models for raw ECG signals
- **Image Processing**: 2D CNN (ResNet, VGG, AlexNet) for Gramian Angular Field (GAF) images
- **Fusion Models**: Joint learning combining both signal and image modalities

### Target Classes (5-class classification)
- **NORM**: Normal ECG
- **MI**: Myocardial Infarction
- **STTC**: ST/T change
- **CD**: Conduction Disturbance  
- **HYP**: Hypertrophy

---

## Project Structure

```
ecg_fm/
├── core/
│   ├── data/
│   │   ├── signal_dataloader.py       # ECG signal dataloader
│   │   ├── image_dataloader.py        # GAF/MTF image dataloader
│   │   ├── fusion_dataloader.py       # Joint signal+image dataloader
│   │   ├── ptbxl.py                  # PTB-XL dataset handling
│   │   └── precission_cache.py       # Memory-efficient caching
│   ├── models/
│   │   ├── rnn.py                    # RNN (GRU/LSTM) model
│   │   ├── cnn1d_rnn.py              # 1D CNN + RNN hybrid
│   │   ├── cnn2d_resnet.py           # ResNet for images
│   │   ├── cnn2d_vggnet.py           # VGGNet for images
│   │   ├── cnn2d_alexnet.py          # AlexNet for images
│   │   └── joint_fusion.py           # Fusion model
│   ├── training/
│   │   ├── training.py               # Main training loop
│   │   │── config.py                 # Training config
│   │   └── history.py                # Training history manager
│   └── metrics/
│       └── metrics.py                # Comprehensive metrics computation
├── data/
│   ├── signals/                      # ECG signal batches (train/val/test)
│   ├── images/                       # GAF/MTF images (train/val/test)
│   └── labels/                       # Label files (.npy)
├── checkpoints/                      # Saved model weights
├── results/                          # Evaluation results (auto-generated)
├── config.py                         # Main configuration
├── main.py                           # Training entry point
├── evaluate.py                       # **Evaluation & comparison script**
└── requirements.txt                  # Dependencies
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support, optional)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd ecg_fm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

---

## Quick Start

### 1. Data Preparation

Ensure your data is organized:
```
data/
├── signals/
│   ├── train/
│   ├── validation/
│   └── test/
├── images/
│   ├── train/
│   │   ├── gaf/
│   │   └── mtf/
│   ├── validation/
│   │   ├── gaf/
│   │   └── mtf/
│   └── test/
│       ├── gaf/
│       └── mtf/
└── labels/
    ├── y_train.npy
    ├── y_val.npy
    └── y_test.npy
```

### 2. Configuration

Edit `config.py` to match your setup:
```python
class Config:
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cuda'  # or 'cpu'
    SAVE_DIR = 'checkpoints/'
    NUM_WORKERS = 4
```

### 3. Training

Train a specific model:
```bash
python core/training/train_rnn_training.py      # Train RNN
python core/training/train_cnn2d_training.py    # Train CNN2D
python core/training/train_jointfusion_training.py  # Train Fusion
```

Or use main.py for custom training:
```bash
python main.py
```

### 4. Evaluation & Comparison

**This is the key script for your submission!**

```bash
python evaluate.py
```

#### What `evaluate.py` does:

1. **Loads all trained models** from `checkpoints/`
2. **Evaluates on test set** with appropriate data modalities
3. **Computes comprehensive metrics** (per-class and macro-averaged)
4. **Generates comparison tables**:
   - `01_macro_comparison.csv` - Macro-averaged metrics
   - `02_per_class_f1_comparison.csv` - Per-class F1-scores
   - `03_per_class_sensitivity_comparison.csv` - Sensitivity per class
   - `04_per_class_auc_roc_comparison.csv` - AUC-ROC per class
   - `05_best_model_detailed.csv` - Detailed breakdown of best model

5. **Generates visualizations** in `results/visualizations/`:
   - Confusion matrices
   - ROC curves
   - Precision-Recall curves
   - Model comparison plots
   - Metric heatmaps

6. **Produces summary report**: `results/EVALUATION_SUMMARY.txt`

#### Output Structure

After running `evaluate.py`:
```
results/
├── 01_macro_comparison.csv              # Macro metrics table
├── 02_per_class_f1_comparison.csv       # Per-class F1 scores
├── 03_per_class_sensitivity_comparison.csv
├── 04_per_class_auc_roc_comparison.csv
├── 05_best_model_detailed.csv
├── EVALUATION_SUMMARY.txt               # Summary report
├── metrics_report.txt                   # Detailed text report
└── visualizations/
    ├── model_comparison_bars.png
    ├── heatmap_f1_score.png
    ├── heatmap_sensitivity.png
    ├── heatmap_specificity.png
    ├── heatmap_auc_roc.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── pr_curves.png
```

---

## Model Details

### Signal-based Models

#### RNNModel
- **Input**: ECG signal (batch, 1000, 3) - 1000 time steps, 3 leads
- **Architecture**: GRU/LSTM with optional attention + FC layers
- **Parameters**: Configurable hidden size, number of layers, dropout
- **Use case**: Sequential pattern recognition in time-domain

#### CNN1D_RNN
- **Input**: ECG signal (batch, 1000, 3)
- **Architecture**: 1D convolutions + RNN + attention
- **Use case**: Local and temporal feature extraction

### Image-based Models

#### ResNet
- **Input**: GAF image (batch, 3, 224, 224)
- **Architecture**: Residual blocks with skip connections
- **Features**: Better gradient flow, supports very deep networks
- **Use case**: High-resolution image features

#### VGGNet
- **Input**: GAF image (batch, 3, 224, 224)
- **Architecture**: Stacked 3×3 convolutions
- **Features**: Simple, interpretable, well-balanced
- **Use case**: General-purpose image classification

#### AlexNet
- **Input**: GAF image (batch, 3, 224, 224)
- **Architecture**: Large convolutional layers with dropout
- **Features**: Fast training, good generalization
- **Use case**: Quick baseline model

### Fusion Model

#### JointFusion
- **Input**: Both signal (batch, 1000, 3) and image (batch, 3, 224, 224)
- **Architecture**: Separate encoders + fusion layer
- **Features**: Leverages complementary information
- **Use case**: Best overall performance (usually)

---

## Training Details

### Loss Function
- Binary Cross-Entropy (BCEWithLogitsLoss) for multi-label classification

### Optimizer
- Adam with weight decay (L2 regularization)

### Learning Rate Schedule
- ReduceLROnPlateau: Reduces LR when validation loss plateaus

### Early Stopping
- Stops training after 5 epochs without improvement

### Data Split
- Train: 70% of PTB-XL dataset
- Validation: 15%
- Test: 15%

---

## Key Metrics

### Per-Class Metrics
- **Sensitivity (Recall)**: TP / (TP + FN) - Ability to detect positive cases
- **Specificity**: TN / (TN + FP) - Ability to correctly reject negative cases
- **Precision**: TP / (TP + FP) - Accuracy of positive predictions
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve
- **AUC-PR**: Area under Precision-Recall curve

### Macro-Averaged Metrics
- Equally weighted average across all classes
- Better for imbalanced datasets

---

## Results Interpretation

### Expected Performance
Based on the literature (PTB-XL benchmark):
- **RNN Models**: F1 ≈ 0.75-0.80 (good for temporal patterns)
- **CNN2D Models**: F1 ≈ 0.78-0.85 (good for visual patterns in images)
- **Fusion Models**: F1 ≈ 0.80-0.88 (best overall)

### Class-wise Challenges
- **NORM**: Usually easiest (high sensitivity/specificity)
- **HYP**: Moderate difficulty
- **MI, STTC, CD**: More challenging due to similarity

---

## Usage Examples

### Example 1: Train RNN and evaluate
```python
# Train
python core/training/test_rnn_training.py

# Evaluate
python evaluate.py
```

### Example 2: Train multiple models
```bash
python core/training/train_rnn_training.py
python core/training/train_cnn2d_training.py
python core/training/train_jointfusion_training.py

# Compare all
python evaluate.py
```

### Example 3: Custom evaluation (Python script)
```python
from evaluate import ModelEvaluator
from config import Config

config = Config()
evaluator = ModelEvaluator(config)

models_config = {
    'MyRNNModel': 'checkpoints/RNNModel_best.pth',
    'MyResNet': 'checkpoints/ResNet_best.pth',
}

evaluator.evaluate_all_models(models_config)
evaluator.generate_comparison_tables()
evaluator.generate_visualizations()
```

---

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` in `config.py`
- Use `num_workers=0` if multiprocessing issues occur

### Data Not Loading
- Verify signal files are named `batch_0.npy`, `batch_1.npy`, etc.
- Verify image files are in `gaf/` or `mtf/` subdirectories
- Check label files (.npy) match data count

### Model Not Training
- Verify model checkpoint paths are correct
- Check `DEVICE` setting (should be 'cuda' or 'cpu')
- Ensure sufficient GPU memory (`nvidia-smi`)

---

## Citation & References

### Dataset
- Wagner, P., et al. (2020). "PTB-XL: A large publicly available electrocardiography dataset"

### Methods Referenced
- ResNet: He et al. (2016) - Deep Residual Learning
- VGGNet: Simonyan & Zisserman (2015) - Very Deep Convolutional Networks
- RNN with Attention: Vaswani et al. (2017) - Attention is All You Need

---

## License

This project is provided as-is for educational and evaluation purposes.

---

## Contact & Support

For questions about this implementation, refer to:
- `config.py` - All hyperparameters
- `core/metrics/metrics.py` - Metric definitions
- `core/training/training.py` - Training loop details
- `evaluate.py` - Evaluation pipeline

---

**Last Updated**: February 2026
