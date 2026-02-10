# ECG Classification - Multi-Modal Deep Learning

A comprehensive deep learning solution for ECG classification using signals and GAF/MTF images with multiple model architectures. This project implements multi-modal learning combining ECG time-series signals with Gramian Angular Field (GAF) visualizations for robust 5-class cardiac condition classification.

## Project Overview

### Architecture
This project implements multi-modal ECG classification using:
- **Signal-Based Models**: RNN (GRU/LSTM with attention) and 1D CNN for temporal pattern recognition in raw ECG signals
- **Image-Based Models**: 2D CNN (ResNet, VGG, AlexNet) for visual pattern recognition in GAF/MTF images
- **Fusion Models**: Joint learning framework combining both signal and image modalities for enhanced performance

### Target Classes (5-class Classification)
1. **NORM**: Normal Electrocardiogram
2. **MI**: Myocardial Infarction (Heart attack)
3. **STTC**: ST/T-wave Changes
4. **CD**: Conduction Disturbance  
5. **HYP**: Hypertrophy (Enlarged heart)

### Dataset
- **Source**: PTB-XL (PhysioNet) - Large publicly available ECG dataset
- **Records**: ~21,000 ECG recordings
- **Sampling Rate**: 100 Hz
- **Signal Duration**: 10 seconds (1000 time steps)
- **Leads**: 12-lead ECG (reduced to 3 leads in this implementation)

---

## Installation & Setup

### Prerequisites
- **Python**: 3.8 or higher
- **GPU** (optional but recommended): CUDA 11.0+ for faster training
- **Memory**: 8GB RAM minimum (16GB+ recommended for batch processing)

### Step 1: Clone or Download Repository
```bash
cd ecg_fm
```

### Step 2: Create Virtual Environment
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Setup
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Configuration

### Main Configuration File: `config.py`

```python
class Config:
    # Data paths (adjust to your setup)
    TRAIN_SIGNAL_PATH = 'data/signals/train/'
    TRAIN_IMAGE_PATH = 'data/images/train/'
    TRAIN_LABEL_PATH = 'data/labels/y_train.npy'
    
    VAL_SIGNAL_PATH = 'data/signals/validation/'
    VAL_IMAGE_PATH = 'data/images/validation/'
    VAL_LABEL_PATH = 'data/labels/y_val.npy'
    
    TEST_SIGNAL_PATH = 'data/signals/test/'
    TEST_IMAGE_PATH = 'data/images/test/'
    TEST_LABEL_PATH = 'data/labels/y_test.npy'
    
    # Model hyperparameters
    HIDDEN_SIZE = 128              # RNN hidden units / CNN filters
    NUM_LAYERS = 2                 # RNN layers / CNN depth
    DROPOUT = 0.3                  # Regularization
    
    # Training parameters
    BATCH_SIZE = 32                # Adjust based on GPU memory
    NUM_EPOCHS = 50                # Training iterations
    LEARNING_RATE = 0.001          # Adam optimizer learning rate
    WEIGHT_DECAY = 1e-4            # L2 regularization
    PATIENCE = 5                   # Early stopping patience
    SAVE_BEST = True               # Save best checkpoint
    
    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = 'checkpoints/'
    NUM_WORKERS = 4                # Data loading workers (use 0 on Windows if issues)
    
    # Classes
    CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    NUM_CLASSES = 5
```

---

## Quick Start Guide

### 1. Data Preparation

Ensure your data follows this structure:
```
data/
├── signals/
│   ├── train/
│   │   ├── batch_0.npy
│   │   ├── batch_1.npy
│   │   └── ...
│   ├── validation/
│   └── test/
├── images/
│   ├── train/
│   │   ├── gaf/
│   │   │   ├── batch_0.npy
│   │   │   └── ...
│   │   └── mtf/
│   │       └── ...
│   ├── validation/
│   └── test/
└── labels/
    ├── y_train.npy  # Shape: (N, 5) - one-hot encoded
    ├── y_val.npy
    └── y_test.npy
```

**Signal format**: `(batch_size, 1000, 3)` - 1000 time steps, 3 leads
**Image format**: `(batch_size, 3, 224, 224)` - RGB images, 224×224 pixels

### 2. Train Models

Train individual models:
```bash
# Train RNN-based model (signal)
python core/training/train_rnn_training.py

# Train CNN2D model (image)
python core/training/train_cnn2d_training.py

# Train Fusion model (signal + image)
python core/training/train_jointfusion_training.py
```

**Expected output**: `.pth` checkpoint files in `checkpoints/` directory

### 3. Run Evaluation Pipeline

The `evaluate.py` script is the core of your submission:

```bash
python evaluate.py
```

This powerful script:
- ✓ Loads all trained models from `checkpoints/`
- ✓ Evaluates each on the test set with appropriate modality
- ✓ Computes 11+ metrics per class (sensitivity, specificity, precision, F1, AUC, etc.)
- ✓ Generates 5 comparison CSV tables
- ✓ Produces 7+ visualization plots
- ✓ Creates a comprehensive summary report

---

## Evaluation Pipeline: `evaluate.py`

### Generated Outputs

**CSV Tables** (`results/`):
- `01_macro_comparison.csv` - All models' macro-averaged metrics
- `02_per_class_f1_comparison.csv` - Per-class F1-scores
- `03_per_class_sensitivity_comparison.csv` - Per-class sensitivity/recall
- `04_per_class_auc_roc_comparison.csv` - Per-class AUC-ROC scores
- `05_best_model_detailed.csv` - Best model: confusion matrix values and all metrics

**Text Reports**:
- `EVALUATION_SUMMARY.txt` - High-level summary with key metrics
- `DETAILED_METRICS_REPORT.txt` - Extended metrics for best model

**Visualizations** (`results/visualizations/`):
- `06_model_comparison_bars.png` - Bar chart of all models' performance
- `07_heatmap_f1_score.png` - Per-class heatmap for F1-scores
- `07_heatmap_sensitivity.png` - Per-class heatmap for sensitivity
- `07_heatmap_specificity.png` - Per-class heatmap for specificity
- `07_heatmap_auc_roc.png` - Per-class heatmap for AUC-ROC
- `08_confusion_matrices.png` - Best model confusion matrices (5 classes)
- `09_roc_curves.png` - ROC curves for all 5 classes
- `10_pr_curves.png` - Precision-Recall curves for all 5 classes

### Example Output

```
results/
├── 01_macro_comparison.csv
│   Model,Sensitivity,Specificity,Precision,F1-Score,Accuracy,AUC-ROC,AUC-PR
│   JointFusion,0.8234,0.9102,0.8567,0.8393,0.8945,0.9234,0.8876
│   ResNet,0.7956,0.8934,0.8234,0.8092,0.8756,0.9045,0.8678
│   ...
│
├── EVALUATION_SUMMARY.txt
│   ================================================================================
│   ECG CLASSIFICATION - MODEL EVALUATION REPORT
│   ================================================================================
│   Evaluation Date: 2026-02-10 15:30:00
│   Number of Models Evaluated: 6
│   Class Names: NORM, MI, STTC, CD, HYP
│   
│   BEST MODEL: JointFusion
│   ...
│
└── visualizations/
    ├── 06_model_comparison_bars.png
    ├── 07_heatmap_*.png
    ├── 08_confusion_matrices.png
    ├── 09_roc_curves.png
    └── 10_pr_curves.png
```

---

## Model Architectures

### Signal-Based Models

#### RNN Models (GRU/LSTM)
- **Input**: ECG signal `(batch, 1000, 3)` - 1000 time steps, 3 leads
- **Architecture**: 
  - Option 1: Bidirectional GRU with self-attention
  - Option 2: Bidirectional LSTM with self-attention
- **Convenience Functions**: `create_gru()`, `create_bigru()`, `create_lstm()`, `create_bilstm()`, `create_bigru_attention()`
- **Parameters**: 
  - Hidden size: 128
  - Num layers: 2
  - Dropout: 0.3
- **Why RNNs**: Excellent for sequential patterns; bidirectional captures both forward and backward context
- **Best for**: Subtle temporal abnormalities (ST-T changes, conduction changes)

#### CNN1D_RNN (Hybrid)
- **Input**: ECG signal `(batch, 1000, 3)`
- **Architecture**: 1D convolutions (local feature extraction) → RNN (temporal modeling) → attention → FC
- **Advantages**: Combines local pattern detection (CNN) with temporal relationships (RNN)
- **Best for**: Balanced feature learning from raw signals

### Image-Based Models

#### AlexNet
- **Input**: GAF image `(batch, 3, 224, 224)` - RGB images, 224×224
- **Architecture**: Large convolutional filters + dropout regularization
- **Convenience Functions**: `create_alexnet()` + attention variants (channel, spatial, CBAM, self-attention)
- **Parameters**: 
  - Num classes: 5
  - Dropout: 0.3
- **Why AlexNet**: Fast training, good generalization, relatively simple
- **Best for**: Quick baseline, good balance of speed and accuracy

#### VGGNet
- **Input**: GAF image `(batch, 3, 224, 224)`
- **Architecture**: Stacked 3×3 convolutions for fine-grained feature extraction
- **Characteristics**: Very deep, interpretable layer-wise features
- **Why VGGNet**: Well-established architecture, consistent performance
- **Best for**: When training time is less critical than feature quality

#### ResNet
- **Input**: GAF image `(batch, 3, 224, 224)`
- **Architecture**: Residual blocks with skip connections, 18-layer (2 blocks per stage)
- **Advantages**: 
  - Skip connections solve vanishing gradient problem
  - Can train very deep networks
  - Excellent gradient flow
- **Configuration**: `layers=[2, 2, 2, 2]` - ResNet18 equivalent
- **Best for**: High-accuracy requirements; handles very deep architectures

### Fusion Model

#### JointFusion
- **Input**: 
  - Signal: `(batch, 1000, 3)`
  - Image: `(batch, 3, 224, 224)`
- **Architecture**:
  - Signal encoder: RNN/CNN1D extracting temporal features
  - Image encoder: CNN2D extracting visual features
  - Fusion layer: Concatenates encodings
  - Classifier: FC layers on fused representation
- **Advantages**: Leverages complementary information from two modalities
- **Expected Performance**: Typically 2-5% F1-score improvement over single modality
- **Best for**: Achieving top performance; captures both temporal and visual patterns

---

## Metrics & Evaluation

### Per-Class Metrics (Computed for Each of 5 Classes)

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Sensitivity** | TP / (TP + FN) | Ability to detect positive cases (recall) |
| **Specificity** | TN / (TN + FP) | Ability to correctly reject negative cases |
| **Precision** | TP / (TP + FP) | Accuracy of positive predictions |
| **F1-Score** | 2 × (Prec × Rec) / (Prec + Rec) | Balanced measure of precision & recall |
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **AUC-ROC** | Area under ROC curve | Performance across all thresholds |
| **AUC-PR** | Area under Precision-Recall curve | Better for imbalanced data |
| **TP, FP, FN, TN** | Counts | Confusion matrix elements |
| **NPV** | TN / (TN + FN) | Negative Predictive Value |
| **Support** | Count | Number of positive samples in test set |

### Macro-Averaged Metrics
- **Definition**: Simple average of per-class metrics
- **Advantage**: Fair evaluation when classes have different support
- **Focus**: Overall model balance across all conditions

### Micro-Averaged Metrics (if needed)
- **Definition**: Weighted by class frequency
- **Advantage**: Reflects real-world performance on imbalanced data

---

## Training System

### Loss Function
- **Binary Cross-Entropy with Logits** (BCEWithLogitsLoss)
- **Why**: Multi-label classification (one patient can have multiple conditions)
- **Stability**: Combines sigmoid activation + BCE for numerical stability

### Optimizer
- **Algorithm**: Adam
- **Learning Rate**: 0.001 (configurable)
- **Weight Decay**: 1e-4 (L2 regularization)
- **Why Adam**: Adaptive learning rates, momentum, good generalization

### Learning Rate Scheduling
- **Strategy**: ReduceLROnPlateau
- **Trigger**: Validation loss plateaus
- **Reduction**: Multiplies LR by 0.5
- **Why**: Prevents overfitting; fine-tunes near local minima

### Early Stopping
- **Patience**: 5 epochs
- **Metric**: Validation loss
- **Why**: Prevents overfitting; saves computational resources
- **Saves**: Best checkpoint automatically (best validation performance)

### Data Split
- **Train**: 70% of PTB-XL (~14,700 samples)
- **Validation**: 15% (~3,150 samples) - for hyperparameter tuning
- **Test**: 15% (~3,150 samples) - final evaluation (untouched during training)

---

## Usage Examples

### Example 1: Train Single Model
```bash
# Train RNN model on signal data
python core/training/train_rnn_training.py

# Check checkpoint was saved
ls checkpoints/RNNModel_best.pth
```

### Example 2: Train Multiple Models
```bash
# Train all three model types
python core/training/train_rnn_training.py
python core/training/train_cnn2d_training.py
python core/training/train_jointfusion_training.py

# All checkpoints saved in checkpoints/
```

### Example 3: Run Full Evaluation
```bash
# Evaluate all trained models
python evaluate.py

# Results generated in results/ directory
# Browse the CSV files and PNG plots
```

### Example 4: Python API Usage
```python
from evaluate import load_model, get_test_loader, evaluate_model
from config import Config

config = Config()

# Load a specific model
model, model_type = load_model('RNNModel', 'checkpoints/RNNModel_best.pth', config)

# Get test loader
test_loader = get_test_loader(model_type, config)

# Evaluate
from core.metrics.metrics import ECGMetrics
metrics_calc = ECGMetrics(class_names=config.CLASS_NAMES)
metrics = evaluate_model(model, test_loader, model_type, metrics_calc, config.DEVICE)

# Print results
print(f"F1-Score: {metrics['macro_avg']['f1_score']:.4f}")
```

---

## Expected Performance

### Benchmarks (Based on Literature & PTB-XL Dataset)

| Model | Modality | F1-Score Range | Best Use Case |
|-------|----------|----------------|---------------|
| RNN (BiGRU) | Signal | 0.75 - 0.82 | Temporal pattern recognition |
| CNN1D_RNN | Signal | 0.76 - 0.83 | Hybrid local-temporal features |
| AlexNet | Image (GAF) | 0.78 - 0.85 | Fast baseline, good performance |
| VGGNet | Image (GAF) | 0.79 - 0.86 | Fine-grained visual features |
| ResNet | Image (GAF) | 0.80 - 0.87 | Deep learning with skip connections |
| JointFusion | Signal + Image | 0.82 - 0.89 | **BEST: Combines both modalities** |

### Performance by Class

**NORM** (Normal)
- Typically: 0.85-0.95 F1-score
- Reason: Most distinct pattern; easiest to classify
- Challenge: Low false positive rate required

**HYP** (Hypertrophy)
- Typically: 0.75-0.85 F1-score
- Reason: Clear visual signature in GAF; moderate difficulty
- Challenge: Overlap with other conditions

**MI** (Myocardial Infarction)
- Typically: 0.70-0.82 F1-score
- Reason: Distinct but variable patterns
- Challenge: Different MI types (anterior, inferior, etc.)

**STTC** (ST/T Changes)
- Typically: 0.65-0.80 F1-score
- Reason: Subtle abnormalities; most challenging
- Challenge: Overlaps with MI and HYP

**CD** (Conduction Disturbance)
- Typically: 0.68-0.82 F1-score
- Reason: Clear temporal indicators (wide QRS)
- Challenge: Varies in presentation

---


```bash
# verify, each file should be: batch_*.npy with correct shapes
# Signals: (batch_size, 1000, 3)
# Images: (batch_size, 3, 224, 224)
# Labels: (batch_size, 5) one-hot encoded
```

```bash
# Verify data shapes:
python -c "import numpy as np; print(np.load('data/signals/train/batch_0.npy').shape)"
# Should print: (batch_size, 1000, 3)

python -c "import numpy as np; print(np.load('data/images/train/gaf/batch_0.npy').shape)"
# Should print: (batch_size, 3, 224, 224)

python -c "import numpy as np; print(np.load('data/labels/y_train.npy').shape)"
# Should print: (total_samples, 5)
```

---

## File Reference Guide

### Key Files to Know

| File | Purpose | Key Functions |
|------|---------|----------------|
| **config.py** | Central configuration | Define all paths and hyperparams |
| **core/training/training.py** | Training loop | `train_model()` with epoch tracking |
| **core/metrics/metrics.py** | Metrics computation | `ECGMetrics`, `ModelComparison`, `TrainingVisualizer` |
| **evaluate.py** | **Evaluation pipeline** | 10 modular functions for comprehensive evaluation |
| **core/data/*.py** | Data loading | `create_signal_dataloader()`, `create_image_dataloader()`, `create_fusion_dataloader()` |
| **core/models/*.py** | Model definitions | Individual model classes and convenience functions |

### Metrics Classes (`core/metrics/metrics.py`)

```python
class ECGMetrics:
    """Compute comprehensive metrics per class"""
    - calculate_metrics()              # Main method
    - plot_confusion_matrices()
    - plot_roc_curves()
    - plot_precision_recall_curves()
    - generate_metrics_report()
    - calculate_confusion_matrix()

class ModelComparison:
    """Compare multiple models"""
    - compare_models()         # Bar plot comparison
    - create_comparison_table()
    - plot_metric_heatmap()

class TrainingVisualizer:
    """Visualize training progress across epochs"""
    - plot_loss_curves()
    - plot_macro_metrics()
    - plot_per_class_metric()
    - plot_training_summary()
    - compare_models_training()
```

### Evaluate.py Functions (Modular Design)

```python
# Model Loading
load_model()              # Smart model instantiation + checkpoint loading
get_test_loader()         # Appropriate dataloader per model type

# Evaluation
evaluate_model()          # Single model evaluation
evaluate_all_models()     # Main evaluation orchestrator

# Report Generation
generate_macro_table()           # Macro metrics CSV
generate_per_class_tables()      # Per-class metrics CSVs
generate_best_model_table()      # Best model details CSV

# Visualizations
generate_visualizations()        # Comparison plots & heatmaps
generate_best_model_analysis()   # ROC, PR, confusion matrices

# Summary
save_summary_report()     # Text report
main()                    # Orchestration
```

---

## References & Citations

### Dataset
- Wagner, P., et al. (2020). "PTB-XL: A Large Publicly Available Electrocardiography Dataset." *Scientific Data*, 7, 154.
- https://www.physionet.org/content/ptb-xl/

### Methods
- **RNN & Attention**: Vaswani, A., et al. (2017). "Attention is All You Need." *NIPS*.
- **ResNet**: He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
- **VGG**: Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*.
- **AlexNet**: Krizhevsky, A., et al. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *NIPS*.

### Related Work on ECG Classification with Deep Learning
- Strodthoff, N., Wagner, P., Wenzel, M., & Samek, W. (2021). "Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications."
- Multi-modal learning surveys and applications

---

## License

This is private and experimental. 
Neither allowed to embed into training, or production, nor allowed to replicate.
Copyright protected.
Author: Sujit Patel.

---

### Version 1.0 (Initial)
- Multi-modal ECG classification framework
- Multiple model architectures (3 signal-based, 3 image-based, 1 fusion)
- PTB-XL dataset integration
- Basic metrics and evaluation

---

## Support & Documentation

For detailed information, refer to:
- `config.py` - All hyperparameters and file paths
- `README.md` - This documentation
- `EVALUATE_GUIDE.md` - Detailed evaluation interpretation guide (if present)
- Model files (e.g., `core/models/rnn.py`) - Architecture documentation
- Inline code comments - Implementation details

---
**Last Updated**: February 10, 2026
