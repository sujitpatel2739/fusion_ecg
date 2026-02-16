import torch
import os

# Configurations, Adjust Properly
class Config:
    # Data paths
    TRAIN_SIGNAL_PATH = 'data/signals/train/'
    TRAIN_IMAGE_PATH = 'data/images/train/'
    TRAIN_LABEL_PATH = 'data/labels/y_train.npy'
    VAL_SIGNAL_PATH = 'data/signals/validation/'
    VAL_IMAGE_PATH = 'data/images/validation/'
    VAL_LABEL_PATH = 'data/labels/y_val.npy'
    TEST_SIGNAL_PATH = 'data/signals/test/'
    TEST_IMAGE_PATH = 'data/images/test/'
    TEST_LABEL_PATH = 'data/labels/y_test.npy'

    METRICS_SAVE_PATH = 'saved_metrics'
    if not os.path.exists(METRICS_SAVE_PATH):
        os.makedirs(METRICS_SAVE_PATH)
    TRAIN_HISTORY_SAVE_PATH = 'training_history'
    if not os.path.exists(TRAIN_HISTORY_SAVE_PATH):
        os.makedirs(TRAIN_HISTORY_SAVE_PATH)
    EVAL_METRICS_SAVE_PATH = 'evaluation_metrics'
    if not os.path.exists(EVAL_METRICS_SAVE_PATH):
        os.makedirs(EVAL_METRICS_SAVE_PATH)

    # Data loading parameters
    SAVED_BATCH_SIZE = 32

    # Model parameters
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3

    # Optimization
    WEIGHT_DECAY = 1e-4  # L2 regularization
    PATIENCE = 5  # Early stopping patience

    # Training parameters
    BATCH_SIZE = 64 # Should be in multiple of SAVE_BATCH_SIZE
    NUM_EPOCHS = 10  # Small number for testing
    LEARNING_RATE = 0.001
    SAVE_BEST = True

    # Other
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = '/content/checkpoints/'
    NUM_WORKERS = 4

    # Classes
    CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    NUM_CLASSES = 5