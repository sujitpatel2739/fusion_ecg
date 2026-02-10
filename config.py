import torch

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
    
    # Model parameters
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Optimization
    WEIGHT_DECAY = 1e-4  # L2 regularization
    PATIENCE = 5  # Early stopping patience
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 5  # Small number for testing
    LEARNING_RATE = 0.001
    SAVE_BEST = True
    
    # Other
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = 'checkpoints/'
    NUM_WORKERS = 4
    
    # Classes
    CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    NUM_CLASSES = 5
    