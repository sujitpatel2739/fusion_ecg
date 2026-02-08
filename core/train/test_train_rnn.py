import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

# Importing DataLoaders
from core.data.signal_dataloader import create_signals_dataloaders
# Importing models
from core.models.model import create_gru, create_bigru, create_bigru_attention, create_bilstm
from core.train.train_rnn import train_model

# Configuration
class Config:
    # Data paths
    TRAIN_SIGNAL_PATH = 'data/signals/train/'
    TRAIN_LABELS_PATH = 'data/labels/y_train.npy'
    VAL_SIGNAL_PATH = 'data/signals/validation/'
    VAL_LABELS_PATH = 'data/labels/y_val.npy'
    TEST_SIGNAL_PATH = 'data/signals/test/'
    TEST_LABELS_PATH = 'data/labels/y_test.npy'
    
    # Model parameters
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 5  # Small number for testing
    LEARNING_RATE = 0.001
    
    # Other
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = 'checkpoints/test_run'
    NUM_WORKERS = 4
    
    # Classes
    CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    NUM_CLASSES = 5


def plot_training_history(histories, model_names):
    """
    Plot training curves for all models
    
    Args:
        histories: List of history dictionaries
        model_names: List of model names
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Helper to plot if metric exists
    def _plot_metric(ax, key_names, title, ylabel=None):
        plotted = False
        for history, name in zip(histories, model_names):
            for key in key_names:
                if key in history:
                    ax.plot(history[key], label=name)
                    plotted = True
                    break
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True)
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Metric not available', ha='center', va='center', fontsize=10, color='gray')

    # Plotting metrics with fallbacks for common key names
    _plot_metric(axes[0], ['train_loss', 'loss'], 'Training Loss', ylabel='Loss')
    _plot_metric(axes[1], ['val_loss', 'validation_loss'], 'Validation Loss', ylabel='Loss')
    _plot_metric(axes[2], ['sensitivity', 'recall', 'sens'], 'Sensitivity / Recall', ylabel='Sensitivity')
    _plot_metric(axes[3], ['specificity', 'precision', 'spec'], 'Specificity / Precision', ylabel='Score')

    plt.tight_layout()
    out_path = 'training_curves.png'
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved training curves to {out_path}")
    

def main():
    config = Config()
    
    # Create save directory
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    print("="*70)
    print("RNN MODEL TRAINING TEST")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    
    # Use the configured batch size for both saved batches and DataLoader
    SAVED_BATCH_SIZE = config.BATCH_SIZE
    DATALOADER_BATCH_SIZE = config.BATCH_SIZE
    NUM_WORKERS = 4
    
    config = Config()

    # Load data
    print("\nLoading data...")
    print("Creating signals dataloaders...")
    # Create dataloaders
    train_dataloader = create_signals_dataloaders(
        image_path=config.TRAIN_SIGNAL_PATH,
        labels_path=config.TRAIN_LABELS_PATH,
        saved_batch_size=SAVED_BATCH_SIZE,
        batch_size=DATALOADER_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    data_iter = iter(train_dataloader)
    batch0 = next(data_iter)
    # batch0 is a tuple (signals, labels) so use the first element's first dim
    batch_size_report = batch0[0].size(0) if hasattr(batch0[0], 'size') else len(batch0[0])
    print(f"✓ Train batches: {len(train_dataloader)}")
    print(f"✓ Batch size: {batch_size_report}")

    val_dataloader = create_signals_dataloaders(
        image_path=config.VAL_SIGNAL_PATH,
        labels_path=config.VAL_LABELS_PATH,
        saved_batch_size=SAVED_BATCH_SIZE,
        batch_size=DATALOADER_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    data_iter = iter(val_dataloader)
    batch0 = next(data_iter)
    batch_size_report = batch0[0].size(0) if hasattr(batch0[0], 'size') else len(batch0[0])
    print(f"✓ Val batches: {len(val_dataloader)}")
    print(f"✓ Batch size: {batch_size_report}")
    del data_iter
    del batch0
    
    # Create models to test
    models_to_train = {
        'BiGRU': create_bigru(
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ),
        'BiLSTM': create_bilstm(
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ),
        'BiGRU_Attention': create_bigru_attention(
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            attention_type='additive'
        )
    }

    # Train each model
    histories = []
    model_names = []
    
    # Loss function (Binary Cross Entropy for multi-label)
    criterion = nn.BCEWithLogitsLoss()
    for name, model in models_to_train.items():
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        history = train_model(
            model,
            train_dataloader,
            val_dataloader,
            criterion,
            optimizer,
            scheduler,
            config
        )
        histories.append(history)
        model_names.append(name)
    
    # Plot results
    plot_training_history(histories, model_names)
    
    print("\n" + "="*70)
    print("✓ ALL TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()