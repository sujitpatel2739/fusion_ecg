import torch
import torch.nn as nn
import torch.optim as optim
import os

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from core.training.config import Config
# Importing DataLoader
from core.data.fusion_dataloader import create_fusion_dataloader
# Importing models
from core.models.joint_fusion import JointFusion
from core.training.training import train_model
from core.training.history import save_histories, plot_training_history
    
def main():
    config = Config()
    
    # Create save directory
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    print("="*45)
    print("RNN MODEL TRAINING TEST")
    print("="*45)
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
    train_dataloader = create_fusion_dataloader(
        signal_path=Config.TRAIN_SIGNAL_PATH,
        image_path=config.TRAIN_IMAGE_PATH,
        labels_path=config.TRAIN_LABEL_PATH,
        saved_batch_size=SAVED_BATCH_SIZE,
        batch_size=DATALOADER_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    data_iter = iter(train_dataloader)
    batch0 = next(data_iter)
    # batch0 is a tuple (signals, labels) so use the first element's first dim
    print(f"✓ Train batches: {len(train_dataloader)}")
    print(f"✓ Batch size: {len(batch0[0][0])}")

    val_dataloader = create_fusion_dataloader(
        signal_path=Config.VAL_SIGNAL_PATH,
        image_path=config.VAL_IMAGE_PATH,
        labels_path=config.VAL_LABEL_PATH,
        saved_batch_size=SAVED_BATCH_SIZE,
        batch_size=DATALOADER_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    data_iter = iter(val_dataloader)
    batch0 = next(data_iter)
    print(f"✓ Val batches: {len(val_dataloader)}")
    print(f"✓ Batch size: {len(batch0[0][0])}")
    del data_iter
    del batch0
    
    # Create models to test
    models_to_train = {
        'JointFusion': JointFusion(3, 128, None, None, 5)
    }

    # Train each model
    histories = []
    model_names = []
    
    # Loss function (Binary Cross Entropy for multi-label)
    criterion = nn.BCEWithLogitsLoss()
    for name, model in models_to_train.items():
        print(f"Training Model: {name}")
        # Optimizer
        optimizer = optim.Adam(model.parameters(),
                               lr=config.LEARNING_RATE,
                               weight_decay=config.WEIGHT_DECAY)
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3,
            min_lr=1e-6
        )
        
        history = train_model(
            model,
            train_dataloader,
            val_dataloader,
            criterion,
            optimizer,
            scheduler,
            config,
            save_best=True
        )
        histories.append(history)
        model_names.append(name)
    
    # Save histories to disk for future analysis
    print("\nSaving training histories...")
    save_histories(histories, model_names)
    
    # Plot results
    plot_training_history(histories, model_names)
    
    print("\n" + "="*45)
    print("✓ ALL TRAINING COMPLETE")
    print("="*45)


if __name__ == "__main__":
    main()