import torch
import torch.nn as nn
import torch.optim as optim
import os

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from config import Config
from core.metrics.metrics import ECGMetrics, TrainingVisualizer
# Importing DataLoaders
from core.data.image_dataloader import create_image_dataloader
# Importing models
from core.models.cnn2d_alexnet import create_alexnet, create_alexnet_spatial_attention, create_alexnet_cbam, create_alexnet_channel_attention
from core.models.cnn2d_resnet  import ResNet18
from core.training.training import train_model
from core.training.history import save_histories, plot_training_history

def main():
    config = Config()
    
    # Create save directory
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    print("="*40)
    print("RNN MODEL TRAINING TEST")
    print("="*40)
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
    train_dataloader = create_image_dataloader(
        image_path=config.TRAIN_IMAGES_PATH,
        labels_path=config.TRAIN_LABELS_PATH,
        saved_batch_size=SAVED_BATCH_SIZE,
        transform_type='gaf',
        batch_size=DATALOADER_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    data_iter = iter(train_dataloader)
    batch0 = next(data_iter)
    # batch0 is a tuple (images, labels) so use the first element's first dim
    batch_size_report = batch0[0].size(0) if hasattr(batch0[0], 'size') else len(batch0[0])
    print(f"✓ Train batches: {len(train_dataloader)}")
    print(f"✓ Batch size: {batch_size_report}")

    val_dataloader = create_image_dataloader(
        image_path=config.VAL_IMAGES_PATH,
        labels_path=config.VAL_LABELS_PATH,
        saved_batch_size=SAVED_BATCH_SIZE,
        transform_type='gaf',
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
        'AlexNet': create_alexnet(),
        'AlexNetSpatialAttention': create_alexnet_spatial_attention(),
        'AlexNetChannelAttention': create_alexnet_channel_attention(),
        'AlexNetCBAMAttention': create_alexnet_cbam(),
        'ResNet': ResNet18(3, 5, base_filters=16)
    }

    # Train each model
    histories = []
    model_names = []
    
    # Loss function (Binary Cross Entropy for multi-label)
    criterion = nn.BCEWithLogitsLoss()
    metrics = ECGMetrics(config.CLASS_NAMES)
    viz = TrainingVisualizer(config.CLASS_NAMES)
    
    for name, model in models_to_train.items():
        print(f"Training Model: {name}")
        # Optimizer
        model = model.to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(),
                               lr=config.LEARNING_RATE,
                               weight_decay=config.WEIGHT_DECAY)
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3,
            min_lr=1e-6
        )
    
        history, class_wise_metrics = train_model(
            model,
            name,
            train_dataloader,
            val_dataloader,
            criterion,
            optimizer,
            scheduler,
            metrics,
            config,
        )
        histories.append(history)
        model_names.append(name)
    
        # Plot history
        viz.plot_loss_curves(history, f'{config.TRAIN_HISTORY_SAVE_PATH}/{name}_loss_currves.png', name)
        viz.plot_macro_metrics(history, f'{config.TRAIN_HISTORY_SAVE_PATH}/{name}_macro_metrics.png', name)
        viz.plot_training_summary(history, f'{config.TRAIN_HISTORY_SAVE_PATH}/{name}_training_history.png', name)
    
        # Per-class metrics (requires class_wise_metrics)
        viz.plot_per_class_metric(class_wise_metrics, save_path = f'{config.TRAIN_HISTORY_SAVE_PATH}/{name}_per_class_metric.png', name=name)
        viz.plot_per_class_metric(class_wise_metrics, )  # All classes
    
    
    # Save histories to disk for future analysis
    print("\nSaving training histories...")
    save_histories(histories, model_names)
    
    # Plot results
    plot_training_history(histories, model_names)
    
    print("\n" + "="*40)
    print("✓ ALL TRAINING COMPLETE")
    print("="*40)


if __name__ == "__main__":
    main()