"""
Training script for RNNs, CNN2D, CNN1D, and Join Fusion model on ECG Signals + Images
Tests if models are learning properly
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Args:
        model: model to train/validate
        dataloader: Training/Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: 'cuda' or 'cpu'
    
    Returns:
        avg_loss: Average training loss
    """
    model.train()
    total_loss = 0
    count = 0
    for x, labels, in tqdm(dataloader, desc="Training"):
        if isinstance(x, (tuple,list)):
            x = (x[0].to(device), x[1].to(device)) # (images, images/signals): (batch, 3, 224, 224), (batch, 1000, 3)
        else:
            x = x.to(device) # (images/signals)
        labels = labels.to(device) # (batch, 5)
         
        optimizer.zero_grad()
        
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count+= 1
        if count >= 15:
            break
        
    return total_loss / count

def validate_one_epoch(model, val_loader, criterion, metrics, device):
    """
    Validation phase - calculate LOSS + ALL METRICS
    """
    model.eval()
    total_loss = 0
    
    # Store ALL predictions and labels for this epoch
    all_predictions = []
    all_labels = []
    all_probs = []
    count = 0
    
    with torch.no_grad():
        for x, labels in tqdm(val_loader, desc='Validation'):
            if isinstance(x, (tuple,list)):
                x = (x[0].to(device), x[1].to(device))
            else:
                x = x.to(device)
            labels = labels.to(device)
            
            logits = model(x)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Store for metric calculation
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            count += 1
            if count >= 15:
                break
            
    # Calculate average loss
    avg_loss = total_loss / count
    
    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)  # (N, 5)
    all_labels = np.vstack(all_labels)            # (N, 5)
    all_probs = np.vstack(all_probs)              # (N, 5)


    # Calculate ALL metrics using predictions from entire epoch
    val_metrics = metrics.calculate_metrics(
        all_labels, 
        all_predictions, 
        all_probs
    )
    
    return avg_loss, val_metrics


def train_model(model,
                model_name,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                schedular,
                metrics,
                config
                ):
    """
    Complete training loop
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f1': [],
        'accuracy': [],
        'auc_roc': [],
        'auc_pr': [],
    }
    class_wise_metrics = {cls: {'sensitivity': [], 'specificity': [], 'precision': [], 'f1': [], 'accuracy': [], 'auc_roc': [], 'auc_pr': []} for cls in config.CLASS_NAMES}
    best_val_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Training - get only loss
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # Validation - get loss + all metrics
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, metrics, config.DEVICE)
        
        schedular.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['sensitivity'].append(val_metrics['macro_avg']['sensitivity'])
        history['specificity'].append(val_metrics['macro_avg']['specificity'])
        history['precision'].append(val_metrics['macro_avg']['precision'])
        history['f1'].append(val_metrics['macro_avg']['f1_score'])
        history['accuracy'].append(val_metrics['macro_avg']['accuracy'])
        history['auc_roc'].append(val_metrics['macro_avg']['auc_roc'])
        history['auc_pr'].append(val_metrics['macro_avg']['auc_pr'])
        
        for cls in config.CLASS_NAMES:
            class_wise_metrics[cls]['sensitivity'].append(val_metrics[cls]['sensitivity'])
            class_wise_metrics[cls]['specificity'].append(val_metrics[cls]['specificity'])
            class_wise_metrics[cls]['precision'].append(val_metrics[cls]['precision'])
            class_wise_metrics[cls]['f1'].append(val_metrics[cls]['f1_score'])
            class_wise_metrics[cls]['accuracy'].append(val_metrics[cls]['accuracy'])
            class_wise_metrics[cls]['auc_roc'].append(val_metrics[cls]['auc_roc'])
            class_wise_metrics[cls]['auc_pr'].append(val_metrics[cls]['auc_pr'])

        # Print summary
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics (Macro):")
        print(f"  Sensitivity: {val_metrics['macro_avg']['sensitivity']:.4f}")
        print(f"  Specificity: {val_metrics['macro_avg']['specificity']:.4f}")
        print(f"  Precision: {val_metrics['macro_avg']['precision']:.4f}")
        print(f"  F1-Score: {val_metrics['macro_avg']['f1_score']:.4f}")
        print(f"  Accuracy: {val_metrics['macro_avg']['accuracy']:.4f}")
        print(f"  AUC-ROC: {val_metrics['macro_avg']['auc_roc']:.4f}")
        print(f"  AUC-PR: {val_metrics['macro_avg']['auc_pr']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            if config.SAVE_BEST == True:
                save_path = os.path.join(config.SAVE_DIR, f'{model_name}_best.pth')
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'history': history
                }, save_path)
            
                print(f"✓ Saved best model (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best epoch: {best_epoch+1} with Val Loss: {best_val_loss:.4f}")
            break
    
        print(f"\n{'='*45}")
        print(f"Training complete!")
        print(f"Best epoch: {best_epoch+1}")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"{'='*45}")
    
    return history, class_wise_metrics