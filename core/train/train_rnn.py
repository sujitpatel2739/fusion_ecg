"""
Training script for RNN models on ECG signals
Tests if models are learning properly
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

from core.metrics.metrics import calculate_metrics 

    
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Training phase - only calculate LOSS
    """
    model.train()
    total_loss = 0
    # count = 0
    
    for signals, labels in tqdm(train_loader, desc='Training'):
        signals = signals.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(signals)
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # count += 1
        # if count == 5:
        #     break
        
    return total_loss / len(train_loader)


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Validation phase - calculate LOSS + ALL METRICS
    """
    model.eval()
    total_loss = 0
    
    # Store ALL predictions and labels for this epoch
    all_predictions = []
    all_labels = []
    all_probs = []
    # count = 0
    
    with torch.no_grad():
        for signals, labels in tqdm(val_loader, desc='Validation'):
            signals = signals.to(device)
            labels = labels.to(device)
            
            output = model(signals)
            if isinstance(output, tuple):
                logits, _ = output
            else:
                logits = output
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Store for metric calculation
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            # count += 1
            # if count == 5:
            #     break
            
    # Calculate average loss
    avg_loss = total_loss / len(val_loader)
    
    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)  # (N, 4)
    all_labels = np.vstack(all_labels)            # (N, 4)
    all_probs = np.vstack(all_probs)              # (N, 4)


    # Calculate ALL metrics using predictions from entire epoch
    metrics = calculate_metrics(
        all_labels, 
        all_predictions, 
        all_probs
    )
    
    return avg_loss, metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, schedular, config):
    """
    Complete training loop
    """
    history = {
        'train_loss': [],     
        'val_loss': [],       
        'val_sensitivity': [],
        'val_specificity': [],
        'val_precision': [],  
        'val_f1': [],         
        'val_auc': []         
    }
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Training - get only loss
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # Validation - get loss + all metrics
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, config.DEVICE)
        
        schedular.step(val_loss)
        
        # Store ONE VALUE per metric for this epoch
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_sensitivity'].append(val_metrics['macro_avg']['sensitivity'])
        history['val_specificity'].append(val_metrics['macro_avg']['specificity'])
        history['val_precision'].append(val_metrics['macro_avg']['precision'])
        history['val_f1'].append(val_metrics['macro_avg']['f1_score'])
        history['val_auc'].append(val_metrics['macro_avg']['auc_roc'])
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Sensitivity: {val_metrics['macro_avg']['sensitivity']:.4f}")
        print(f"Val Specificity: {val_metrics['macro_avg']['specificity']:.4f}")
        print(f"Val F1: {val_metrics['macro_avg']['f1_score']:.4f}")
    
    return history
