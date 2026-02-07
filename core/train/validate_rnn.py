import numpy as np
import torch
from tqdm import tqdm

def validate_one_epoch(model, dataloader, criterion, device, Config):
    """
    Validate model
    
    Returns:
        avg_loss: Average validation loss
        metrics: Dictionary with accuracy, sensitivity, specificity
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in tqdm(dataloader, desc='Validation'):
            signals, labels = signals.to(device), labels.to(device)
            output = model(signals)
            if isinstance(output, tuple):
                logits, att_weights = output  # Model with attention
            else:
                logits = output
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            # predictions (threshold at 0.5)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    