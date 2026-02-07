import numpy as np
import torch
from tqdm import tqdm

def validate(model, dataloader, criterion, device, Config):
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
            # predictions (threshold at 0.5)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics (accuracy, sensitivity, specificity)
    all_predictions = np.vstack(all_predictions)  # (N, 5)
    all_labels = np.vstack(all_labels)  # (N, 5)

    for i, class_name in enumerate(Config.CLASS_NAME):
        y_true = all_labels[:, i]
        y_pred = all_predictions[:, i]

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
            
    metrics = {}
    