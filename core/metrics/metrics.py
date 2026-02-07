import numpy as np
from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate comprehensive metrics for each class
    
    Args:
        y_true: True binary labels (N, num_classes)
        y_pred: Predicted binary labels (N, num_classes)
        y_prob: Predicted probabilities (N, num_classes)
    """
    
    metrics = {}
    
    for i, class_name in enumerate(['NORM', 'MI', 'STTC', 'CD', 'HYP']):
        # Extract this class
        yt = y_true[:, i]
        yp = y_pred[:, i]
        yprob = y_prob[:, i]
        
        # Calculate confusion matrix elements
        tp = ((yt == 1) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        
        # Basic metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Advanced metrics
        f1 = f1_score(yt, yp, zero_division=0)
        
        try:
            auc_roc = roc_auc_score(yt, yprob)
            auc_pr = average_precision_score(yt, yprob)
        except:
            auc_roc = 0.0
            auc_pr = 0.0
        
        metrics[class_name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'support': yt.sum()  # Number of positive samples
        }
    
    # Macro averages
    metrics['macro_avg'] = {
        'sensitivity': np.mean([m['sensitivity'] for m in metrics.values() if isinstance(m, dict)]),
        'specificity': np.mean([m['specificity'] for m in metrics.values() if isinstance(m, dict)]),
        'precision': np.mean([m['precision'] for m in metrics.values() if isinstance(m, dict)]),
        'f1_score': np.mean([m['f1_score'] for m in metrics.values() if isinstance(m, dict)]),
        'auc_roc': np.mean([m['auc_roc'] for m in metrics.values() if isinstance(m, dict)]),
        'auc_pr': np.mean([m['auc_pr'] for m in metrics.values() if isinstance(m, dict)])
    }
    
    return metrics