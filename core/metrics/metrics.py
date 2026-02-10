"""
Metrics Module for ECG Classification.
Includes confusion matrix, ROC curves, and comprehensive model comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score, 
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    accuracy_score
)
import pandas as pd
from typing import Dict, List, Tuple
import os


class ECGMetrics:
    """Comprehensive metrics calculator for ECG classification"""
    
    def __init__(self, class_names=['NORM', 'MI', 'STTC', 'CD', 'HYP']):
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_confusion_matrix(self, y_true, y_pred):
        """
        Calculate confusion matrix for each class (one-vs-rest)
        
        Args:
            y_true: (N, num_classes) binary labels
            y_pred: (N, num_classes) binary predictions
            
        Returns:
            dict: Confusion matrices for each class
        """
        cm_dict = {}
        
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            cm_dict[class_name] = cm
        
        return cm_dict
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """
        Calculate comprehensive metrics for multi-label classification
        
        Args:
            y_true: (N, num_classes) true binary labels
            y_pred: (N, num_classes) predicted binary labels  
            y_prob: (N, num_classes) predicted probabilities
            
        Returns:
            dict: All metrics organized by class and averages
        """        
        metrics = {}
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            yprob = y_prob[:, i]
            
            # Confusion matrix elements
            cm = confusion_matrix(yt, yp)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle edge case where only one class is present
                tp = fp = tn = fn = 0
                if cm.shape == (1, 1):
                    if yt[0] == 1:
                        tp = cm[0, 0]
                    else:
                        tn = cm[0, 0]
            
            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
            
            f1 = f1_score(yt, yp, zero_division=0)
            accuracy = accuracy_score(yt, yp)
            
            # AUC metrics
            try:
                auc_roc = roc_auc_score(yt, yprob)
                auc_pr = average_precision_score(yt, yprob)
            except:
                auc_roc = 0.0
                auc_pr = 0.0
            
            metrics[class_name] = {
                'confusion_matrix': cm,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'npv': float(npv),
                'f1_score': float(f1),
                'accuracy': float(accuracy),
                'auc_roc': float(auc_roc),
                'auc_pr': float(auc_pr),
                'support': int(yt.sum())
            }
        
        # Calculate macro averages
        metrics['macro_avg'] = {
            'sensitivity': np.mean([m['sensitivity'] for m in metrics.values() if isinstance(m, dict)]),
            'specificity': np.mean([m['specificity'] for m in metrics.values() if isinstance(m, dict)]),
            'precision': np.mean([m['precision'] for m in metrics.values() if isinstance(m, dict)]),
            'f1_score': np.mean([m['f1_score'] for m in metrics.values() if isinstance(m, dict)]),
            'accuracy': np.mean([m['accuracy'] for m in metrics.values() if isinstance(m, dict)]),
            'auc_roc': np.mean([m['auc_roc'] for m in metrics.values() if isinstance(m, dict)]),
            'auc_pr': np.mean([m['auc_pr'] for m in metrics.values() if isinstance(m, dict)])
        }
        
        # Calculate micro averages (treating all classes equally)
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        y_prob_flat = y_prob.ravel()
        
        return metrics
    
    
    def plot_confusion_matrices(self, y_true, y_pred, save_path='confusion_matrices.png', 
                                model_name='Model'):
        """
        Plot confusion matrices for all classes in a grid
        
        Args:
            y_true: (N, num_classes) true labels
            y_pred: (N, num_classes) predictions
            save_path: Path to save figure
            model_name: Name of the model for title
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[i], cbar=True)
            
            axes[i].set_title(f'{class_name}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
        
        # Hide the last subplot if we have 5 classes
        axes[5].axis('off')
        
        plt.suptitle(f'Confusion Matrices - {model_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrices to {save_path}")
    
    def plot_roc_curves(self, y_true, y_prob, save_path='roc_curves.png',
                       model_name='Model'):
        """
        Plot ROC curves for all classes
        
        Args:
            y_true: (N, num_classes) true labels
            y_prob: (N, num_classes) predicted probabilities
            save_path: Path to save figure
            model_name: Name of the model
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            try:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                
                axes[i].plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
                axes[i].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(f'{class_name}', fontweight='bold')
                axes[i].legend(loc='lower right')
                axes[i].grid(True, alpha=0.3)
            except:
                axes[i].text(0.5, 0.5, 'Insufficient data', 
                           ha='center', va='center')
                axes[i].set_title(f'{class_name}', fontweight='bold')
        
        axes[5].axis('off')
        
        plt.suptitle(f'ROC Curves - {model_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ROC curves to {save_path}")
    
    def plot_precision_recall_curves(self, y_true, y_prob, 
                                     save_path='pr_curves.png',
                                     model_name='Model'):
        """
        Plot Precision-Recall curves for all classes
        
        Args:
            y_true: (N, num_classes) true labels
            y_prob: (N, num_classes) predicted probabilities
            save_path: Path to save figure
            model_name: Name of the model
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            try:
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
                auc_pr = average_precision_score(y_true[:, i], y_prob[:, i])
                
                axes[i].plot(recall, precision, 
                           label=f'AP = {auc_pr:.3f}', linewidth=2)
                axes[i].set_xlabel('Recall')
                axes[i].set_ylabel('Precision')
                axes[i].set_title(f'{class_name}', fontweight='bold')
                axes[i].legend(loc='lower left')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlim([0, 1])
                axes[i].set_ylim([0, 1])
            except:
                axes[i].text(0.5, 0.5, 'Insufficient data',
                           ha='center', va='center')
                axes[i].set_title(f'{class_name}', fontweight='bold')
        
        axes[5].axis('off')
        
        plt.suptitle(f'Precision-Recall Curves - {model_name}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved PR curves to {save_path}")
    
    def generate_metrics_report(self, metrics, save_path='metrics_report.txt'):
        """
        Generate a detailed text report of all metrics, for a single epoch.
        
        Args:
            metrics: Dictionary from calculate_all_metrics()
            save_path: Path to save report
        """
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ECG CLASSIFICATION METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Per-class metrics
            f.write("PER-CLASS METRICS:\n")
            f.write("-" * 80 + "\n")
            
            for class_name in self.class_names:
                m = metrics[class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  Confusion Matrix:\n")
                f.write(f"    TN: {m['tn']:6d}  FP: {m['fp']:6d}\n")
                f.write(f"    FN: {m['fn']:6d}  TP: {m['tp']:6d}\n")
                f.write(f"  Metrics:\n")
                f.write(f"    Sensitivity (Recall): {m['sensitivity']:.4f}\n")
                f.write(f"    Specificity:          {m['specificity']:.4f}\n")
                f.write(f"    Precision:            {m['precision']:.4f}\n")
                f.write(f"    NPV:                  {m['npv']:.4f}\n")
                f.write(f"    F1-Score:             {m['f1_score']:.4f}\n")
                f.write(f"    Accuracy:             {m['accuracy']:.4f}\n")
                f.write(f"    AUC-ROC:              {m['auc_roc']:.4f}\n")
                f.write(f"    AUC-PR:               {m['auc_pr']:.4f}\n")
                f.write(f"    Support:              {m['support']}\n")
            
            # Macro averages
            f.write("\n" + "=" * 80 + "\n")
            f.write("MACRO AVERAGES (Equal weight per class):\n")
            f.write("-" * 80 + "\n")
            m = metrics['macro_avg']
            f.write(f"  Sensitivity: {m['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {m['specificity']:.4f}\n")
            f.write(f"  Precision:   {m['precision']:.4f}\n")
            f.write(f"  F1-Score:    {m['f1_score']:.4f}\n")
            f.write(f"  Accuracy:    {m['accuracy']:.4f}\n")
            f.write(f"  AUC-ROC:     {m['auc_roc']:.4f}\n")
            f.write(f"  AUC-PR:      {m['auc_pr']:.4f}\n")
            
            # Micro averages
            f.write("\n" + "=" * 80 + "\n")
            f.write("MICRO AVERAGES (Equal weight per sample):\n")
            f.write("-" * 80 + "\n")
            m = metrics['micro_avg']
            f.write(f"  Sensitivity: {m['sensitivity']:.4f}\n")
            f.write(f"  Precision:   {m['precision']:.4f}\n")
            f.write(f"  F1-Score:    {m['f1_score']:.4f}\n")
            f.write(f"  AUC-ROC:     {m['auc_roc']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"Saved metrics report to {save_path}")


class ModelComparison:
    """Compare multiple models' performance"""
    
    def __init__(self, class_names=['NORM', 'MI', 'STTC', 'CD', 'HYP']):
        self.class_names = class_names
    
    def compare_models(self, models_metrics: Dict[str, Dict], 
                      save_path='model_comparison.png'):
        """
        Create comparison visualizations for multiple models
        
        Args:
            models_metrics: Dict mapping model_name -> metrics dict
            save_path: Path to save comparison plot
        """
        # Extract data for plotting
        model_names = list(models_metrics.keys())
        metrics_to_plot = ['sensitivity', 'specificity', 'precision', 
                          'f1_score', 'auc_roc']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot per-class comparisons
        for i, class_name in enumerate(self.class_names):
            ax = axes[i]
            
            # Prepare data
            data = {metric: [] for metric in metrics_to_plot}
            
            for model_name in model_names:
                metrics = models_metrics[model_name][class_name]
                for metric in metrics_to_plot:
                    data[metric].append(metrics[metric])
            
            # Create grouped bar plot
            x = np.arange(len(model_names))
            width = 0.15
            
            for j, metric in enumerate(metrics_to_plot):
                offset = (j - 2) * width
                ax.bar(x + offset, data[metric], width, 
                      label=metric.replace('_', ' ').title())
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title(f'{class_name}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1])
        
        # Plot macro averages comparison
        ax = axes[5]
        data = {metric: [] for metric in metrics_to_plot}
        
        for model_name in model_names:
            metrics = models_metrics[model_name]['macro_avg']
            for metric in metrics_to_plot:
                data[metric].append(metrics[metric])
        
        x = np.arange(len(model_names))
        width = 0.15
        
        for j, metric in enumerate(metrics_to_plot):
            offset = (j - 2) * width
            ax.bar(x + offset, data[metric], width,
                  label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Macro Average', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        plt.suptitle('Model Comparison Across All Classes',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved model comparison to {save_path}")
    
    def create_comparison_table(self, models_metrics: Dict[str, Dict],
                               save_path='comparison_table.csv'):
        """
        Create a CSV table comparing all models
        
        Args:
            models_metrics: Dict mapping model_name -> metrics dict
            save_path: Path to save CSV
        """
        rows = []
        
        for model_name, metrics in models_metrics.items():
            # Macro averages row
            row = {
                'Model': model_name,
                'Class': 'MACRO AVG',
                'Sensitivity': metrics['macro_avg']['sensitivity'],
                'Specificity': metrics['macro_avg']['specificity'],
                'Precision': metrics['macro_avg']['precision'],
                'F1-Score': metrics['macro_avg']['f1_score'],
                'AUC-ROC': metrics['macro_avg']['auc_roc'],
                'AUC-PR': metrics['macro_avg']['auc_pr']
            }
            rows.append(row)
            
            # Per-class rows
            for class_name in self.class_names:
                m = metrics[class_name]
                row = {
                    'Model': model_name,
                    'Class': class_name,
                    'Sensitivity': m['sensitivity'],
                    'Specificity': m['specificity'],
                    'Precision': m['precision'],
                    'F1-Score': m['f1_score'],
                    'AUC-ROC': m['auc_roc'],
                    'AUC-PR': m['auc_pr']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False, float_format='%.4f')
        
        print(f"Saved comparison table to {save_path}")
        
        return df
    
    def plot_metric_heatmap(self, models_metrics: Dict[str, Dict],
                           metric='f1_score',
                           save_path='metric_heatmap.png'):
        """
        Create a heatmap showing a specific metric across models and classes
        
        Args:
            models_metrics: Dict mapping model_name -> metrics dict
            metric: Which metric to visualize
            save_path: Path to save figure
        """
        model_names = list(models_metrics.keys())
        
        # Build matrix: rows = models, cols = classes + macro avg
        cols = self.class_names + ['Macro Avg']
        data = []
        
        for model_name in model_names:
            row = []
            for class_name in self.class_names:
                row.append(models_metrics[model_name][class_name][metric])
            row.append(models_metrics[model_name]['macro_avg'][metric])
            data.append(row)
        
        data = np.array(data)
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu',
                   xticklabels=cols, yticklabels=model_names,
                   cbar_kws={'label': metric.replace('_', ' ').title()},
                   vmin=0, vmax=1)
        
        plt.title(f'{metric.replace("_", " ").title()} Comparison Across Models',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved metric heatmap to {save_path}")


class TrainingVisualizer:
    """Visualize training progress and convergence across epochs.
    
    Works with flattened history structure:
    history = {
        'train_loss': [...],
        'val_loss': [...],
        'sensitivity': [...],
        'specificity': [...],
        'precision': [...],
        'f1': [...],
        'accuracy': [...],
        'auc_roc': [...],
        'auc_pr': [...]
    }
    
    class_wise_metrics = {
        'NORM': {'sensitivity': [...], 'f1': [...], 'auc_roc': [...], ...},
        'MI': {...},
        ...
    }
    """
    
    def __init__(self, class_names=['NORM', 'MI', 'STTC', 'CD', 'HYP']):
        self.class_names = class_names
    
    def plot_loss_curves(self, history, save_path='loss_curves.png', model_name='Model'):
        """
        Plot training and validation loss over epochs
        
        Args:
            history: Dict with 'train_loss' and 'val_loss' lists
            save_path: Path to save figure
            model_name: Name of the model for title
        """
        if 'train_loss' not in history or 'val_loss' not in history:
            print("Warning: 'train_loss' or 'val_loss' not found in history.")
            return
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=6)
        plt.plot(epochs, history['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=6)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Training and Validation Loss - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved loss curves to {save_path}")
    
    def plot_macro_metrics(self, history, save_path='macro_metrics.png', model_name='Model'):
        """
        Plot macro-averaged metrics over epochs
        
        Args:
            history: Dict with metric keys containing per-epoch lists
                    {'sensitivity': [...], 'specificity': [...], 'f1': [...], 'auc_roc': [...], ...}
            save_path: Path to save figure
            model_name: Name of the model
        """
        # Map history keys to display names (note: 'f1' in history, 'f1_score' for display)
        metric_mapping = {
            'sensitivity': 'sensitivity',
            'specificity': 'specificity',
            'precision': 'precision',
            'f1': 'f1_score',
            'accuracy': 'accuracy',
            'auc_roc': 'auc_roc',
            'auc_pr': 'auc_pr'
        }
        
        # Extract available metrics
        metrics_to_plot = {}
        for hist_key, display_key in metric_mapping.items():
            if hist_key in history and history[hist_key]:
                metrics_to_plot[display_key] = history[hist_key]
        
        if not metrics_to_plot:
            print("Warning: No metrics found in history. Skipping plot_macro_metrics.")
            return
        
        epochs = range(1, len(next(iter(metrics_to_plot.values()))) + 1)
        
        # Create subplots
        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for idx, (metric_name, values) in enumerate(metrics_to_plot.items()):
            ax = axes[idx]
            ax.plot(epochs, values, 'o-', 
                   color=colors[idx % len(colors)], linewidth=2, markersize=6)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Score', fontsize=10)
            ax.set_title(metric_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # Hide extra subplots
        for idx in range(len(metrics_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Macro-Averaged Metrics Over Epochs - {model_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved macro metrics curves to {save_path}")
    
    def plot_per_class_metric(self, class_wise_metrics, metric='f1', class_name=None, 
                             save_path=None, model_name='Model'):
        """
        Plot a specific metric for a specific class over epochs
        
        Args:
            class_wise_metrics: Dict with class_name -> {metric: [...], ...}
            metric: Metric name to plot (e.g., 'f1', 'sensitivity', 'auc_roc')
            class_name: Class to plot. If None, plots all classes in subplots
            save_path: Path to save figure. If None, auto-generated from metric and class
            model_name: Name of the model
        """
        if not class_wise_metrics:
            print("Warning: class_wise_metrics is empty. Skipping plot_per_class_metric.")
            return
        
        if class_name is not None:
            # Plot single class
            if class_name not in class_wise_metrics:
                print(f"Warning: {class_name} not found in class_wise_metrics.")
                return
            
            if metric not in class_wise_metrics[class_name]:
                print(f"Warning: {metric} not found for {class_name}.")
                return
            
            if save_path is None:
                save_path = f"{metric}_{class_name}_over_epochs.png"
            
            values = class_wise_metrics[class_name][metric]
            epochs = range(1, len(values) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, values, 'o-', linewidth=2, markersize=6, color='#1f77b4')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.title(f'{class_name} - {metric.replace("_", " ").title()} Over Epochs - {model_name}',
                     fontsize=13, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved {metric} curve for {class_name} to {save_path}")
        
        else:
            # Plot all classes in subplots
            if save_path is None:
                save_path = f"{metric}_all_classes_over_epochs.png"
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, cls_name in enumerate(self.class_names):
                ax = axes[i]
                
                if cls_name in class_wise_metrics and metric in class_wise_metrics[cls_name]:
                    values = class_wise_metrics[cls_name][metric]
                    epochs = range(1, len(values) + 1)
                    ax.plot(epochs, values, 'o-', linewidth=2, markersize=6)
                    ax.set_ylim([0, 1])
                else:
                    ax.text(0.5, 0.5, f'{metric} not available', ha='center', va='center')
                
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
                ax.set_title(f'{cls_name}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            # Hide the last subplot
            axes[5].axis('off')
            
            plt.suptitle(f'{metric.replace("_", " ").title()} Over Epochs - {model_name}',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved {metric} curves for all classes to {save_path}")
    
    def plot_training_summary(self, history, save_path='training_summary.png', model_name='Model'):
        """
        Create a comprehensive summary grid: loss + key macro metrics
        
        Args:
            history: Training history dict with 'train_loss', 'val_loss', and metric lists
            save_path: Path to save figure
            model_name: Name of the model
        """
        if 'train_loss' not in history or not history['train_loss']:
            print("Warning: 'train_loss' not found. Skipping plot_training_summary.")
            return
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        # Plot 1: Loss
        axes[0].plot(epochs, history['train_loss'], 'o-', label='Train', linewidth=2)
        axes[0].plot(epochs, history.get('val_loss', []), 's-', label='Val', linewidth=2)
        axes[0].set_title('Loss', fontweight='bold')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2-6: Key macro metrics
        key_metrics = [
            ('f1', 'F1-Score'),
            ('auc_roc', 'AUC-ROC'),
            ('sensitivity', 'Sensitivity'),
            ('specificity', 'Specificity'),
            ('accuracy', 'Accuracy')
        ]
        
        for idx, (metric_key, display_name) in enumerate(key_metrics):
            ax = axes[idx + 1]
            if metric_key in history and history[metric_key]:
                ax.plot(epochs, history[metric_key], 'o-', linewidth=2, markersize=5)
                ax.set_ylim([0, 1])
            ax.set_title(display_name, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Summary - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training summary to {save_path}")
    
    def compare_models_training(self, histories: Dict[str, Dict], metric='f1',
                               class_wise_metrics_dict: Dict[str, Dict] = None,
                               save_path='models_comparison_training.png'):
        """
        Compare training progress of multiple models
        
        Args:
            histories: Dict mapping model_name -> history dict
            metric: Metric to compare for macro avg (e.g., 'f1', 'auc_roc')
            class_wise_metrics_dict: Optional - Dict mapping model_name -> class_wise_metrics for per-class comparison
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Loss curves
        ax = axes[0]
        for model_name, history in histories.items():
            if 'val_loss' in history and history['val_loss']:
                epochs = range(1, len(history['val_loss']) + 1)
                ax.plot(epochs, history['val_loss'], 'o-', label=f'{model_name}', linewidth=2, markersize=5)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Validation Loss', fontsize=11)
        ax.set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Metric curves (macro avg)
        ax = axes[1]
        for model_name, history in histories.items():
            if metric in history and history[metric]:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], 'o-', label=f'{model_name}', linewidth=2, markersize=5)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'Macro-Avg {metric.replace("_", " ").title()} Comparison', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.suptitle('Model Training Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved model training comparison to {save_path}")
