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
from config import Config

config = Config()

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


    def plot_confusion_matrices(self, y_true, y_pred, save_path = f'{config.METRICS_SAVE_PATH}/confusion_matrices.png',
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

    def plot_roc_curves(self, y_true, y_prob, save_path = f'{config.METRICS_SAVE_PATH}/roc_curves.png',
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
                                     save_path = f'{config.METRICS_SAVE_PATH}/pr_curves.png',
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

    def generate_metrics_report(self, metrics, save_path = f'{config.METRICS_SAVE_PATH}/metrics_report.txt'):
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

