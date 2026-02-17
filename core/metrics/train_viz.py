import os
import matplotlib.pyplot as plt
from typing import Dict
from config import Config

config = Config()

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

    def plot_loss_curves(self, history, save_path = f'{config.TRAIN_HISTORY_SAVE_PATH}/loss_curves.png', model_name='Model'):
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

    def plot_macro_metrics(self, history, save_path = f'{config.TRAIN_HISTORY_SAVE_PATH}/macro_metrics.png', model_name='Model'):
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
                              save_path = f'{config.TRAIN_HISTORY_SAVE_PATH}/per_class_metrics.png', model_name='Model'):
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

    def plot_training_summary(self, history, save_path = f'{config.TRAIN_HISTORY_SAVE_PATH}/training_summary.png', model_name='Model'):
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
                                save_path = f'{config.TRAIN_HISTORY_SAVE_PATH}/model_training_comparison.png'):
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

       