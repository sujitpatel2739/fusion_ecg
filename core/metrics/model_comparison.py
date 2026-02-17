import matplotlib.pyplot as plt
from typing import Dict
import numpy as np
import seaborn as sns
import pandas as pd
from config import Config

config = Config()

class ModelComparison:
    """Compare multiple models' performance"""

    def __init__(self, class_names=['NORM', 'MI', 'STTC', 'CD', 'HYP']):
        self.class_names = class_names

    def compare_models(self, models_metrics: Dict[str, Dict], save_path = f'{config.METRICS_SAVE_PATH}/model_comparison.png'):
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
                                save_path = f'{config.METRICS_SAVE_PATH}/comparison_table.csv'):
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
                           metric='f1_score', save_path = f'{config.METRICS_SAVE_PATH}/metric_heatmap.png'):
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

