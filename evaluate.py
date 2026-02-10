"""
Evaluation for ECG classification models.
Supports signal-based (RNN, CNN1D), image-based (CNN2D), and fusion models.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from tqdm import tqdm
import torch.nn as nn

from config import Config
from core.metrics.metrics import ECGMetrics, ModelComparison
from core.data.signal_dataloader import create_signal_dataloader
from core.data.image_dataloader import create_image_dataloader
from core.data.fusion_dataloader import create_fusion_dataloader

# Import model creation functions
from core.models.rnn import (
    create_gru, create_bigru, create_lstm, create_bilstm, create_bigru_attention
)
from core.models.cnn2d_alexnet import (
    create_alexnet, create_alexnet_channel_attention, 
    create_alexnet_spatial_attention, create_alexnet_cbam, create_alexnet_self_attention
)

# Import model classes for direct instantiation
from core.models.cnn1d_rnn import CNN1D_RNN
from core.models.cnn2d_resnet import ResNet
from core.models.cnn2d_vggnet import VGGNet
from core.models.joint_fusion import JointFusion


# ==================== MODEL LOADING ==================================================

def load_model(model_name: str, checkpoint_path: str, 
               config: Config) -> Tuple[nn.Module, str]:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_name: Name/key of the model
        checkpoint_path: Path to saved checkpoint
        config: Config object with hyperparameters
        
    Returns:
        (model, model_type): Loaded model and its type ('signal', 'image', 'fusion')
    """
    print(f"\nLoading {model_name}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model_state = checkpoint['model_state_dict']
    
    # Create model based on name
    if 'GRU' in model_name:
        model = create_gru(config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
        model_type = 'signal'
    
    elif 'BiGRU' in model_name:
        model = create_bigru(config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
        model_type = 'signal'
    
    elif 'LSTM' in model_name:
        model = create_lstm(config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
        model_type = 'signal'
    
    elif 'BiLSTM' in model_name:
        model = create_bilstm(config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
        model_type = 'signal'
    
    elif 'CNN1D' in model_name or 'CNN1D_RNN' in model_name:
        model = CNN1D_RNN(in_channels=3, hidden_size=config.HIDDEN_SIZE,
                         num_layers=config.NUM_LAYERS, num_classes=config.NUM_CLASSES,
                         dropout=config.DROPOUT)
        model_type = 'signal'
    
    elif 'AlexNet' in model_name or 'CNN2D' in model_name:
        model = create_alexnet(num_classes=config.NUM_CLASSES, dropout=config.DROPOUT)
        model_type = 'image'
    
    elif 'ResNet' in model_name:
        model = ResNet(layers=[2, 2, 2, 2], num_classes=config.NUM_CLASSES)
        model_type = 'image'
    
    elif 'VGGNet' in model_name or 'VGG' in model_name:
        model = VGGNet(num_classes=config.NUM_CLASSES)
        model_type = 'image'
    
    elif 'Fusion' in model_name or 'JointFusion' in model_name:
        model = JointFusion(in_channels=3, out_channels=128,
                           num_classes=config.NUM_CLASSES)
        model_type = 'fusion'
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    model.load_state_dict(model_state)
    model.to(config.DEVICE)
    model.eval()
    
    print(f"Loaded {model_name} ({model_type})")
    return model, model_type


# ==================== DATALOADERS ==================================================

def get_test_loader(model_type: str, config: Config):
    """
    Get appropriate test dataloader for model type.
    
    Args:
        model_type: 'signal', 'image', or 'fusion'
        config: Config object
        
    Returns:
        DataLoader for test set
    """
    
    if model_type == 'signal':
        return create_signal_dataloader(
            config.TEST_SIGNAL_PATH, config.TEST_LABEL_PATH,
            batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS
        )
    
    elif model_type == 'image':
        return create_image_dataloader(
            config.TEST_IMAGE_PATH, config.TEST_LABEL_PATH,
            transform_type='gaf', batch_size=config.BATCH_SIZE,
            shuffle=False, num_workers=config.NUM_WORKERS
        )
    
    elif model_type == 'fusion':
        return create_fusion_dataloader(
            config.TEST_SIGNAL_PATH, config.TEST_IMAGE_PATH,
            config.TEST_LABEL_PATH, signal_window=1000,
            transform_type='gaf', batch_size=config.BATCH_SIZE,
            shuffle=False, num_workers=config.NUM_WORKERS
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ==================== EVALUATION ==================================================

def evaluate_model(model: nn.Module, test_loader, model_type: str,
                  metrics_calc: ECGMetrics, device: str) -> Dict:
    """
    Evaluate single model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test dataloader
        model_type: 'signal', 'image', or 'fusion'
        metrics_calc: ECGMetrics calculator
        device: 'cuda' or 'cpu'
        
    Returns:
        Dictionary with all metrics and predictions
    """
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, labels in tqdm(test_loader, desc="Evaluating"):
            if isinstance(x, (tuple,list)):
                x = (x[0].to(device), x[1].to(device))
            else:
                x = x.to(device)
            
            labels = labels.to(device)
            
            # Forward pass
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Store
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate  
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_predictions)
    y_prob = np.vstack(all_probs)
    
    # Calculate metrics
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_prob)
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    metrics['y_prob'] = y_prob
    
    return metrics


def evaluate_all_models(models_config: Dict[str, str], config: Config) -> Dict:
    """
    Evaluate all models in configuration.
    
    Args:
        models_config: Dict mapping model_name -> checkpoint_path
        config: Config object
        
    Returns:
        Dictionary with results for all models
    """
    metrics_calc = ECGMetrics(class_names=config.CLASS_NAMES)
    all_results = {}
    
    print("\n" + "="*50)
    print("EVALUATING MODELS")
    print("="*50)
    
    for model_name, checkpoint_path in models_config.items():
        if not os.path.exists(checkpoint_path):
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
            continue
        
        try:
            model, model_type = load_model(model_name, checkpoint_path, config)
            test_loader = get_test_loader(model_type, config)
            metrics = evaluate_model(model, test_loader, model_type, 
                                    metrics_calc, config.DEVICE)
            
            # Print summary
            m = metrics['macro_avg']
            print(f"  F1: {m['f1_score']:.4f} | Sens: {m['sensitivity']:.4f} | AUC: {m['auc_roc']:.4f}")
            
            all_results[model_name] = metrics
            
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
            continue
    
    return all_results


# ==================== REPORT GENERATION ==================================================

def generate_macro_table(results: Dict, output_dir: str):
    """
    Generate macro-averaged comparison table.
    
    Args:
        results: Dict with model results
        output_dir: Directory to save CSV
    """
    
    data = []
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Sensitivity': metrics['macro_avg']['sensitivity'],
            'Specificity': metrics['macro_avg']['specificity'],
            'Precision': metrics['macro_avg']['precision'],
            'F1-Score': metrics['macro_avg']['f1_score'],
            'Accuracy': metrics['macro_avg']['accuracy'],
            'AUC-ROC': metrics['macro_avg']['auc_roc'],
            'AUC-PR': metrics['macro_avg']['auc_pr']
        }
        data.append(row)
    
    df = pd.DataFrame(data).sort_values('F1-Score', ascending=False)
    
    print("\n" + "="*50)
    print("TABLE 1: Macro-Averaged Metrics (All Models)")
    print("="*50)
    print(df.to_string(index=False))
    
    output_file = os.path.join(output_dir, '01_macro_comparison.csv')
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Saved: {output_file}")


def generate_per_class_tables(results: Dict, class_names: list, output_dir: str):
    """
    Generate per-class comparison tables for F1, Sensitivity, AUC-ROC.
    
    Args:
        results: Dict with model results
        class_names: List of class names
        output_dir: Directory to save CSVs
    """
    
    metrics_list = [
        ('f1_score', 'F1-Score', '02_per_class_f1_comparison.csv'),
        ('sensitivity', 'Sensitivity', '03_per_class_sensitivity_comparison.csv'),
        ('auc_roc', 'AUC-ROC', '04_per_class_auc_roc_comparison.csv'),
    ]
    
    for idx, (metric_key, metric_name, filename) in enumerate(metrics_list, 2):
        data = []
        for model_name, metrics in results.items():
            row = {'Model': model_name}
            for class_name in class_names:
                row[class_name] = metrics[class_name][metric_key]
            row['Macro-Avg'] = metrics['macro_avg'][metric_key]
            data.append(row)
        
        df = pd.DataFrame(data).sort_values('Macro-Avg', ascending=False)
        output_file = os.path.join(output_dir, filename)
        df.to_csv(output_file, index=False, float_format='%.4f')
        
        print(f"\nTABLE {idx}: Per-Class {metric_name}")
        print("="*50)
        print(df.to_string(index=False))
        print(f"Saved: {output_file}")


def generate_best_model_table(results: Dict, class_names: list, output_dir: str) -> str:
    """
    Generate detailed table for best performing model.
    
    Args:
        results: Dict with model results
        class_names: List of class names
        output_dir: Directory to save CSV
        
    Returns:
        Name of best model
    """
    
    # Find best model
    best_model = max(results.items(), 
                     key=lambda x: x[1]['macro_avg']['f1_score'])
    best_name, best_metrics = best_model
    
    data = []
    for class_name in class_names:
        m = best_metrics[class_name]
        data.append({
            'Class': class_name,
            'TP': m['tp'], 'FP': m['fp'], 'FN': m['fn'], 'TN': m['tn'],
            'Sensitivity': m['sensitivity'], 'Specificity': m['specificity'],
            'Precision': m['precision'], 'F1-Score': m['f1_score'],
            'Accuracy': m['accuracy'], 'AUC-ROC': m['auc_roc'], 'AUC-PR': m['auc_pr'],
        })
    
    df = pd.DataFrame(data)
    
    print("\nTABLE 5: Best Model - Detailed Metrics")
    print("="*50)
    print(f"Model: {best_name}")
    print(df.to_string(index=False))
    
    output_file = os.path.join(output_dir, '05_best_model_detailed.csv')
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Saved: {output_file}")
    
    return best_name


def generate_visualizations(results: Dict, class_names: list, output_dir: str):
    """
    Generate model comparison visualizations.
    
    Args:
        results: Dict with model results
        class_names: List of class names
        output_dir: Directory to save visualizations
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    comparator = ModelComparison(class_names=class_names)
    
    # 1. Model comparison bar plot
    comparator.compare_models(
        results,
        save_path=os.path.join(output_dir, '06_model_comparison_bars.png')
    )
    print("Saved: 06_model_comparison_bars.png")
    
    # 2. Metric heatmaps
    for metric in ['f1_score', 'sensitivity', 'specificity', 'auc_roc']:
        comparator.plot_metric_heatmap(
            results,
            metric=metric,
            save_path=os.path.join(output_dir, f'07_heatmap_{metric}.png')
        )
        print(f"Saved: 07_heatmap_{metric}.png")


def generate_best_model_analysis(results: Dict, class_names: list,
                                best_model_name: str, output_dir: str):
    """
    Generate detailed analysis plots for best model.
    
    Args:
        results: Dict with model results
        class_names: List of class names
        best_model_name: Name of best model
        output_dir: Directory to save plots
    """
    
    best_metrics = results[best_model_name]
    metrics_calc = ECGMetrics(class_names=class_names)
    
    print(f"\nGenerating detailed analysis for {best_model_name}...")
    
    # Extract predictions
    y_true = best_metrics['y_true']
    y_pred = best_metrics['y_pred']
    y_prob = best_metrics['y_prob']
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate plots
    metrics_calc.plot_confusion_matrices(
        y_true, y_pred,
        save_path=os.path.join(viz_dir, '08_confusion_matrices.png'),
        model_name=best_model_name
    )
    print("Saved: 08_confusion_matrices.png")
    
    metrics_calc.plot_roc_curves(
        y_true, y_prob,
        save_path=os.path.join(viz_dir, '09_roc_curves.png'),
        model_name=best_model_name
    )
    print("Saved: 09_roc_curves.png")
    
    metrics_calc.plot_precision_recall_curves(
        y_true, y_prob,
        save_path=os.path.join(viz_dir, '10_pr_curves.png'),
        model_name=best_model_name
    )
    print("Saved: 10_pr_curves.png")
    
    # Generate text report
    metrics_calc.generate_metrics_report(
        best_metrics,
        save_path=os.path.join(output_dir, 'DETAILED_METRICS_REPORT.txt')
    )
    print("Saved: DETAILED_METRICS_REPORT.txt")


def save_summary_report(results: Dict, class_names: list, output_file: str):
    """
    Save comprehensive summary report as text.
    
    Args:
        results: Dict with model results
        class_names: List of class names
        output_file: Path to save report
    """
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find best model
    best_model_name, best_metrics = max(
        results.items(),
        key=lambda x: x[1]['macro_avg']['f1_score']
    )
    
    with open(output_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write("ECG CLASSIFICATION - MODEL EVALUATION REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Evaluation Date: {pd.Timestamp.now()}\n")
        f.write(f"Number of Models Evaluated: {len(results)}\n")
        f.write(f"Class Names: {', '.join(class_names)}\n\n")
        
        # Summary metrics
        f.write("-"*80 + "\n")
        f.write("MACRO-AVERAGED METRICS (All Models)\n")
        f.write("-"*80 + "\n\n")
        
        for model_name, metrics in sorted(
            results.items(),
            key=lambda x: x[1]['macro_avg']['f1_score'],
            reverse=True
        ):
            m = metrics['macro_avg']
            f.write(f"{model_name}:\n")
            f.write(f"  F1-Score:    {m['f1_score']:.4f}\n")
            f.write(f"  Sensitivity: {m['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {m['specificity']:.4f}\n")
            f.write(f"  Precision:   {m['precision']:.4f}\n")
            f.write(f"  Accuracy:    {m['accuracy']:.4f}\n")
            f.write(f"  AUC-ROC:     {m['auc_roc']:.4f}\n")
            f.write(f"  AUC-PR:      {m['auc_pr']:.4f}\n\n")
        
        # Best model details
        f.write("\n" + "="*50 + "\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write("="*50 + "\n\n")
        
        f.write("Per-Class Performance:\n")
        f.write("-"*80 + "\n")
        for class_name in class_names:
            m = best_metrics[class_name]
            f.write(f"\n{class_name}:\n")
            f.write(f"  Sensitivity: {m['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {m['specificity']:.4f}\n")
            f.write(f"  Precision:   {m['precision']:.4f}\n")
            f.write(f"  F1-Score:    {m['f1_score']:.4f}\n")
            f.write(f"  AUC-ROC:     {m['auc_roc']:.4f}\n")
            f.write(f"  Support:     {m['support']}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("OUTPUT FILES:\n")
        f.write("-"*80 + "\n")
        f.write("  01_macro_comparison.csv - Macro metrics for all models\n")
        f.write("  02_per_class_f1_comparison.csv - Per-class F1-scores\n")
        f.write("  03_per_class_sensitivity_comparison.csv - Per-class sensitivity\n")
        f.write("  04_per_class_auc_roc_comparison.csv - Per-class AUC-ROC\n")
        f.write("  05_best_model_detailed.csv - Best model details\n")
        f.write("  06-10_*.png - Visualizations (comparisons, confusion, ROC, PR)\n")
        f.write("  DETAILED_METRICS_REPORT.txt - Extended metrics report\n")
        f.write("="*50 + "\n")
    
    print(f"Summary report saved to {output_file}")
    

def main():
    """Main evaluation pipeline"""
    
    print("\n" + "="*50)
    print("ECG CLASSIFICATION - MODEL EVALUATION")
    print("="*50)
    
    config = Config()
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define models to evaluate
    models_config = {
        'RNNModel': os.path.join(config.SAVE_DIR, 'RNNModel_best.pth'),
        'CNN1D_RNN': os.path.join(config.SAVE_DIR, 'CNN1D_RNN_best.pth'),
        'ResNet': os.path.join(config.SAVE_DIR, 'ResNet_best.pth'),
        'VGGNet': os.path.join(config.SAVE_DIR, 'VGGNet_best.pth'),
        'AlexNet': os.path.join(config.SAVE_DIR, 'AlexNet_best.pth'),
        'JointFusion': os.path.join(config.SAVE_DIR, 'JointFusion_best.pth'),
    }
    
    # Evaluate all available models
    results = evaluate_all_models(models_config, config)
    
    if results:
        # Generate outputs
        print("\n" + "="*50)
        print("GENERATING REPORTS AND VISUALIZATIONS")
        print("="*50)
        
        generate_macro_table(results, output_dir)
        generate_per_class_tables(results, config.CLASS_NAMES, output_dir)
        best_model = generate_best_model_table(results, config.CLASS_NAMES, output_dir)
        generate_visualizations(results, config.CLASS_NAMES, output_dir)
        generate_best_model_analysis(results, config.CLASS_NAMES, best_model, output_dir)
        save_summary_report(results, config.CLASS_NAMES, 
                           os.path.join(output_dir, 'EVALUATION_SUMMARY.txt'))
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETE!")
        print("="*50)
        print(f"\nResults saved to: {output_dir}/")
        print("  - CSV tables: *.csv")
        print("  - Visualizations: visualizations/")
        print("  - Summary report: EVALUATION_SUMMARY.txt")
        print("="*50 + "\n")
    else:
        print("\nNo models were successfully evaluated.")


if __name__ == '__main__':
    main()
