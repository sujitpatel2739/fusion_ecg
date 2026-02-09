import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import os

def save_histories(histories, model_names, save_dir='core/histories'):
    """
    Save training histories for each model to disk.
    
    Args:
        histories: List of history dictionaries (one per model)
        model_names: List of model name strings
        save_dir: Directory to save history files (default: core/training/saved)
    
    Returns:
        List of saved file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    for model_name, history in zip(model_names, histories):
        # Create filename with model name and timestamp
        filename = f"{model_name}_history_{timestamp}.pkl"
        filepath = os.path.join(save_dir, filename)
        
        # Save history dict with pickle
        with open(filepath, 'wb') as f:
            pickle.dump(history, f)
        
        saved_files.append(filepath)
        print(f"✓ Saved {model_name} history to {filepath}")
    
    return saved_files

def load_history(filepath):
    """
    Load a single training history from disk.
    
    Args:
        filepath: Path to the pickled history file
    
    Returns:
        history dictionary
    """
    with open(filepath, 'rb') as f:
        history = pickle.load(f)
    
    return history


def load_all_histories(save_dir='core/training/saved', pattern=None):
    """
    Load all or filtered training histories from save directory.
    
    Args:
        save_dir: Directory containing history files
        pattern: Optional substring to filter files (e.g., 'JointFusion')
    
    Returns:
        Dictionary mapping filenames to histories
    """
    histories_dict = {}
    
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} not found.")
        return histories_dict
    
    for filename in os.listdir(save_dir):
        if filename.endswith('.pkl'):
            if pattern is None or pattern in filename:
                filepath = os.path.join(save_dir, filename)
                try:
                    history = load_history(filepath)
                    histories_dict[filename] = history
                    print(f"✓ Loaded {filename}")
                except Exception as e:
                    print(f"✗ Failed to load {filename}: {e}")
    
    return histories_dict


def plot_training_history(histories, model_names):
    """
    Plot training curves for all models
    
    Args:
        histories: List of history dictionaries
        model_names: List of model names
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Helper to plot if metric exists
    def _plot_metric(ax, key_names, title, ylabel=None):
        plotted = False
        for history, name in zip(histories, model_names):
            for key in key_names:
                if key in history:
                    ax.plot(history[key], label=name)
                    plotted = True
                    break
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True)
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Metric not available', ha='center', va='center', fontsize=10, color='gray')

    # Plotting metrics with fallbacks for common key names
    _plot_metric(axes[0], ['train_loss', 'loss'], 'Training Loss', ylabel='Loss')
    _plot_metric(axes[1], ['val_loss', 'validation_loss'], 'Validation Loss', ylabel='Loss')
    _plot_metric(axes[2], ['sensitivity', 'recall', 'sens'], 'Sensitivity / Recall', ylabel='Sensitivity')
    _plot_metric(axes[3], ['specificity', 'precision', 'spec'], 'Specificity / Precision', ylabel='Score')

    plt.tight_layout()
    out_path = 'training_curves.png'
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved training curves to {out_path}")