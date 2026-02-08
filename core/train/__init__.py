"""
Training utilities and training loop implementations.
Contains training logic for RNN models.
"""

try:
    from .train_rnn import train_model
    __all__ = [
        'train_model',
    ]
except ImportError:
    # Handle case where train_rnn.py is not yet created
    __all__ = []
