"""
ECG-FM Core Module
Provides data loading, model definitions, and training utilities.
"""

from . import data
from . import models
from . import training
from . import metrics

__all__ = [
    'data',
    'models',
    'training',
    'metrics',
]

__version__ = '1.0.0'
