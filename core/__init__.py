"""
ECG-FM Core Module
Provides data loading, model definitions, and training utilities.
"""

from . import data
from . import models
from . import train
from . import metrics

__all__ = [
    'data',
    'models',
    'train',
    'metrics',
]

__version__ = '1.0.0'
