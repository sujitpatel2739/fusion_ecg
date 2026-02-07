"""
Training script for RNN models on ECG signals
Tests if models are learning properly
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Import your models
from models import create_gru, create_bigru, create_bigru_attention

# Configuration
class Config:
    # Data paths
    TRAIN_SIGNALS = 'data/signals/X_train.npy'
    TRAIN_LABELS = 'data/labels/y_train.npy'
    VAL_SIGNALS = 'data/signals/X_validation.npy'
    VAL_LABELS = 'data/labels/y_validation.npy'
    
    # Model parameters
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10  # Small number for testing
    LEARNING_RATE = 0.001
    
    # Other
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = 'checkpoints/test_run'
    NUM_WORKERS = 4
    
    # Classes
    CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    NUM_CLASSES = 5
    
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for signals, labels in tqdm(dataloader, desc='Training on Signals'):
        signals = signals.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        output = model(signals)
        if isinstance(output, tuple):
            logits, att_weights = output  # Model with attention
        else:
            logits = output
            
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss/len(dataloader)