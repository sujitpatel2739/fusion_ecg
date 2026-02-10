"""
Attention mechanisms for ECG classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, scale):
    """
    Scaled dot-product attention mechanism
    
    Args:
        Q: Query (batch, seq_len, hidden_size)
        K: Key (batch, seq_len, hidden_size)
        V: Value (batch, seq_len, hidden_size)
        scale: Scaling factor (usually sqrt(hidden_size))
    
    Returns:
        output: (batch, seq_len, hidden_size)
        weights: (batch, seq_len, seq_len)
    """
    scores = torch.bmm(Q, K.transpose(1, 2)) / scale  # (batch, seq_len, seq_len)
    weights = F.softmax(scores, dim=-1)
    output = torch.bmm(weights, V)  # (batch, seq_len, hidden_size)
    return output, weights


class AdditiveAttention(nn.Module):
    """
    Additive attention (Bahdanau attention)
    Computes attention weights over sequence and returns weighted sum
    """
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        # Compute attention scores
        attention_scores = self.attention(x)  # (batch, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, seq_len)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Compute context vector (weighted sum)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            x  # (batch, seq_len, hidden_size)
        )  # (batch, 1, hidden_size)
        context = context.squeeze(1)  # (batch, hidden_size)
        
        return context, attention_weights


class SelfAttention(nn.Module):
    """
    Self-attention mechanism
    """
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__() 
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            attention: (batch, seq_len, hidden_size)
            weights: (batch, seq_len, seq_len)
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention, weights = scaled_dot_product_attention(Q, K, V, self.scale)
        return attention, weights
    
    
class CrossAttention:
    """
    Note: This is private and experimental. 
    Neither allowed to embed into training, nor allowed to replicate.
    Copyright protected.
    Author: Sujit Patel.
    https://github.com/sujitpatel2739/fusion_ecg
    """
    def __init__(self, hidden_size):
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5
        
    def forward(self, x, context):
        """
        Args:
            x: (batch, seq_len_x, hidden_size) - query source
            context: (batch, seq_len_c, hidden_size) - key/value source
        """
        
        Q = self.query(x)
        K = self.key(context)
        V = self.value(context)
        
        attention, weights = scaled_dot_product_attention(Q, K, V, self.scale)
        return attention, weights