"""
RNN Models for ECG Classification
- GRU
- BiGRU  
- LSTM
- BiLSTM
- GRU with Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============= Base RNN Model =============
class BaseRNN(nn.Module):
    """Base class for RNN models"""
    def __init__(
        self,
        input_size=3,       # 3 leads
        hidden_size=128,
        num_layers=2,
        num_classes=5,      # NORM, MI, STTC, CD, HYP
        dropout=0.3,
        bidirectional=False,
        rnn_type='gru'      # 'gru' or 'lstm'
    ):
        super(BaseRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        # TODO: Create RNN layer (GRU or LSTM)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        # TODO: Calculate output size after RNN
        self.output_size = self.hidden_size * (2 if self.bidirectional else 1)
        # TODO: Create dense layers
        self.fc1 = nn.Linear(self.output_size, 64)
        self.fc2 = nn.Linear(64, self.num_classes)
        # TODO: Create batch normalization
        self.bn1 = nn.BatchNorm1d(self.num_classes)
        # TODO: Create dropout
        self.dropout1 = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_leads, signal_length) = (batch, 3, 1000)
        Returns:
            logits: (batch, num_classes) = (batch, 4)
        """
        # TODO: Transpose to (batch, seq_len, features)
        x_transposed = x[:, ]
        # TODO: RNN forward pass
        # TODO: Extract hidden state
        # TODO: Dense layers
        # TODO: Return logits
        pass


# ============= GRU Model =============
class GRUModel(nn.Module):
    """Unidirectional GRU"""
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=4, dropout=0.3):
        # TODO: Implement using BaseRNN or from scratch
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass


# ============= BiGRU Model =============
class BiGRUModel(nn.Module):
    """Bidirectional GRU - Best performer from paper"""
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=4, dropout=0.3):
        # TODO: Implement
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        # Remember to concatenate forward and backward hidden states
        pass


# ============= LSTM Model =============
class LSTMModel(nn.Module):
    """Unidirectional LSTM"""
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=4, dropout=0.3):
        # TODO: Implement
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass


# ============= BiLSTM Model =============
class BiLSTMModel(nn.Module):
    """Bidirectional LSTM"""
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=4, dropout=0.3):
        # TODO: Implement
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass


# ============= Attention Mechanism =============
class AttentionLayer(nn.Module):
    """
    Attention mechanism for sequence data
    Computes attention weights over sequence and returns weighted sum
    """
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        # TODO: Create attention weight layer
        # Typically: Linear layer to compute attention scores
        pass
    
    def forward(self, rnn_output):
        """
        Args:
            rnn_output: (batch, seq_len, hidden_size * num_directions)
        Returns:
            context: (batch, hidden_size * num_directions)
            attention_weights: (batch, seq_len)
        """
        # TODO: Compute attention scores
        # TODO: Apply softmax to get weights
        # TODO: Compute weighted sum (context vector)
        pass


# ============= BiGRU with Attention =============
class BiGRUAttention(nn.Module):
    """Bidirectional GRU with Attention mechanism"""
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=4, dropout=0.3):
        super(BiGRUAttention, self).__init__()
        
        # TODO: Create BiGRU layer
        # TODO: Create Attention layer
        # TODO: Create dense layers
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_leads, signal_length)
        Returns:
            logits: (batch, num_classes)
            attention_weights: (batch, seq_len) - for visualization
        """
        # TODO: Transpose to (batch, seq_len, features)
        # TODO: BiGRU forward pass
        # TODO: Apply attention to RNN outputs
        # TODO: Dense layers
        # TODO: Return logits and attention weights
        pass