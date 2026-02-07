"""
Complete ECG Classification Model
Combines RNN block + optional Attention + FC layers
"""

import torch
import torch.nn as nn
from rnn_block import RNN_BLOCK, FullyConnected
from attention import AdditiveAttention, SelfAttention


class ECGModel(nn.Module):
    """
    Configurable ECG classification model
    Supports: GRU, BiGRU, LSTM, BiLSTM with optional Attention
    """
    def __init__(
        self,
        input_size=3,
        hidden_size=128,
        num_layers=2,
        num_classes=5,
        dropout=0.3,
        bidirectional=False,
        attention=None,  # None, 'additive', or 'self'
        rnn_type='gru'   # 'gru' or 'lstm'
    ):
        super(ECGModel, self).__init__()
        
        self.bidirectional = bidirectional
        self.attention_type = attention
        self.hidden_size = hidden_size 
        self.d = 2 if self.bidirectional else 1
        
        self.rnn_block = RNN_BLOCK(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            rnn_type=rnn_type
        )
        
        # Attention layer
        if attention == 'additive':
            self.attention = AdditiveAttention(hidden_size=hidden_size * self.d)
        elif attention == 'self':
            self.attention = SelfAttention(hidden_size=hidden_size * self.d)
        else:
            self.attention = None
        
        # Fully connected layers
        self.fc = FullyConnected(
            input_size=hidden_size * self.d,
            output_size=num_classes,
            dropout=dropout
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_leads, signal_length) = (batch, 3, 1000)
        Returns:
            logits: (batch, num_classes)
            attention_weights: (batch, seq_len) if attention, else None
        """
        # RNN forward pass
        rnn_out, hidden = self.rnn_block(x)
        # rnn_out: (batch, seq_len, hidden_size*d)
        # hidden: (batch, hidden_size*d)
        
        attention_weights = None
        
        # Apply attention if specified
        if self.attention_type == 'additive':
            context, attention_weights = self.attention(rnn_out)
            # context: (batch, hidden_size*d)
            fc_input = context
            
        elif self.attention_type == 'self':
            attended, attention_weights = self.attention(rnn_out)
            # attended: (batch, seq_len, hidden_size*d)
            # Use mean pooling over sequence
            fc_input = attended.mean(dim=1)  # (batch, hidden_size*d)
            
        else:
            # No attention - use hidden state from RNN
            fc_input = hidden
        
        # Fully connected layers
        logits = self.fc(fc_input)
        
        if attention_weights is not None:
            return (logits, attention_weights)
        else:
            return logits


# ============= Convenience Functions =================================================

def create_gru(hidden_size=128, num_layers=2, dropout=0.3):
    """Create unidirectional GRU model"""
    return ECGModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=False,
        attention=None,
        rnn_type='gru'
    )


def create_bigru(hidden_size=128, num_layers=2, dropout=0.3):
    """Create bidirectional GRU model"""
    return ECGModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
        attention=None,
        rnn_type='gru'
    )


def create_lstm(hidden_size=128, num_layers=2, dropout=0.3):
    """Create unidirectional LSTM model"""
    return ECGModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=False,
        attention=None,
        rnn_type='lstm'
    )


def create_bilstm(hidden_size=128, num_layers=2, dropout=0.3):
    """Create bidirectional LSTM model"""
    return ECGModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
        attention=None,
        rnn_type='lstm'
    )


def create_bigru_attention(hidden_size=128, num_layers=2, dropout=0.3, attention_type='additive'):
    """Create bidirectional GRU with attention"""
    return ECGModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
        attention=attention_type,  # 'additive' or 'self'
        rnn_type='gru'
    )