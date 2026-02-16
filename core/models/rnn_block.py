"""
RNN Models for ECG Classification
- GRU
- BiGRU  
- LSTM
- BiLSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN_BLOCK(nn.Module):
    """Base class for GRU/LSTM models"""
    def __init__(
        self,
        input_size=3,       # 3 leads
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
        rnn_type='gru'      # 'gru' or 'lstm'
    ):
        super(RNN_BLOCK, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.d = 2 if self.bidirectional else 1

        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unidentified rnn type: {rnn_type}")

    def forward(self, x):
        """
        Args:
            x: (batch, num_leads, signal_length) = (batch, 3, 1000)
        Returns:
            out: (batch, seq_length, hidden_size * d)
            hidden: (batch, hidden_size * d)
        """
        device = x.device

        # Transpose to (batch, seq_len, features)
        x_transposed = x.transpose(1, 2)  # (batch, 1000, 3)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * self.d, x.size(0), self.hidden_size).to(device)

        if self.rnn_type == 'gru':
            out, hn = self.rnn(x_transposed, h0)
        elif self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers * self.d, x.size(0), self.hidden_size).to(device)
            out, (hn, cn) = self.rnn(x_transposed, (h0, c0))

        # Extract hidden state
        if not self.bidirectional:
            hidden = hn[-1]  # (batch, hidden_size)
        else:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hn[-2], hn[-1]], dim=1)  # (batch, hidden_size*2)

        return out, hidden


class FullyConnected(nn.Module):
    """Fully connected layers for classification"""
    def __init__(self, input_size, output_size, dropout=0.3):
        super(FullyConnected, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size * 2)
        self.bn1 = nn.BatchNorm1d(input_size * 2)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(input_size * 2, input_size)
        self.bn2 = nn.BatchNorm1d(input_size)
        self.dropout2 = nn.Dropout(dropout)

        self.output = nn.Linear(input_size, output_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        """
        Args:
            x: (batch, input_size)
        Returns:
            logits: (batch, output_size)
        """
        device = x.device

        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        logits = self.output(out)
        return logits