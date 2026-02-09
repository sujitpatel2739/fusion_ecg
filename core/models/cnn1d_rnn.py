import torch
import torch.nn as nn

from core.models.rnn_block import RNN_BLOCK

class CRNN_1D(nn.Module):
    """
    1D Convolutional Recurrent Neural Network.
    As described in the official paper.
    """
    def __init__(
        self,
        input_channels=3,
        num_classes=5,
        rnn_type='gru',
        hidden_size=128,
        num_rnn_layers=2,
        bidirectional = False,
        dropout=0.5
    ):
        super(CRNN_1D, self).__init__()
        
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        self.cnn_features = nn.Sequential(
            nn.Conv1d(input_channels, input_channels*11, kernel_size=5, padding=2),
            nn.BatchNorm1d(input_channels*11),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(input_channels*11, input_channels*22, kernel_size=5, padding=2),
            nn.BatchNorm1d(input_channels*22),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(input_channels*22, input_channels*44, kernel_size=5, padding=2),
            nn.BatchNorm1d(input_channels*44),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # Calculating sequence length after CNN
        # Input: 1000; After 3 maxpools (stride=2): 1000/8 = 125
        self.seq_len = 125
        self.cnn_output_channels = input_channels*44 # 132
        
        self.rnn = RNN_BLOCK(
            input_size=self.cnn_output_channels,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            dropout=dropout,
            bidirectional=self.bidirectional,
            rnn_type=self.rnn_type
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 1000) - Raw ECG signals
        Returns:
            logits: (batch, 5)
        """
        # CNN feature extraction
        features = self.cnn_features(x)  # (batch, 132, 125)
        out, hidden = self.rnn(features)
        # Classification
        logits = self.classifier(hidden)  # (batch, 5)
        
        return logits
