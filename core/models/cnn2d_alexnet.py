"""
2D CNN Models for ECG Image Classification
- CNN2D (AlexNet)
- CNN2D with Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_attention import ChannelAttention, SpatialAttention


# ============= CNN2D (AlexNet) =======================================================
class CNN2D(nn.Module):
    """
    AlexNet inspired architecture adapted for ECG images
    we can use reduced filters for smaller dataset
    """
    def __init__(self, in_channels, out_channels, num_classes=5, dropout=0.5, attention_type='both', reduction_ratio = 16):
        super(CNN2D, self).__init__()
        # Features
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride = 2),
            
            nn.Conv2d(in_channels * 4, in_channels * 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride = 2),

            nn.Conv2d(in_channels * 16, in_channels * 32, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 32, in_channels * 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 64, out_channels, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride = 2),

        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride = 2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride = 2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.attention_layers = []
        
        if 'channel' in attention_type:
            self.attention_layers.append(ChannelAttention(in_channels, reduction_ratio))
        if 'spatial' in attention_type:
            self.attention_layers.append(ChannelAttention(in_channels, reduction_ratio))
        if 'self' in attention_type:
            self.attention_layers.append(ChannelAttention(in_channels, reduction_ratio))
            
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.num_classes = num_classes
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(out_channels * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            logits: (batch, num_classes)
        """
        out = self.features(x)
        out = self.avgpool(out)
        out = nn.Flatten(out, 1)
        logits = self.classifier(out)
        return logits


# ============= AlexNet with Attention =============
class AlexNetAttention(nn.Module):
    """
    AlexNet with Channel and Spatial Attention (CBAM)
    """
    def __init__(self, num_classes=4, dropout=0.5):
        super(AlexNetAttention, self).__init__()
        
        # TODO: Define same conv blocks as AlexNet
        
        # TODO: Add attention modules after certain conv blocks
        # Typically after conv3 or conv5
        
        # TODO: Define classifier
        
        pass
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            logits: (batch, num_classes)
        """
        # TODO: Conv blocks with attention inserted
        
        # TODO: Classifier
        
        pass