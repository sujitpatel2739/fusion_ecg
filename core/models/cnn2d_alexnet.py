"""
2D CNN Models for ECG Image Classification
- CNN2D (AlexNet-inspired)
- CNN2D with various attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_attention import ChannelAttention, SpatialAttention, SelfAttentionConv


# ============= CNN2D (AlexNet-inspired) ==============================================
class CNN2D(nn.Module):
    """
    AlexNet-inspired architecture adapted for ECG images
    Supports multiple attention mechanisms: channel, spatial, self, or combinations
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB-like images)
        base_filters: Base number of filters (default: 16 for small datasets)
        num_classes: Number of output classes (default: 5 for ECG)
        dropout: Dropout rate (default: 0.5)
        attention_type: None, 'channel', 'spatial', 'cbam', or 'self'
        reduction_ratio: Channel reduction ratio for attention (default: 16)
    """
    def __init__(
        self,
        in_channels=3,
        base_filters=16,
        num_classes=5,
        dropout=0.5,
        attention_type=None,  # None, 'channel', 'spatial', 'cbam', 'self'
        reduction_ratio=16
    ):
        super(CNN2D, self).__init__()
        
        self.attention_type = attention_type
        
        # Calculate channel sizes
        c1 = base_filters          # 16
        c2 = base_filters * 3      # 48
        c3 = base_filters * 6      # 96
        c4 = base_filters * 4      # 64
        c5 = base_filters * 4      # 64
        
        # Feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Attention after conv3 (if specified)
        self.attention = self._build_attention(c3, attention_type, reduction_ratio)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to fixed size
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(c5 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
    
    def _build_attention(self, num_channels, attention_type, reduction_ratio):
        """
        Build attention module based on type
        
        Args:
            num_channels: Number of channels at this layer
            attention_type: Type of attention
            reduction_ratio: Channel reduction ratio
        
        Returns:
            nn.Module or None
        """
        if attention_type is None or attention_type == 'none':
            return None
        
        elif attention_type == 'channel':
            return ChannelAttention(num_channels, reduction_ratio)
        
        elif attention_type == 'spatial':
            return SpatialAttention(num_channels, kernel_size=7)
        
        elif attention_type == 'cbam':
            # CBAM = Channel + Spatial attention
            return nn.Sequential(
                ChannelAttention(num_channels, reduction_ratio),
                SpatialAttention(num_channels, kernel_size=7)
            )
        
        elif attention_type == 'self':
            return SelfAttentionConv(num_channels, reduction_ratio)
        
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            logits: (batch, num_classes)
        """
        # Feature extraction
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        # Attention
        if self.attention is not None:
            out = self.attention(out)
        
        out = self.conv4(out)
        out = self.conv5(out)
        
        # Pooling
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits


# ============= Convenience Functions =================================================

def create_alexnet(num_classes=5, dropout=0.5):
    """Create AlexNet without attention"""
    return CNN2D(
        in_channels=3,
        base_filters=16,
        num_classes=num_classes,
        dropout=dropout,
        attention_type=None
    )


def create_alexnet_channel_attention(num_classes=5, dropout=0.5):
    """Create AlexNet with channel attention"""
    return CNN2D(
        in_channels=3,
        base_filters=16,
        num_classes=num_classes,
        dropout=dropout,
        attention_type='channel'
    )


def create_alexnet_spatial_attention(num_classes=5, dropout=0.5):
    """Create AlexNet with spatial attention"""
    return CNN2D(
        in_channels=3,
        base_filters=16,
        num_classes=num_classes,
        dropout=dropout,
        attention_type='spatial'
    )


def create_alexnet_cbam(num_classes=5, dropout=0.5):
    """Create AlexNet with CBAM (Channel + Spatial attention)"""
    return CNN2D(
        in_channels=3,
        base_filters=16,
        num_classes=num_classes,
        dropout=dropout,
        attention_type='cbam'
    )


def create_alexnet_self_attention(num_classes=5, dropout=0.5):
    """Create AlexNet with self-attention (Conv-based)"""
    return CNN2D(
        in_channels=3,
        base_filters=16,
        num_classes=num_classes,
        dropout=dropout,
        attention_type='self',
        reduction_ratio=8 
    )