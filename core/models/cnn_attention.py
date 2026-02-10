import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel attention module (from CBAM paper)
    Focuses on 'WHAT' is meaningful
    """
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            out: (batch, channels, height, width)
        """
        batch, channels, _, _ = x.size()
        
        # Average pooling
        avg_out = self.avg_pool(x).view(batch, channels)
        avg_out = self.fc(avg_out)
        
        # Max pooling
        max_out = self.max_pool(x).view(batch, channels)
        max_out = self.fc(max_out)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        attention = attention.view(batch, channels, 1, 1)
        
        return x * attention.expand_as(x)

class SpatialAttention(nn.Module):
    """
    Spatial attention module (from CBAM paper)
    Focuses on 'WHERE' is meaningful
    """
    def __init__(self, in_channels, kernel_size=5):
        super(SpatialAttention, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            out: (batch, channels, height, width)
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        combined = torch.cat([avg_out, max_out], dim=1)
        
        attention = self.conv(combined)
        attention = self.sigmoid(attention)
        
        return x * attention
    
    
class SelfAttentionConv(nn.Module):
    """
    Self-Attention using 1x1 convolutions (Non-Local block)
    Works on 2D feature maps from CNNs
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super(SelfAttentionConv, self).__init__()
        
        # Reduce channels for spatial learning
        self.inter_channels = in_channels // reduction_ratio
        
        # Q, K, V using 1x1 convolutions
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        
        # Output projection, exapnding back the reduced channels
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        
        # Learnable scaling parameter (optional)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            out: (batch, channels, height, width)
        """
        batch, channels, height, width = x.size()
        
        Q = self.query_conv(x)   # (batch, inter_channels, H, W)
        K = self.key_conv(x)     # (batch, inter_channels, H, W)
        V = self.value_conv(x)   # (batch, inter_channels, H, W)
        
        # Reshape for matrix multiplication: (batch, inter_channels, H*W)
        Q = Q.view(batch, self.inter_channels, -1)
        K = K.view(batch, self.inter_channels, -1)
        V = V.view(batch, self.inter_channels, -1)
        
        # Transpose K for dot product
        K = K.permute(0, 2, 1)  # (batch, H*W, inter_channels)
        
        # Attention scores
        attention = torch.bmm(K, Q)  # (batch, H*W, H*W)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        attention = attention.permute(0, 2, 1) # (batch, H*W, H*W)
        out = torch.bmm(V, attention)  # (batch, inter_channels, H*W)
        
        # Reshape back to spatial dimensions
        out = out.view(batch, self.inter_channels, height, width)
        
        # Project back to original channels
        out = self.out_conv(out)  # (batch, channels, H, W)
        
        # Residual connection with learnable weight; optionally sacling by gamma factor
        out = self.gamma * out + x
        
        return out
    
    
class CrossAttention(nn.Module):
    """
    Note: This is private and experimental. 
    Neither allowed to embed into training, nor allowed to replicate.
    Copyright protected.
    Author: Sujit Patel.
    https://github.com/sujitpatel2739/fusion_ecg

    Cross-Attention between GAF and MTF images
    GAF provides Query, MTF provides Key and Value
    
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super(CrossAttention, self).__init__()
        
        self.inter_channels = in_channels // reduction_ratio
        
        # Q from GAF
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        
        # K, V from MTF
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, gaf_features, mtf_features):
        """
        Args:
            gaf_features: (batch, channels, H, W) - Features from GAF CNN
            mtf_features: (batch, channels, H, W) - Features from MTF CNN
        Returns:
            out: (batch, channels, H, W) - Attended features
        """
        batch, channels, height, width = gaf_features.size()
        
        # Q from GAF, K & V from MTF
        Q = self.query_conv(gaf_features)   # (batch, inter_channels, H, W)
        K = self.key_conv(mtf_features)     # (batch, inter_channels, H, W)
        V = self.value_conv(mtf_features)   # (batch, inter_channels, H, W)
        
        # Reshape
        Q = Q.view(batch, self.inter_channels, -1)
        K = K.view(batch, self.inter_channels, -1)
        V = V.view(batch, self.inter_channels, -1)
        
        K = K.permute(0, 2, 1)  # (batch, H*W, inter_channels)
        
        # Attention
        attention = torch.bmm(K, Q)  # (batch, H*W, H*W)
        attention = F.softmax(attention, dim=-1)
        
        # Apply to values
        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = out.view(batch, self.inter_channels, height, width)
        out = self.out_conv(out)
        
        # Residual with GAF features
        out = self.gamma * out + gaf_features
        
        return out