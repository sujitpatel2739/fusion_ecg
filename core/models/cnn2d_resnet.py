import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Basic Residual Block with skip connection
    Used in ResNet-18/34
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity or projection)
        self.downsample = downsample
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x  # Save input for skip connection
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)  # Project to match dimensions
        
        out += identity  # Element-wise addition (the key innovation!)
        out = self.relu(out)
        
        return out
    
class ResNet18(nn.Module):
    """
    ResNet-18 for ECG image classification
    """
    def __init__(self, in_channels, num_classes=5, base_filters=16):
        super(ResNet18, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(base_filters, base_filters, 2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters*2, 2, stride=2)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, 2, stride=2)
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(base_filters*8, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a composite layer with (num_blocks) residual blocks"""
        downsample = None
        
        # If dimensions change, we need projection for skip connection
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        # First block may have stride > 1 (downsampling)
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks have stride=1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, downsample=None))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            logits: (batch, num_classes)
        """
        # Initial conv
        out = self.conv1(x)      # (batch, 16, 112, 112)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)    # (batch, 16, 56, 56)
        
        # Residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global pooling and FC
        out = self.avgpool(out)    # (batch, 128, 1, 1)
        out = torch.flatten(out, 1)  # (batch, 128)
        logits = self.fc(out)         # (batch, 5)
        
        return logits