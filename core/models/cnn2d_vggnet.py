import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    """
    VGG-style block: multiple Conv layers + ReLU + pooling
    """
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        
        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels if i==0 else out_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1,
                        ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # Add MaxPool2d at the end
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features_block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.features_block(x)

class VGGNet(nn.Module):
    """
    VGG-style network for ECG images
    Uses VGG blocks with batch normalization
    """
    def __init__(self, in_channels, out_channels, num_classes=5, dropout=0.5):
        super(VGGNet, self).__init__()
        
        self.block1 = VGGBlock(in_channels, in_channels*6, 2)
        self.block2 = VGGBlock(in_channels*6, in_channels*12, 2)
        self.block3 = VGGBlock(in_channels*12, in_channels*24, 3)
        self.block4 = VGGBlock(in_channels*24, in_channels*48, 3)
        self.block5 = VGGBlock(in_channels*48, out_channels, 3)
        
        self.adaptmaxpool1 = nn.AdaptiveMaxPool2d((3, 3))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(out_channels * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            logits: (batch, num_classes)
        """
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        
        # TODO: Adaptive pooling and flatten
        out = self.adaptmaxpool1(out)
        out = torch.flatten(out, 1)
        
        logits = self.classifier(out)
        return logits