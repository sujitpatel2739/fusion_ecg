import torch
import torch.nn as nn


class JointFusion(nn.Module):
    """
    Joint fusion: Combine features from multiple encoders
    """
    def __init__(self,in_channels=3, out_channels=128, encoder_1d=None, encoder_2d=None, num_classes=5):
        super(JointFusion, self).__init__()
        
        # Encoder for 1D signals
        if encoder_1d:
            self.encoder_1d = encoder_1d
        else:
            self.encoder_1d = nn.Sequential(
                nn.Conv1d(in_channels, in_channels*11, kernel_size=5, padding=2),
                nn.BatchNorm1d(in_channels*11),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(in_channels*11, in_channels*22, kernel_size=5, padding=2),
                nn.BatchNorm1d(in_channels*22),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(in_channels*22, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1)
            )
        
        # Encoder for GAF images
        if encoder_2d:
            self.encoder_2d = encoder_2d
        else:
            self.encoder_2d = nn.Sequential(
                nn.Conv2d(in_channels, in_channels*11, kernel_size=5, padding=2),
                nn.BatchNorm2d(in_channels*11),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels*11, in_channels*22, kernel_size=5, padding=2),
                nn.BatchNorm2d(in_channels*22),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels*22, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(out_channels*2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Shared classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (signal, image)
            signal: (batch, 3, 1000)
            image: (batch, 3, 224, 224)
        """
        signal, image = x
        # Extract features from each modality
        feat_1d = self.encoder_1d(signal)  # (batch, 128, 1)
        feat_1d = feat_1d.squeeze(-1)          # (batch, 128)
        
        feat_2d = self.encoder_2d(image)  # (batch, 128, 1, 1)
        feat_2d = feat_2d.view(feat_2d.size(0), -1)  # (batch, 128)
        
        # Concatenate features
        fused = torch.cat([feat_1d, feat_2d], dim=1)  # (batch, 256)
        
        # Fusion layer
        fused = self.fusion(fused)  # (batch, 256)
        
        # Classification
        logits = self.classifier(fused)
        return logits
    
    
