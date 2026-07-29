"""Core CNN architecture for wafer defect detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompleteCNN(nn.Module):
    """Convolutional Neural Network for multi-label classification.
    
    Processes 52x52 pixel single-channel wafer map data. Designed to isolate
    and classify simultaneous hardware failure topologies. It includes
    quantization stubs to support Post-Training Static Quantization.
    """
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4) 

        self.fc1 = nn.Linear(128 * 13 * 13, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 1, 52, 52) or (B, 52, 52).
            
        Returns:
            Output logits of shape (B, num_classes).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.quant(x)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = F.relu(self.bn3(self.conv3(x)))             
        x = F.relu(self.bn4(self.conv4(x)))           

        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))          
        x = self.dropout(F.relu(self.fc2(x)))           
        x = self.fc3(x)                                 

        x = self.dequant(x)
        return x

    @torch.jit.export
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent feature vectors before the final classification layer.
        
        This is useful for calculating Maximum Mean Discrepancy (MMD) during
        data drift monitoring.
        
        Args:
            x: Input tensor of shape (B, 1, 52, 52) or (B, 52, 52).
            
        Returns:
            Latent feature vector of shape (B, 512).
        """
        features, _ = self.forward_with_features(x)
        return features

    @torch.jit.export
    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass, returning both features and logits.
        
        Args:
            x: Input tensor of shape (B, 1, 52, 52) or (B, 52, 52).
            
        Returns:
            A tuple of (features, logits).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.quant(x)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1)
        features = self.dequant(F.relu(self.fc1(x)))
        
        out = self.dropout(F.relu(self.fc1(x)))
        out = self.dropout(F.relu(self.fc2(out)))
        logits = self.fc3(out)
        logits = self.dequant(logits)
        
        return features, logits