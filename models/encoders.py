"""
CNN Encoders for ESP-based protein-ligand modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualBlock3D


class ESP_CNN_Encoder(nn.Module):
    """
    CNN encoder for ESP voxel grids
    Progressive downsampling: 32³ → 16³ → 8³ → 4³ → global embedding
    """
    
    def __init__(self, input_channels=8, embedding_dim=256, dropout=0.3, return_spatial=False):
        """
        Args:
            input_channels: Number of input channels (e.g., 8 for ESP + features)
            embedding_dim: Final embedding dimension
            dropout: Dropout rate
            return_spatial: If True, return spatial features instead of global pooling
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.return_spatial = return_spatial
        
        # Initial projection: expand channels
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )
        
        # Encoder blocks with progressive downsampling
        # 32³ → 16³ (stride 2)
        self.block1 = ResidualBlock3D(32, 64, stride=2, dropout=dropout)
        
        # 16³ → 8³ (stride 2)
        self.block2 = ResidualBlock3D(64, 128, stride=2, dropout=dropout)
        
        # 8³ → 4³ (stride 2)
        self.block3 = ResidualBlock3D(128, embedding_dim, stride=2, dropout=dropout)
        
        # Global pooling and projection (only used when return_spatial=False)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C_in, H, W, D] input voxel grid
        
        Returns:
            If return_spatial=False: [B, embedding_dim] global embedding
            If return_spatial=True: [B, embedding_dim, H', W', D'] spatial features
        """
        # Stem
        x = self.stem(x)  # [B, 32, H, W, D]
        
        # Encoder blocks
        x = self.block1(x)  # [B, 64, H/2, W/2, D/2]
        x = self.block2(x)  # [B, 128, H/4, W/4, D/4]
        x = self.block3(x)  # [B, embedding_dim, H/8, W/8, D/8]
        
        if self.return_spatial:
            # Return spatial feature maps for cross-attention
            return x  # [B, embedding_dim, H/8, W/8, D/8]
        else:
            # Global pooling path
            x = self.global_pool(x)          # [B, embedding_dim, 1, 1, 1]
            x = x.view(x.size(0), -1)        # [B, embedding_dim]
            x = self.projection(x)           # [B, embedding_dim]
            x = F.normalize(x, dim=-1)       # L2 normalize
            return x
