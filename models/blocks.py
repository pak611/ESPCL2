"""
Basic building blocks for 3D neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """3D Residual Block with optional downsampling and dilation"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3, dilation=1):
        super().__init__()
        
        # First conv: can have stride for downsampling
        # Padding calculation: (kernel_size - 1) * dilation / 2
        padding1 = dilation if stride == 1 else 1
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=padding1, dilation=dilation if stride == 1 else 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout)
        
        # Second conv: always stride=1
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ChannelVoxelMasking(nn.Module):
    """
    Multi-scale Channel and Spatial Masking for 3D voxel grids
    Uses mixture of patch scales to remove meaningful functional regions
    """
    
    def __init__(self, channel_mask_ratio=0.0, spatial_mask_ratio=0.0, 
                 patch_size=4, mask_value=0.0):
        """
        Args:
            channel_mask_ratio: Probability of masking entire channels (0-1)
            spatial_mask_ratio: Probability of masking spatial patches (0-1)
            patch_size: Base size for cubic patches (used for small masks)
            mask_value: Value to use for masked regions
        """
        super().__init__()
        self.channel_mask_ratio = channel_mask_ratio
        self.spatial_mask_ratio = spatial_mask_ratio
        self.patch_size = patch_size
        self.mask_value = mask_value
        
        # Multi-scale patch sizes: small (local), medium (functional regions), large (subpockets)
        # Distribution: 50% small, 35% medium, 15% large
        self.patch_scales = [
            (patch_size, 0.50),      # Small: 4³ - local corruption
            (patch_size * 2, 0.35),  # Medium: 8³ - functional regions
            (patch_size * 3, 0.15),  # Large: 12³ - subpocket blocks
        ]
    
    def forward(self, x):
        """
        Apply multi-scale channel and spatial masking
        
        Args:
            x: [B, C, H, W, D] input feature maps
        
        Returns:
            masked_x: [B, C, H, W, D] masked features
            mask_info: dict with masking statistics
        """
        if not self.training or (self.channel_mask_ratio == 0 and self.spatial_mask_ratio == 0):
            return x, {'channel_masked': 0, 'spatial_masked': 0}
        
        B, C, H, W, D = x.shape
        masked_x = x.clone()
        
        # Channel masking: randomly zero out entire channels (including ESP)
        n_channels_masked = 0
        if self.channel_mask_ratio > 0:
            # Mask random channels from all available channels (0 to C-1)
            n_mask = int(C * self.channel_mask_ratio)
            
            for b in range(B):
                if n_mask > 0:
                    # Random channels to mask (can include channel 0 now)
                    mask_channels = torch.randperm(C)[:n_mask]
                    masked_x[b, mask_channels, :, :, :] = self.mask_value
                    n_channels_masked += n_mask
        
        # Multi-scale spatial patch masking
        n_patches_masked = 0
        if self.spatial_mask_ratio > 0:
            # Determine total voxel budget to mask
            total_voxels = H * W * D
            target_masked_voxels = int(total_voxels * self.spatial_mask_ratio)
            
            for b in range(B):
                masked_voxels = 0
                attempts = 0
                max_attempts = 100  # Prevent infinite loop
                
                while masked_voxels < target_masked_voxels and attempts < max_attempts:
                    attempts += 1
                    
                    # Sample patch scale based on distribution
                    rand = torch.rand(1).item()
                    cumulative = 0.0
                    current_patch_size = self.patch_size
                    
                    for scale_size, prob in self.patch_scales:
                        cumulative += prob
                        if rand < cumulative:
                            current_patch_size = scale_size
                            break
                    
                    # Calculate patch grid for this scale
                    n_patches_h = H // current_patch_size
                    n_patches_w = W // current_patch_size
                    n_patches_d = D // current_patch_size
                    
                    if n_patches_h == 0 or n_patches_w == 0 or n_patches_d == 0:
                        continue  # Skip if patch is too large for grid
                    
                    # Randomly select a patch to mask
                    ph = torch.randint(0, n_patches_h, (1,)).item()
                    pw = torch.randint(0, n_patches_w, (1,)).item()
                    pd = torch.randint(0, n_patches_d, (1,)).item()
                    
                    # Calculate patch boundaries
                    h_start = ph * current_patch_size
                    h_end = min(h_start + current_patch_size, H)
                    w_start = pw * current_patch_size
                    w_end = min(w_start + current_patch_size, W)
                    d_start = pd * current_patch_size
                    d_end = min(d_start + current_patch_size, D)
                    
                    # Mask the patch
                    masked_x[b, :, h_start:h_end, w_start:w_end, d_start:d_end] = self.mask_value
                    masked_voxels += (h_end - h_start) * (w_end - w_start) * (d_end - d_start)
                    n_patches_masked += 1
        
        return masked_x, {
            'channel_masked': n_channels_masked, 
            'spatial_masked': n_patches_masked
        }
