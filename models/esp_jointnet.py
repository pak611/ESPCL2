"""
ESP-CL: Joint Embedding Contrastive Learning for Drug-Target Interaction
Model Architecture Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FixedPositionalEncoding3D(nn.Module):
    """
    Fixed (sinusoidal) 3D positional encoding for spatial features
    Extends 2D sinusoidal encoding to 3D volumes
    """
    def __init__(self, embedding_dim, max_len=1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W, D] spatial features
        Returns:
            x: [B, C, H, W, D] features with positional encoding added
        """
        B, C, H, W, D = x.shape
        device = x.device
        
        # Create position indices
        pos_h = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(2)  # [H, 1, 1]
        pos_w = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2)  # [1, W, 1]
        pos_d = torch.arange(D, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)  # [1, 1, D]
        
        # Compute dimension-wise encodings
        div_term = torch.exp(torch.arange(0, C, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / C))
        
        pe = torch.zeros(C, H, W, D, device=device)
        
        # Alternate sine and cosine for each dimension
        # Height dimension
        pe[0::6, :, :, :] = torch.sin(pos_h * div_term[0::3].unsqueeze(1).unsqueeze(2).unsqueeze(3))
        pe[1::6, :, :, :] = torch.cos(pos_h * div_term[0::3].unsqueeze(1).unsqueeze(2).unsqueeze(3))
        # Width dimension
        pe[2::6, :, :, :] = torch.sin(pos_w * div_term[1::3].unsqueeze(0).unsqueeze(2).unsqueeze(3))
        pe[3::6, :, :, :] = torch.cos(pos_w * div_term[1::3].unsqueeze(0).unsqueeze(2).unsqueeze(3))
        # Depth dimension
        pe[4::6, :, :, :] = torch.sin(pos_d * div_term[2::3].unsqueeze(0).unsqueeze(1).unsqueeze(3))
        pe[5::6, :, :, :] = torch.cos(pos_d * div_term[2::3].unsqueeze(0).unsqueeze(1).unsqueeze(3))
        
        pe = pe.unsqueeze(0)  # [1, C, H, W, D]
        return x + pe


class LearnedPositionalEncoding3D(nn.Module):
    """
    Learned 3D positional encoding as parameter
    More flexible than fixed encoding
    """
    def __init__(self, embedding_dim, max_h=16, max_w=16, max_d=16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_h = max_h
        self.max_w = max_w
        self.max_d = max_d
        
        # Learnable positional embedding parameters
        self.pos_embed = nn.Parameter(torch.zeros(1, embedding_dim, max_h, max_w, max_d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W, D] spatial features
        Returns:
            x: [B, C, H, W, D] features with positional encoding added
        """
        B, C, H, W, D = x.shape
        
        # Handle variable spatial sizes by interpolating
        if H != self.max_h or W != self.max_w or D != self.max_d:
            pos_embed = F.interpolate(
                self.pos_embed, 
                size=(H, W, D), 
                mode='trilinear', 
                align_corners=False
            )
        else:
            pos_embed = self.pos_embed
            
        return x + pos_embed


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
                        continue
                    
                    # Randomly place patch
                    p_h = torch.randint(0, n_patches_h, (1,)).item()
                    p_w = torch.randint(0, n_patches_w, (1,)).item()
                    p_d = torch.randint(0, n_patches_d, (1,)).item()
                    
                    # Calculate voxel ranges
                    h_start = p_h * current_patch_size
                    h_end = min(h_start + current_patch_size, H)
                    w_start = p_w * current_patch_size
                    w_end = min(w_start + current_patch_size, W)
                    d_start = p_d * current_patch_size
                    d_end = min(d_start + current_patch_size, D)
                    
                    # Mask the patch across all channels
                    masked_x[b, :, h_start:h_end, w_start:w_end, d_start:d_end] = self.mask_value
                    
                    # Track voxels masked
                    patch_voxels = (h_end - h_start) * (w_end - w_start) * (d_end - d_start)
                    masked_voxels += patch_voxels
                    n_patches_masked += 1
        
        mask_info = {
            'channel_masked': n_channels_masked / B if B > 0 else 0,
            'spatial_masked': n_patches_masked / B if B > 0 else 0
        }
        
        return masked_x, mask_info


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for pocket-ligand interaction"""
    
    def __init__(self, embedding_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value):
        """
        Args:
            query: [B, seq_len, D] query embeddings
            key_value: [B, seq_len, D] key/value embeddings
        
        Returns:
            output: [B, seq_len, D] attended embeddings
        """
        # Cross-attention
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        
        # Residual connection and normalization
        output = self.norm(query + self.dropout(attn_output))
        
        return output


class PairwiseCrossAttention(nn.Module):
    """
    Pairwise attention mechanism inspired by AlphaFold
    Creates pair representations for (pocket_pos, ligand_pos) interactions
    """
    
    def __init__(self, channels=256, pair_channels=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.pair_channels = pair_channels
        
        # Initialize pair representation from single representations
        # z_ij = Linear(concat(pocket_i, ligand_j))
        self.pair_init = nn.Linear(channels * 2, pair_channels)
        
        # Pair attention: updates pairs based on other pairs
        # Simplified triangle update: pair(i,j) attends to pairs(i,*) and pairs(*,j)
        self.pair_attn = nn.MultiheadAttention(
            embed_dim=pair_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.pair_norm = nn.LayerNorm(pair_channels)
        self.pair_ffn = nn.Sequential(
            nn.Linear(pair_channels, pair_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(pair_channels * 2, pair_channels)
        )
        
        # Project pairs back to single representations
        # pocket_i' = aggregate(pair(i, *))
        # ligand_j' = aggregate(pair(*, j))
        self.to_pocket = nn.Linear(pair_channels, channels)
        self.to_ligand = nn.Linear(pair_channels, channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, pocket_seq, ligand_seq):
        """
        Args:
            pocket_seq: [B, N_p, C] pocket position embeddings
            ligand_seq: [B, N_l, C] ligand position embeddings
        
        Returns:
            pocket_out: [B, N_p, C] updated pocket embeddings
            ligand_out: [B, N_l, C] updated ligand embeddings
        """
        B, N_p, C = pocket_seq.shape
        N_l = ligand_seq.shape[1]
        
        # Create pair representations: [B, N_p, N_l, C*2]
        # Expand and concatenate: pocket_i with every ligand_j
        pocket_expanded = pocket_seq.unsqueeze(2).expand(-1, -1, N_l, -1)  # [B, N_p, N_l, C]
        ligand_expanded = ligand_seq.unsqueeze(1).expand(-1, N_p, -1, -1)  # [B, N_p, N_l, C]
        pairs = torch.cat([pocket_expanded, ligand_expanded], dim=-1)  # [B, N_p, N_l, C*2]
        
        # Initialize pair embeddings
        pair_emb = self.pair_init(pairs)  # [B, N_p, N_l, pair_C]
        
        # Flatten pairs for attention: [B, N_p*N_l, pair_C]
        pair_flat = pair_emb.view(B, N_p * N_l, self.pair_channels)
        
        # Self-attention over pairs (simplified triangle update)
        pair_attn_out, _ = self.pair_attn(pair_flat, pair_flat, pair_flat)
        pair_flat = self.pair_norm(pair_flat + self.dropout(pair_attn_out))
        
        # Feedforward
        pair_ffn_out = self.pair_ffn(pair_flat)
        pair_flat = pair_flat + self.dropout(pair_ffn_out)
        
        # Reshape back: [B, N_p, N_l, pair_C]
        pair_updated = pair_flat.view(B, N_p, N_l, self.pair_channels)
        
        # Aggregate pairs back to single representations
        # Pocket: average over ligand dimension
        pocket_from_pairs = pair_updated.mean(dim=2)  # [B, N_p, pair_C]
        pocket_out = self.to_pocket(pocket_from_pairs)  # [B, N_p, C]
        pocket_out = pocket_seq + pocket_out  # Residual
        
        # Ligand: average over pocket dimension
        ligand_from_pairs = pair_updated.mean(dim=1)  # [B, N_l, pair_C]
        ligand_out = self.to_ligand(ligand_from_pairs)  # [B, N_l, C]
        ligand_out = ligand_seq + ligand_out  # Residual
        
        return pocket_out, ligand_out


class SpatialCrossAttention(nn.Module):
    """Spatial cross-attention operating on 3D feature maps with multiple layers"""
    
    def __init__(self, channels=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        
        # Stack multiple transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(channels, num_heads, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, query_vol, key_value_vol):
        """
        Args:
            query_vol: [B, C, H, W, D] query 3D feature map
            key_value_vol: [B, C, H', W', D'] key/value 3D feature map
        
        Returns:
            output: [B, C, H, W, D] attended feature map
        """
        B, C, H, W, D = query_vol.shape
        _, _, H_kv, W_kv, D_kv = key_value_vol.shape
        
        # Reshape to sequence: [B, C, H*W*D] -> [B, H*W*D, C]
        query_seq = query_vol.view(B, C, H*W*D).permute(0, 2, 1)  # [B, N_q, C]
        kv_seq = key_value_vol.view(B, C, H_kv*W_kv*D_kv).permute(0, 2, 1)  # [B, N_kv, C]
        
        # Pass through all transformer layers
        for layer in self.layers:
            query_seq = layer(query_seq, kv_seq)
        
        # Reshape back to volume: [B, N, C] -> [B, C, H, W, D]
        output = query_seq.permute(0, 2, 1).view(B, C, H, W, D)
        
        return output


class TransformerLayer(nn.Module):
    """Single transformer layer with cross-attention and FFN"""
    
    def __init__(self, channels, num_heads, dropout):
        super().__init__()
        
        # Multi-head cross-attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm and feedforward
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),  # Expanded FFN (4x like TransBTS)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query_seq, kv_seq):
        """Single layer forward pass
        Args:
            query_seq: [B, N_q, C]
            kv_seq: [B, N_kv, C]
        Returns:
            query_seq: [B, N_q, C]
        """
        # Cross-attention
        attn_output, _ = self.multihead_attn(query_seq, kv_seq, kv_seq)
        
        # Residual and norm
        query_seq = self.norm1(query_seq + self.dropout(attn_output))
        
        # Feedforward network
        ffn_output = self.ffn(query_seq)
        query_seq = self.norm2(query_seq + ffn_output)
        
        return query_seq


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
        
        # Second conv: always stride=1, can use dilation
        padding2 = dilation
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=padding2, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out


class ESP_CNN_Encoder(nn.Module):
    """3D CNN for encoding voxelized ESP surfaces with multi-channel support"""
    
    def __init__(self, input_channels=8, embedding_dim=256, dropout=0.3, return_spatial=False):
        """
        Args:
            input_channels: Number of input channels (8 for ligands, 22 for pockets)
            embedding_dim: Dimension of output embedding
            dropout: Dropout rate for regularization
            return_spatial: If True, return spatial feature map instead of global pooled embedding
        """
        super().__init__()
        self.return_spatial = return_spatial
        
        # Initial convolution
        self.conv_initial = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Shallow architecture: 3 ResBlocks for faster training
        # 32³ → 16³ → 16³ → 16³
        self.block1 = ResidualBlock3D(32, 64, stride=2, dropout=dropout, dilation=1)   # Downsample: 32³→16³
        self.block2 = ResidualBlock3D(64, 128, stride=1, dropout=dropout, dilation=1)  # 16³→16³
        self.block3 = ResidualBlock3D(128, 256, stride=1, dropout=dropout, dilation=1) # 16³→16³, final=256
        
        # Conditional global pooling (only if not returning spatial features)
        if not return_spatial:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            self.projection = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W, D] voxelized ESP grid
        
        Returns:
            If return_spatial=False: [B, embedding_dim] L2-normalized embeddings
            If return_spatial=True: [B, 256, H/2, W/2, D/2] spatial feature map
        """
        x = self.conv_initial(x)         # [B, 32, H, W, D]
        x = self.block1(x)               # [B, 64, H/2, W/2, D/2]
        x = self.block2(x)               # [B, 128, H/2, W/2, D/2]
        x = self.block3(x)               # [B, 256, H/2, W/2, D/2]
        
        # Return spatial features for cross-attention
        if self.return_spatial:
            return x  # [B, 256, H/2, W/2, D/2]
        
        # Original global pooling path
        x = self.global_pool(x)          # [B, 256, 1, 1, 1]
        x = x.view(x.size(0), -1)        # [B, 256]
        x = self.projection(x)           # [B, embedding_dim]
        x = F.normalize(x, dim=-1)       # L2 normalize
        return x


class ESP_JointNet(nn.Module):
    """Supervised regression model for protein-ligand binding affinity prediction"""
    
    def __init__(self, embedding_dim=256, temperature=0.07, dropout=0.3, 
                 pocket_channels=22, ligand_channels=8, freeze_encoders=False,
                 use_cross_attention=False, use_pairwise_attention=False, num_heads=8,
                 channel_mask_ratio=0.0, spatial_mask_ratio=0.0, mask_patch_size=4):
        """
        Args:
            embedding_dim: Dimension of embeddings
            temperature: Not used in regression mode (kept for compatibility)
            dropout: Dropout rate
            pocket_channels: Number of input channels for pockets (22: ESP + 21 amino acids)
            ligand_channels: Number of input channels for ligands (8: ESP + 7 atom types)
            freeze_encoders: If True, freeze encoder weights (for fine-tuning pretrained models)
            use_cross_attention: If True, use spatial cross-attention between pocket and ligand
            use_pairwise_attention: If True, use pairwise attention (AlphaFold-style) instead of cross-attention
            num_heads: Number of attention heads for cross-attention
            channel_mask_ratio: Ratio of channels to mask during training (0-1)
            spatial_mask_ratio: Ratio of spatial patches to mask during training (0-1)
            mask_patch_size: Size of cubic patches for spatial masking
        """
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        self.use_pairwise_attention = use_pairwise_attention
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # Masking modules (applied before attention)
        self.use_masking = channel_mask_ratio > 0 or spatial_mask_ratio > 0
        if self.use_masking:
            self.pocket_masking = ChannelVoxelMasking(
                channel_mask_ratio=channel_mask_ratio,
                spatial_mask_ratio=spatial_mask_ratio,
                patch_size=mask_patch_size
            )
            self.ligand_masking = ChannelVoxelMasking(
                channel_mask_ratio=channel_mask_ratio,
                spatial_mask_ratio=spatial_mask_ratio,
                patch_size=mask_patch_size
            )
        
        # Encoders: return spatial features if using cross-attention, else return global embeddings
        self.pocket_encoder = ESP_CNN_Encoder(
            input_channels=pocket_channels, 
            embedding_dim=embedding_dim, 
            dropout=dropout,
            return_spatial=use_cross_attention or use_pairwise_attention
        )
        self.ligand_encoder = ESP_CNN_Encoder(
            input_channels=ligand_channels, 
            embedding_dim=embedding_dim, 
            dropout=dropout,
            return_spatial=use_cross_attention or use_pairwise_attention
        )
        
        # Optionally freeze encoders
        if freeze_encoders:
            self.freeze_encoders()
        
        if use_pairwise_attention:
            # Pairwise attention: aggressive downsampling for memory efficiency
            # Ligand: 12³ → 3³ (27 tokens)
            # Pocket: 24³ → 6³ (216 tokens)
            # Pairs: 216 × 27 = 5,832 pairs (much more manageable)
            self.pool_before_pairwise = nn.AvgPool3d(kernel_size=4, stride=4)
            
            # Pairwise cross-attention (AlphaFold-style)
            self.pairwise_attn = PairwiseCrossAttention(
                channels=256,
                pair_channels=64,  # Reduced from 128 for memory
                num_heads=4,
                dropout=0.1
            )
            
            # Post-attention pooling and projection for BOTH CL and regression (shared)
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            
            # Projection to embedding space for regression (after interaction)
            self.projection_p = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, embedding_dim)
            )
            self.projection_l = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, embedding_dim)
            )
            
            joint_dim = embedding_dim * 2
            
            # Regressor for pairwise mode
            self.regressor = nn.Sequential(
                nn.Linear(joint_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
        elif use_cross_attention:
            # Aggressive downsampling before attention to reduce memory
            # Ligand: 12³ → 3³ (27 tokens)
            # Pocket: 24³ → 6³ (216 tokens)
            # Attention: 27 × 216 = 5,832 token pairs (64× reduction from 373K!)
            self.pool_before_attn = nn.AvgPool3d(kernel_size=4, stride=4)
            
            # Linear projection before attention (transform CNN features)
            self.pre_attn_proj_p = nn.Linear(256, 256)
            self.pre_attn_proj_l = nn.Linear(256, 256)
            
            # 3D Positional encoding (learned is typically better for vision tasks)
            self.pos_encoding_p = LearnedPositionalEncoding3D(embedding_dim=256, max_h=6, max_w=6, max_d=6)
            self.pos_encoding_l = LearnedPositionalEncoding3D(embedding_dim=256, max_h=3, max_w=3, max_d=3)
            
            # Spatial cross-attention over downsampled 3D feature maps (4 layers deep)
            self.spatial_cross_attn_p = SpatialCrossAttention(channels=256, num_heads=num_heads, num_layers=4, dropout=0.1)
            self.spatial_cross_attn_l = SpatialCrossAttention(channels=256, num_heads=num_heads, num_layers=4, dropout=0.1)
            
            # Post-attention pooling and projection for BOTH CL and regression (shared)
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            
            # Projection to embedding space for regression (after interaction)
            self.projection_p = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, embedding_dim)
            )
            self.projection_l = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, embedding_dim)
            )
            
            joint_dim = embedding_dim * 2
            
            # Simplified MLP for cross-attention mode
            self.regressor = nn.Sequential(
                nn.Linear(joint_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
        else:
            # Original pathway without cross-attention
            joint_dim = embedding_dim * 2
            self.regressor = nn.Sequential(
                nn.Linear(joint_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )
    
    def forward(self, pocket_esp, ligand_esp):
        """
        Forward pass through both encoders and regression head
        
        Args:
            pocket_esp: [B, C, H_p, W_p, D_p] voxelized pocket ESP
            ligand_esp: [B, C, H_l, W_l, D_l] voxelized ligand ESP
        
        Returns:
            predictions: [B, 1] predicted binding affinity values
            mask_info: dict with masking statistics (if masking enabled)
        """
        mask_info = {}
        
        # Apply masking before encoding (only during training)
        if self.use_masking:
            pocket_esp, pocket_mask_info = self.pocket_masking(pocket_esp)
            ligand_esp, ligand_mask_info = self.ligand_masking(ligand_esp)
            mask_info.update({
                'pocket_channels_masked': pocket_mask_info['channel_masked'],
                'pocket_patches_masked': pocket_mask_info['spatial_masked'],
                'ligand_channels_masked': ligand_mask_info['channel_masked'],
                'ligand_patches_masked': ligand_mask_info['spatial_masked']
            })
        
        if self.use_pairwise_attention:
            # Get spatial feature maps: [B, 256, H/2, W/2, D/2]
            feat_pocket = self.pocket_encoder(pocket_esp)  # [B, 256, 24, 24, 24]
            feat_ligand = self.ligand_encoder(ligand_esp)  # [B, 256, 12, 12, 12]
            
            # Get actual batch size from features (handles DataParallel splitting)
            B = feat_pocket.size(0)
            
            # PAIRWISE ATTENTION FOR INTERACTION MODELING:
            # Aggressive downsampling for pairwise attention
            feat_pocket_ds = self.pool_before_pairwise(feat_pocket)  # [B, 256, H_p, W_p, D_p]
            feat_ligand_ds = self.pool_before_pairwise(feat_ligand)  # [B, 256, H_l, W_l, D_l]
            
            # Get actual spatial dimensions after pooling
            _, _, H_p, W_p, D_p = feat_pocket_ds.shape
            _, _, H_l, W_l, D_l = feat_ligand_ds.shape
            N_p = H_p * W_p * D_p  # Number of pocket tokens
            N_l = H_l * W_l * D_l  # Number of ligand tokens
            
            # Flatten to sequences
            pocket_seq = feat_pocket_ds.view(B, 256, -1).permute(0, 2, 1)  # [B, N_p, 256]
            ligand_seq = feat_ligand_ds.view(B, 256, -1).permute(0, 2, 1)  # [B, N_l, 256]
            
            # Pairwise attention: creates N_p × N_l pair representations
            pocket_seq, ligand_seq = self.pairwise_attn(pocket_seq, ligand_seq)
            
            # Reshape back to volumes
            feat_pocket_ds = pocket_seq.permute(0, 2, 1).view(B, 256, H_p, W_p, D_p)
            feat_ligand_ds = ligand_seq.permute(0, 2, 1).view(B, 256, H_l, W_l, D_l)
            
            # POST-ATTENTION EMBEDDINGS (interaction-aware):
            # Global pool after pairwise attention
            z_pooled_p = self.global_pool(feat_pocket_ds).view(B, -1)  # [B, 256]
            z_pooled_l = self.global_pool(feat_ligand_ds).view(B, -1)  # [B, 256]
            
            # SHARED PROJECTIONS: same embeddings for both CL and regression
            # This allows contrastive loss to directly optimize the features used for prediction
            z_pocket = self.projection_p(z_pooled_p)  # [B, embedding_dim]
            z_ligand = self.projection_l(z_pooled_l)  # [B, embedding_dim]
            
            # L2 normalize for contrastive learning
            z_pocket_cl = F.normalize(z_pocket, dim=-1)
            z_ligand_cl = F.normalize(z_ligand, dim=-1)
            
            # Concatenate for regression (using same embeddings as CL)
            z_combined = torch.cat([z_pocket, z_ligand], dim=-1)  # [B, embedding_dim * 2]
            predictions = self.regressor(z_combined)  # [B, 1]
            
            # Convert mask_info to tensors for DataParallel compatibility
            if mask_info is not None:
                mask_info_tensors = {
                    k: torch.tensor(v, device=predictions.device, dtype=torch.float32) 
                    for k, v in mask_info.items()
                }
            else:
                mask_info_tensors = None
            
            # Return predictions + contrastive embeddings + mask info
            return predictions, z_pocket_cl, z_ligand_cl, mask_info_tensors
        
        elif self.use_cross_attention:
            # Get spatial feature maps: [B, 256, H/2, W/2, D/2]
            feat_pocket = self.pocket_encoder(pocket_esp)  # [B, 256, 24, 24, 24]
            feat_ligand = self.ligand_encoder(ligand_esp)  # [B, 256, 12, 12, 12]
            
            # CROSS-ATTENTION FOR INTERACTION MODELING:
            # Aggressive downsampling before attention to reduce memory
            # Pocket: 24³ → 6³ (13,824 → 216 tokens)
            # Ligand: 12³ → 3³ (1,728 → 27 tokens)
            feat_pocket_ds = self.pool_before_attn(feat_pocket)  # [B, 256, 6, 6, 6]
            feat_ligand_ds = self.pool_before_attn(feat_ligand)  # [B, 256, 3, 3, 3]
            
            # Linear projection before attention to transform CNN features
            B, C = feat_pocket_ds.size(0), feat_pocket_ds.size(1)
            feat_pocket_proj = self.pre_attn_proj_p(feat_pocket_ds.permute(0, 2, 3, 4, 1).reshape(-1, C))
            feat_pocket_proj = feat_pocket_proj.reshape(B, *feat_pocket_ds.shape[2:], C).permute(0, 4, 1, 2, 3)
            feat_ligand_proj = self.pre_attn_proj_l(feat_ligand_ds.permute(0, 2, 3, 4, 1).reshape(-1, C))
            feat_ligand_proj = feat_ligand_proj.reshape(B, *feat_ligand_ds.shape[2:], C).permute(0, 4, 1, 2, 3)
            
            # Add 3D positional encodings (helps attention understand spatial relationships)
            feat_pocket_proj = self.pos_encoding_p(feat_pocket_proj)  # [B, 256, 6, 6, 6]
            feat_ligand_proj = self.pos_encoding_l(feat_ligand_proj)  # [B, 256, 3, 3, 3]
            
            # Spatial cross-attention (residuals handled internally via norm1/norm2)
            # Output shape = query shape (pocket queries pocket_proj, ligand queries ligand_proj)
            feat_p_ctx = self.spatial_cross_attn_p(feat_pocket_proj, feat_ligand_proj)  # [B, 256, 6, 6, 6]
            feat_l_ctx = self.spatial_cross_attn_l(feat_ligand_proj, feat_pocket_proj)  # [B, 256, 3, 3, 3]
            
            # POST-ATTENTION EMBEDDINGS (interaction-aware):
            # Global pool after cross-attention
            z_pooled_p = self.global_pool(feat_p_ctx).view(pocket_esp.size(0), -1)  # [B, 256]
            z_pooled_l = self.global_pool(feat_l_ctx).view(ligand_esp.size(0), -1)  # [B, 256]
            
            # SHARED PROJECTIONS: same embeddings for both CL and regression
            # This allows contrastive loss to directly optimize the features used for prediction
            z_pocket = self.projection_p(z_pooled_p)  # [B, embedding_dim]
            z_ligand = self.projection_l(z_pooled_l)  # [B, embedding_dim]
            
            # L2 normalize for contrastive learning
            z_pocket_cl = F.normalize(z_pocket, dim=-1)
            z_ligand_cl = F.normalize(z_ligand, dim=-1)
            
            # Concatenate for regression (using same embeddings as CL)
            z_combined = torch.cat([z_pocket, z_ligand], dim=-1)  # [B, embedding_dim * 2]
            
            # Predict binding affinity
            predictions = self.regressor(z_combined)  # [B, 1]
            
            # Convert mask_info to tensors for DataParallel compatibility
            if mask_info is not None:
                mask_info_tensors = {
                    k: torch.tensor(v, device=predictions.device, dtype=torch.float32) 
                    for k, v in mask_info.items()
                }
            else:
                mask_info_tensors = None
            
            # Return predictions + contrastive embeddings + mask info
            return predictions, z_pocket_cl, z_ligand_cl, mask_info_tensors
        else:
            # Original pathway: encode to global embeddings directly
            z_pocket = self.pocket_encoder(pocket_esp)  # [B, embedding_dim]
            z_ligand = self.ligand_encoder(ligand_esp)  # [B, embedding_dim]
            z_combined = torch.cat([z_pocket, z_ligand], dim=-1)  # [B, embedding_dim * 2]
            
            # Predict binding affinity
            predictions = self.regressor(z_combined)  # [B, 1]
            
            # No pre-attention embeddings or mask info in non-attention mode
            return predictions, None, None, None
    
    def regression_loss(self, predictions, labels):
        """
        L2 (MSE) Loss for regression
        
        Args:
            predictions: [B, 1] predicted binding affinity values
            labels: [B, 1] or [B] ground truth binding affinity values
        
        Returns:
            loss: scalar MSE loss
        """
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        
        return F.mse_loss(predictions, labels)
    
    def contrastive_loss(self, z_pocket_v1, z_ligand_v1, z_pocket_v2, z_ligand_v2, temperature=0.1):
        """
        Intra-modal consistency loss between two views
        
        View-invariant representation learning:
        - Pocket consistency: P1 ≈ P2 (same pocket, different augmentations)
        - Ligand consistency: L1 ≈ L2 (same ligand, different augmentations)
        
        This enforces augmentation invariance without forcing pockets and ligands
        to have similar representations (which would harm binding specificity).
        
        For each batch, we have:
        - View 1: z_pocket_v1, z_ligand_v1 (both [B, embedding_dim])
        - View 2: z_pocket_v2, z_ligand_v2 (both [B, embedding_dim])
        
        Positive pairs: same molecule across views (P1↔P2, L1↔L2)
        Negative pairs: different molecules in the batch
        
        Args:
            z_pocket_v1: [B, embedding_dim] pocket embeddings from view 1
            z_ligand_v1: [B, embedding_dim] ligand embeddings from view 1
            z_pocket_v2: [B, embedding_dim] pocket embeddings from view 2
            z_ligand_v2: [B, embedding_dim] ligand embeddings from view 2
            temperature: temperature scaling parameter (default 0.1)
        
        Returns:
            loss_contrastive: scalar contrastive loss
        """
        batch_size = z_pocket_v1.size(0)
        device = z_pocket_v1.device
        
        # Ensure embeddings are L2-normalized
        z_pocket_v1 = F.normalize(z_pocket_v1, dim=-1)
        z_ligand_v1 = F.normalize(z_ligand_v1, dim=-1)
        z_pocket_v2 = F.normalize(z_pocket_v2, dim=-1)
        z_ligand_v2 = F.normalize(z_ligand_v2, dim=-1)
        
        # Labels: diagonal elements are positives (same sample across views)
        labels = torch.arange(batch_size, device=device)
        
        # Intra-modal contrastive losses
        # Pocket consistency: P1 ≈ P2 (bidirectional)
        sim_p1_p2 = torch.matmul(z_pocket_v1, z_pocket_v2.T) / temperature  # [B, B]
        sim_p2_p1 = torch.matmul(z_pocket_v2, z_pocket_v1.T) / temperature  # [B, B]
        
        # Ligand consistency: L1 ≈ L2 (bidirectional)
        sim_l1_l2 = torch.matmul(z_ligand_v1, z_ligand_v2.T) / temperature  # [B, B]
        sim_l2_l1 = torch.matmul(z_ligand_v2, z_ligand_v1.T) / temperature  # [B, B]
        
        # InfoNCE losses
        loss_p1_p2 = F.cross_entropy(sim_p1_p2, labels)
        loss_p2_p1 = F.cross_entropy(sim_p2_p1, labels)
        loss_l1_l2 = F.cross_entropy(sim_l1_l2, labels)
        loss_l2_l1 = F.cross_entropy(sim_l2_l1, labels)
        
        # Average all losses
        loss_contrastive = (loss_p1_p2 + loss_p2_p1 + loss_l1_l2 + loss_l2_l1) / 4
        
        return loss_contrastive
    
    def contrastive_loss_with_negatives(self, z_pocket_v1, z_ligand_v1, z_pocket_v2, z_ligand_v2,
                                       z_pocket_neg, z_ligand_neg, neg_mask, temperature=0.1):
        """
        InfoNCE with augmented views as positives + ONLY chemical corruptions as negatives
        
        Positive pairs: (P_i_view1, P_i_view2) - same pocket, different augmentations
        Negative pairs: 
            - Chemical ONLY: corrupted versions (ESP flipped, channels zeroed, etc.)
            - NO in-batch negatives (avoids false negatives from similar molecules)
        
        Args:
            z_pocket_v1: [B, D] clean view 1
            z_ligand_v1: [B, D] clean view 1
            z_pocket_v2: [B, D] clean view 2
            z_ligand_v2: [B, D] clean view 2
            z_pocket_neg: [B, D] corrupted pockets
            z_ligand_neg: [B, D] corrupted ligands
            neg_mask: [B] boolean mask (True = corrupted)
            temperature: InfoNCE temperature
        
        Returns:
            loss: scalar contrastive loss
        """
        B = z_pocket_v1.size(0)
        device = z_pocket_v1.device
        
        # Normalize all embeddings
        z_pocket_v1 = F.normalize(z_pocket_v1, dim=-1)
        z_ligand_v1 = F.normalize(z_ligand_v1, dim=-1)
        z_pocket_v2 = F.normalize(z_pocket_v2, dim=-1)
        z_ligand_v2 = F.normalize(z_ligand_v2, dim=-1)
        z_pocket_neg = F.normalize(z_pocket_neg, dim=-1)
        z_ligand_neg = F.normalize(z_ligand_neg, dim=-1)
        
        # Filter to only corrupted samples (true chemical negatives)
        z_pocket_neg_filtered = z_pocket_neg[neg_mask]  # [N_corrupted, D] where N_corrupted = ~5
        z_ligand_neg_filtered = z_ligand_neg[neg_mask]  # [N_corrupted, D]
        
        if z_pocket_neg_filtered.size(0) == 0:
            # No corrupted samples in batch, return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Pocket contrastive loss (bidirectional)
        # Positives: diagonal ONLY (same pocket across views)
        # Negatives: corrupted pockets ONLY (no in-batch negatives)
        sim_p1_p2_pos = (z_pocket_v1 * z_pocket_v2).sum(dim=-1, keepdim=True) / temperature  # [B, 1] diagonal only
        sim_p1_neg = torch.matmul(z_pocket_v1, z_pocket_neg_filtered.T) / temperature  # [B, N_corrupted]
        
        # Concatenate positive + chemical negatives
        logits_p = torch.cat([sim_p1_p2_pos, sim_p1_neg], dim=1)  # [B, 1+N_corrupted]
        labels_p = torch.zeros(B, dtype=torch.long, device=device)  # Positive is always at index 0
        
        loss_p1 = F.cross_entropy(logits_p, labels_p)
        
        # Reverse direction
        sim_p2_p1_pos = (z_pocket_v2 * z_pocket_v1).sum(dim=-1, keepdim=True) / temperature
        sim_p2_neg = torch.matmul(z_pocket_v2, z_pocket_neg_filtered.T) / temperature
        logits_p2 = torch.cat([sim_p2_p1_pos, sim_p2_neg], dim=1)
        loss_p2 = F.cross_entropy(logits_p2, labels_p)
        
        # Ligand contrastive loss (same structure)
        sim_l1_l2_pos = (z_ligand_v1 * z_ligand_v2).sum(dim=-1, keepdim=True) / temperature
        sim_l1_neg = torch.matmul(z_ligand_v1, z_ligand_neg_filtered.T) / temperature
        logits_l = torch.cat([sim_l1_l2_pos, sim_l1_neg], dim=1)
        loss_l1 = F.cross_entropy(logits_l, labels_p)
        
        sim_l2_l1_pos = (z_ligand_v2 * z_ligand_v1).sum(dim=-1, keepdim=True) / temperature
        sim_l2_neg = torch.matmul(z_ligand_v2, z_ligand_neg_filtered.T) / temperature
        logits_l2 = torch.cat([sim_l2_l1_pos, sim_l2_neg], dim=1)
        loss_l2 = F.cross_entropy(logits_l2, labels_p)
        
        # Average all losses
        loss = (loss_p1 + loss_p2 + loss_l1 + loss_l2) / 4
        
        return loss
    
    def soft_supervised_contrastive_loss(self, z_pocket, z_ligand, labels, temperature=1.5):
        """
        Soft supervised contrastive loss: match embedding similarity to label similarity
        
        This aligns CL with regression objective by teaching the model that complexes
        with similar pKd should have similar embeddings.
        
        Args:
            z_pocket: [B, D] pocket embeddings (post-attention pooled)
            z_ligand: [B, D] ligand embeddings (post-attention pooled)
            labels: [B] or [B, 1] pKd values
            temperature: controls how quickly similarity decays with pKd difference
                        Higher = more lenient (1.5-2.0 typical)
        
        Returns:
            loss: scalar MSE loss between embedding and target similarities
        """
        B = z_pocket.size(0)
        
        # Flatten labels if needed
        if labels.dim() == 2:
            labels = labels.squeeze(-1)
        
        # Combine pocket and ligand embeddings for each complex
        z_combined = torch.cat([z_pocket, z_ligand], dim=-1)  # [B, D*2]
        z_norm = F.normalize(z_combined, dim=-1)
        
        # Compute embedding similarities [B, B]
        embedding_sim = torch.matmul(z_norm, z_norm.T)
        
        # Compute target similarities from pKd labels [B, B]
        label_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
        target_sim = torch.exp(-label_diff / temperature)
        
        # MSE loss between embedding similarity and target similarity
        loss = F.mse_loss(embedding_sim, target_sim)
        
        return loss
    
    def compute_similarity(self, z_pocket, z_ligand):
        """
        Compute cosine similarity between pocket and ligand embeddings
        
        Args:
            z_pocket: [B, embedding_dim] pocket embeddings
            z_ligand: [B, embedding_dim] ligand embeddings
        
        Returns:
            similarity: [B, B] similarity matrix
        """
        # Embeddings are already L2-normalized, so dot product = cosine similarity
        return torch.matmul(z_pocket, z_ligand.T)
    
    def freeze_encoders(self):
        """Freeze encoder weights for fine-tuning"""
        for param in self.pocket_encoder.parameters():
            param.requires_grad = False
        for param in self.ligand_encoder.parameters():
            param.requires_grad = False
        print("Encoder weights frozen. Only regressor will be trained.")
    
    def unfreeze_encoders(self):
        """Unfreeze encoder weights"""
        for param in self.pocket_encoder.parameters():
            param.requires_grad = True
        for param in self.ligand_encoder.parameters():
            param.requires_grad = True
        print("Encoder weights unfrozen.")


class ESP_BindingAffinityPredictor(nn.Module):
    """
    Binding affinity prediction model using pre-trained ESP encoders
    """
    
    def __init__(self, pretrained_model, freeze_encoders=True):
        super().__init__()
        
        self.pocket_encoder = pretrained_model.pocket_encoder
        self.ligand_encoder = pretrained_model.ligand_encoder
        
        if freeze_encoders:
            for param in self.pocket_encoder.parameters():
                param.requires_grad = False
            for param in self.ligand_encoder.parameters():
                param.requires_grad = False
        
        embedding_dim = pretrained_model.embedding_dim
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, pocket_esp, ligand_esp):
        """
        Predict binding affinity
        
        Args:
            pocket_esp: [B, 1, H_p, W_p, D_p] voxelized pocket ESP
            ligand_esp: [B, 1, H_l, W_l, D_l] voxelized ligand ESP
        
        Returns:
            affinity: [B, 1] predicted binding affinity
        """
        z_pocket = self.pocket_encoder(pocket_esp)
        z_ligand = self.ligand_encoder(ligand_esp)
        
        # Concatenate embeddings
        z_combined = torch.cat([z_pocket, z_ligand], dim=-1)
        
        # Predict affinity
        affinity = self.regressor(z_combined)
        return affinity


def load_pretrained_model(checkpoint_path, device='cuda'):
    """
    Load a pre-trained ESP-JointNet model
    
    Args:
        checkpoint_path: path to checkpoint file
        device: device to load model on
    
    Returns:
        model: loaded ESP_JointNet model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = ESP_JointNet(
        embedding_dim=checkpoint.get('embedding_dim', 512),
        temperature=checkpoint.get('temperature', 0.07)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing ESP-JointNet model...")
    
    # Create model with cross-attention enabled
    model = ESP_JointNet(
        embedding_dim=256,
        temperature=0.07,
        pocket_channels=10,
        ligand_channels=9,
        use_cross_attention=True
    )
    model.train()
    
    # Create dummy inputs with correct number of channels
    batch_size = 4
    pocket_esp = torch.randn(batch_size, 10, 32, 32, 32)  # 10 channels for pocket
    ligand_esp = torch.randn(batch_size, 9, 32, 32, 32)   # 9 channels for ligand
    
    # Forward pass - returns 4 values
    predictions, z_pocket_cl, z_ligand_cl, mask_info = model(pocket_esp, ligand_esp)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Pocket CL embeddings shape: {z_pocket_cl.shape}")
    print(f"Ligand CL embeddings shape: {z_ligand_cl.shape}")
    
    # Test regression loss
    labels = torch.randn(batch_size, 1)
    reg_loss = model.regression_loss(predictions, labels)
    print(f"Regression loss: {reg_loss.item():.4f}")
    
    # Test contrastive loss (requires two views)
    # Simulate second view with different augmentation
    predictions2, z_pocket_cl2, z_ligand_cl2, _ = model(pocket_esp, ligand_esp)
    cl_loss = model.contrastive_loss(z_pocket_cl, z_ligand_cl, z_pocket_cl2, z_ligand_cl2)
    print(f"Contrastive loss: {cl_loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nModel architecture test completed successfully!")
