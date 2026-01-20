"""
ESP-CL: Joint Embedding Contrastive Learning for Drug-Target Interaction
Main model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import ESP_CNN_Encoder


class ESP_JointNet(nn.Module):
    """DrugCLIP-style contrastive learning model with CNN encoders"""
    
    def __init__(self, embedding_dim=256, dropout=0.3, 
                 pocket_channels=22, ligand_channels=8, freeze_encoders=False,
                 use_cross_attention=False, num_attn_heads=8, num_attn_layers=2):
        """
        Args:
            embedding_dim: Dimension of raw embeddings from encoders (default 256)
            dropout: Dropout rate
            pocket_channels: Number of input channels for pockets (22: ESP + 21 amino acids)
            ligand_channels: Number of input channels for ligands (8: ESP + 7 atom types)
            freeze_encoders: If True, freeze encoder weights (for fine-tuning pretrained models)
            use_cross_attention: If True, add cross-attention between pocket and ligand features
            num_attn_heads: Number of attention heads for cross-attention
            num_attn_layers: Number of cross-attention layers
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_cross_attention = use_cross_attention
        
        # CNN Encoders: can return either global embeddings or spatial features
        self.pocket_encoder = ESP_CNN_Encoder(
            input_channels=pocket_channels, 
            embedding_dim=embedding_dim, 
            dropout=dropout,
            return_spatial=use_cross_attention  # Return spatial features if using cross-attention
        )
        self.ligand_encoder = ESP_CNN_Encoder(
            input_channels=ligand_channels, 
            embedding_dim=embedding_dim, 
            dropout=dropout,
            return_spatial=use_cross_attention  # Return spatial features if using cross-attention
        )
        
        # Optionally freeze encoders
        if freeze_encoders:
            for param in self.pocket_encoder.parameters():
                param.requires_grad = False
            for param in self.ligand_encoder.parameters():
                param.requires_grad = False
        
        # Projection heads for contrastive learning (DrugCLIP-style)
        # Project from embedding_dim (256) to 128 for contrastive learning
        self.pocket_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 128)
        )
        self.ligand_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 128)
        )
    
    def forward(self, pocket_esp, ligand_esp):
        """
        Forward pass: encode → project → normalize → similarity
        
        Args:
            pocket_esp: [B, C_p, H, W, D] voxelized pocket ESP
            ligand_esp: [B, C_l, H, W, D] voxelized ligand ESP
        
        Returns:
            z_pocket: [B, 128] L2-normalized pocket embeddings
            z_ligand: [B, 128] L2-normalized ligand embeddings
            similarity: [B, 1] element-wise cosine similarity
        """
        # Standard path: encode directly to global embeddings
        z_pocket_raw = self.pocket_encoder(pocket_esp)  # [B, 256]
        z_ligand_raw = self.ligand_encoder(ligand_esp)  # [B, 256]
        
        # Project to 128-dim
        z_pocket = self.pocket_projection(z_pocket_raw)  # [B, 128]
        z_ligand = self.ligand_projection(z_ligand_raw)  # [B, 128]
        
        # L2 normalize for cosine similarity
        z_pocket = F.normalize(z_pocket, dim=-1)  # [B, 128]
        z_ligand = F.normalize(z_ligand, dim=-1)  # [B, 128]
        
        # Compute element-wise cosine similarity
        similarity = (z_pocket * z_ligand).sum(dim=-1, keepdim=True)  # [B, 1]
        
        return z_pocket, z_ligand, similarity


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
        embedding_dim=checkpoint.get('embedding_dim', 256),
        dropout=checkpoint.get('dropout', 0.3)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model
