"""
Loss functions for contrastive learning
"""

import torch
import torch.nn as nn
from .metrics import compute_recall_metrics, compute_auroc_bedroc, compute_similarity_stats


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning with multi-positive support"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, similarity_matrix, protein_seqs=None, ligand_smiles_list=None):
        """
        Args:
            similarity_matrix: [B, B] cosine similarity between all pocket-ligand pairs
                              similarity[i, j] = similarity between pocket_i and ligand_j
            protein_seqs: List of protein sequences for each sample (optional)
            ligand_smiles_list: List of ligand SMILES for each sample (optional)
        
        Returns:
            loss: scalar InfoNCE loss
            metrics: dict with accuracy and similarity stats
        """
        B = similarity_matrix.size(0)
        device = similarity_matrix.device
        
        # Scale by temperature
        logits = similarity_matrix / self.temperature
        
        # Build positive pairs matrix based on sequence+SMILES identity
        if protein_seqs is not None and ligand_smiles_list is not None:
            # Create binary matrix where [i,j]=1 if same protein-ligand pair
            positive_mask = torch.zeros(B, B, dtype=torch.bool, device=device)
            for i in range(B):
                for j in range(B):
                    if protein_seqs[i] == protein_seqs[j] and ligand_smiles_list[i] == ligand_smiles_list[j]:
                        positive_mask[i, j] = True
        else:
            # Fallback: only diagonal elements are positive
            positive_mask = torch.eye(B, dtype=torch.bool, device=device)
        
        # Compute multi-label contrastive loss
        # For each query, we want high similarity to all its positives
        labels = torch.arange(B, device=device)
        
        # Bidirectional loss: pocket→ligand and ligand→pocket
        # Use log-sum-exp for multi-positive contrastive learning
        loss_p2l = 0
        loss_l2p = 0
        
        for i in range(B):
            # P→L: For row i, sum over all positives
            pos_mask_i = positive_mask[i]
            if pos_mask_i.sum() > 0:
                pos_logits = logits[i][pos_mask_i]
                all_logits = logits[i]
                loss_p2l += -torch.logsumexp(pos_logits, dim=0) + torch.logsumexp(all_logits, dim=0)
            
            # L→P: For column i, sum over all positives
            pos_mask_i = positive_mask[:, i]
            if pos_mask_i.sum() > 0:
                pos_logits = logits.T[i][pos_mask_i]
                all_logits = logits.T[i]
                loss_l2p += -torch.logsumexp(pos_logits, dim=0) + torch.logsumexp(all_logits, dim=0)
        
        loss_p2l = loss_p2l / B
        loss_l2p = loss_l2p / B
        loss = (loss_p2l + loss_l2p) / 2
        
        # Compute evaluation metrics
        metrics = {'loss': loss.item()}
        
        # Recall and enrichment metrics (use scaled logits)
        recall_metrics = compute_recall_metrics(logits, labels, B)
        metrics.update(recall_metrics)
        
        # AUROC and BEDROC (use scaled logits for consistent ranking)
        auroc_bedroc_metrics = compute_auroc_bedroc(logits)
        metrics.update(auroc_bedroc_metrics)
        
        # Similarity statistics (use raw similarities for interpretability)
        sim_stats = compute_similarity_stats(similarity_matrix)
        metrics.update(sim_stats)
        
        return loss, metrics
