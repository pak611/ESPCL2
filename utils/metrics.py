"""
Evaluation metrics for contrastive learning and retrieval tasks
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_recall_metrics(logits, labels, B):
    """
    Compute recall@k metrics for different percentages
    
    Args:
        logits: [B, B] scaled similarity scores
        labels: [B] ground truth labels (typically torch.arange(B))
        B: batch size
    
    Returns:
        dict with recall and enrichment factor metrics
    """
    metrics = {}
    
    # Recall@1% (top 1% of batch)
    k1 = max(1, int(B * 0.01))
    _, top1pct_p2l = logits.topk(k1, dim=1)
    _, top1pct_l2p = logits.T.topk(k1, dim=1)
    recall1pct_p2l = (top1pct_p2l == labels.unsqueeze(1)).any(dim=1).float().mean()
    recall1pct_l2p = (top1pct_l2p == labels.unsqueeze(1)).any(dim=1).float().mean()
    
    # Recall@5% (top 5% of batch)
    k5 = max(1, int(B * 0.05))
    _, top5pct_p2l = logits.topk(k5, dim=1)
    _, top5pct_l2p = logits.T.topk(k5, dim=1)
    recall5pct_p2l = (top5pct_p2l == labels.unsqueeze(1)).any(dim=1).float().mean()
    recall5pct_l2p = (top5pct_l2p == labels.unsqueeze(1)).any(dim=1).float().mean()
    
    # Recall@10% (top 10% of batch)
    k10 = max(1, int(B * 0.10))
    _, top10pct_p2l = logits.topk(k10, dim=1)
    _, top10pct_l2p = logits.T.topk(k10, dim=1)
    recall10pct_p2l = (top10pct_p2l == labels.unsqueeze(1)).any(dim=1).float().mean()
    recall10pct_l2p = (top10pct_l2p == labels.unsqueeze(1)).any(dim=1).float().mean()
    
    # Compute enrichment factors
    ef1_p2l = recall1pct_p2l / 0.01
    ef1_l2p = recall1pct_l2p / 0.01
    ef5_p2l = recall5pct_p2l / 0.05
    ef5_l2p = recall5pct_l2p / 0.05
    ef10_p2l = recall10pct_p2l / 0.10
    ef10_l2p = recall10pct_l2p / 0.10
    
    metrics.update({
        'recall1pct_p2l': recall1pct_p2l.item(),
        'recall1pct_l2p': recall1pct_l2p.item(),
        'recall5pct_p2l': recall5pct_p2l.item(),
        'recall5pct_l2p': recall5pct_l2p.item(),
        'recall10pct_p2l': recall10pct_p2l.item(),
        'recall10pct_l2p': recall10pct_l2p.item(),
        'ef1_p2l': ef1_p2l.item(),
        'ef1_l2p': ef1_l2p.item(),
        'ef5_p2l': ef5_p2l.item(),
        'ef5_l2p': ef5_l2p.item(),
        'ef10_p2l': ef10_p2l.item(),
        'ef10_l2p': ef10_l2p.item(),
    })
    
    return metrics


def compute_auroc_bedroc(similarity_matrix, alpha=20.0):
    """
    Compute AUROC and BEDROC for each sample
    
    Args:
        similarity_matrix: [B, B] similarity scores
        alpha: BEDROC parameter (emphasis on early retrieval)
    
    Returns:
        dict with mean AUROC and BEDROC scores
    """
    B = similarity_matrix.size(0)
    device = similarity_matrix.device
    
    auroc_p2l_list = []
    auroc_l2p_list = []
    bedroc_p2l_list = []
    bedroc_l2p_list = []
    
    for i in range(B):
        # P→L: For each pocket, compute AUROC/BEDROC over all ligands
        y_true_p2l = torch.zeros(B, device=device)
        y_true_p2l[i] = 1
        scores_p2l = similarity_matrix[i]
        
        try:
            auroc_p2l = roc_auc_score(y_true_p2l.cpu().numpy(), scores_p2l.cpu().numpy())
            auroc_p2l_list.append(auroc_p2l)
            
            # BEDROC calculation (Truchon & Bayly, 2007)
            bedroc_val = compute_bedroc(scores_p2l, y_true_p2l, alpha)
            if bedroc_val is not None:
                bedroc_p2l_list.append(bedroc_val)
        except:
            pass
        
        # L→P: For each ligand, compute AUROC/BEDROC over all pockets
        y_true_l2p = torch.zeros(B, device=device)
        y_true_l2p[i] = 1
        scores_l2p = similarity_matrix[:, i]
        
        try:
            auroc_l2p = roc_auc_score(y_true_l2p.cpu().numpy(), scores_l2p.cpu().numpy())
            auroc_l2p_list.append(auroc_l2p)
            
            # BEDROC calculation
            bedroc_val = compute_bedroc(scores_l2p, y_true_l2p, alpha)
            if bedroc_val is not None:
                bedroc_l2p_list.append(bedroc_val)
        except:
            pass
    
    auroc_p2l_mean = np.mean(auroc_p2l_list) if auroc_p2l_list else 0.5
    auroc_l2p_mean = np.mean(auroc_l2p_list) if auroc_l2p_list else 0.5
    bedroc_p2l_mean = np.mean(bedroc_p2l_list) if bedroc_p2l_list else 0.0
    bedroc_l2p_mean = np.mean(bedroc_l2p_list) if bedroc_l2p_list else 0.0
    
    return {
        'auroc_p2l': auroc_p2l_mean,
        'auroc_l2p': auroc_l2p_mean,
        'bedroc_p2l': bedroc_p2l_mean,
        'bedroc_l2p': bedroc_l2p_mean,
    }


def compute_bedroc(scores, labels, alpha=20.0):
    """
    Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC)
    Reference: Truchon & Bayly, J. Chem. Inf. Model. 2007
    
    Args:
        scores: [N] prediction scores
        labels: [N] binary labels (1 for active, 0 for inactive)
        alpha: BEDROC parameter controlling early retrieval emphasis
    
    Returns:
        bedroc: float, BEDROC score (or None if computation fails)
    """
    try:
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices].cpu().numpy()
        n = len(sorted_labels)
        n_actives = int(sorted_labels.sum())
        
        if n_actives == 0:
            return None
        
        # Positions of actives (1-indexed)
        ri = np.where(sorted_labels == 1)[0] + 1
        sum_exp = np.sum(np.exp(-alpha * ri / n))
        
        # Random and maximum values
        exp_alpha = np.exp(alpha / n)
        r_random = n_actives * (1 - np.exp(-alpha)) / (exp_alpha - 1) if exp_alpha != 1 else n_actives * alpha / n
        
        sinh_term = np.sinh(alpha / 2)
        cosh_term_1 = np.cosh(alpha / 2)
        cosh_term_2 = np.cosh(alpha / 2 - alpha * n_actives / n)
        r_max = n_actives * sinh_term / (cosh_term_1 - cosh_term_2) if cosh_term_1 != cosh_term_2 else n_actives
        
        # BEDROC formula
        bedroc = (sum_exp - r_random) / (r_max - r_random) if r_max != r_random else 0.0
        
        return bedroc
    except:
        return None


def compute_similarity_stats(similarity_matrix):
    """
    Compute statistics about similarity matrix
    
    Args:
        similarity_matrix: [B, B] similarity scores
    
    Returns:
        dict with pos_sim, neg_sim, separation
    """
    B = similarity_matrix.size(0)
    
    # Positive similarity: diagonal elements
    pos_sim = torch.diagonal(similarity_matrix).mean()
    
    # Negative similarity: off-diagonal elements
    neg_sim = (similarity_matrix.sum() - torch.diagonal(similarity_matrix).sum()) / (B * B - B)
    
    return {
        'pos_sim': pos_sim.item(),
        'neg_sim': neg_sim.item(),
        'separation': (pos_sim - neg_sim).item()
    }
