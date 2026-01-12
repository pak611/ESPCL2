"""
Extract contrastive embeddings from trained ESPCL2 checkpoint
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from models.esp_jointnet import ESP_JointNet


@torch.no_grad()
def extract_embeddings_from_model(model, loader, device, max_samples=5000):
    """
    Extract embeddings after cross-attention from trained model
    
    Returns embeddings, attention-enhanced features for visualization
    """
    model.eval()
    
    pocket_features = []
    ligand_features = []
    pocket_pooled = []
    ligand_pooled = []
    labels = []
    predictions = []
    
    n_samples = 0
    
    for batch in tqdm(loader, desc='Extracting embeddings'):
        if n_samples >= max_samples:
            break
        
        voxels, batch_labels = batch[0].to(device), batch[1]
        
        # Split unified grid: first 10 channels = pocket, last 9 = ligand
        pocket_esp = voxels[:, :10, :, :, :]
        ligand_esp = voxels[:, 10:, :, :, :]
        
        # Forward pass - get contrastive embeddings
        outputs = model(pocket_esp, ligand_esp)
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            # Model returns (predictions, z_pocket_cl, z_ligand_cl, ...)
            pred_affinity = outputs[0]
            z_pocket_cl = outputs[1]  # L2-normalized contrastive embeddings
            z_ligand_cl = outputs[2]
            
            # Use the actual contrastive embeddings (already normalized)
            pocket_emb = z_pocket_cl
            ligand_emb = z_ligand_cl
        else:
            # Fallback: extract from model layers
            pred_affinity = outputs[0] if isinstance(outputs, tuple) else outputs
            pocket_feat = model.pocket_encoder(pocket_esp)
            ligand_feat = model.ligand_encoder(ligand_esp)
            pocket_emb = F.adaptive_avg_pool3d(pocket_feat, 1).squeeze(-1).squeeze(-1).squeeze(-1)
            ligand_emb = F.adaptive_avg_pool3d(ligand_feat, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Store
        pocket_pooled.append(pocket_emb.cpu().numpy())
        ligand_pooled.append(ligand_emb.cpu().numpy())
        labels.append(batch_labels.numpy())
        predictions.append(pred_affinity.squeeze().cpu().numpy())
        
        n_samples += voxels.size(0)
    
    return {
        'pocket_embeddings': np.concatenate(pocket_pooled),
        'ligand_embeddings': np.concatenate(ligand_pooled),
        'labels': np.concatenate(labels),
        'predictions': np.concatenate(predictions)
    }


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained ESPCL2 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to voxelized dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for embeddings (.npz file)')
    parser.add_argument('--max-samples', type=int, default=5000,
                        help='Maximum samples to extract')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Which data split to extract from')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Initialize model with checkpoint config
    config = checkpoint.get('config', {})
    model = ESP_JointNet(
        pocket_channels=10,
        ligand_channels=9,
        use_cross_attention=config.get('use_cross_attention', True),
        use_pairwise_attention=config.get('use_pairwise_attention', False),
        embedding_dim=config.get('embedding_dim', 256),
        dropout=config.get('dropout', 0.3)
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Load dataset
    print(f"Loading dataset from {args.data}")
    data = torch.load(args.data, weights_only=False)
    
    voxels = data['unified_voxels']
    labels = data['labels']
    protein_ids = data.get('protein_ids', None)
    ligand_ids = data.get('ligand_ids', None)
    
    print(f"Dataset: {voxels.shape[0]} samples")
    
    # Create dataset
    dataset = TensorDataset(voxels, labels)
    
    # Split dataset (same as training: 80/10/10)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    
    # Use same random seed for consistent splits
    torch.manual_seed(42)
    indices = torch.randperm(n_total).tolist()
    
    if args.split == 'train':
        split_indices = indices[:n_train]
    elif args.split == 'val':
        split_indices = indices[n_train:n_train+n_val]
    else:  # test
        split_indices = indices[n_train+n_val:]
    
    split_dataset = Subset(dataset, split_indices)
    print(f"Using {args.split} split: {len(split_dataset)} samples")
    
    loader = DataLoader(
        split_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings_from_model(model, loader, device, args.max_samples)
    
    print(f"Extracted embeddings:")
    print(f"  Pocket: {embeddings['pocket_embeddings'].shape}")
    print(f"  Ligand: {embeddings['ligand_embeddings'].shape}")
    print(f"  Labels: {embeddings['labels'].shape}")
    
    # Add metadata if available
    if protein_ids is not None:
        split_protein_ids = [protein_ids[i] for i in split_indices[:len(embeddings['labels'])]]
        split_ligand_ids = [ligand_ids[i] for i in split_indices[:len(embeddings['labels'])]]
        embeddings['protein_ids'] = split_protein_ids
        embeddings['ligand_ids'] = split_ligand_ids
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        pocket_embeddings=embeddings['pocket_embeddings'],
        ligand_embeddings=embeddings['ligand_embeddings'],
        labels=embeddings['labels'],
        predictions=embeddings['predictions'],
        protein_ids=embeddings.get('protein_ids', None),
        ligand_ids=embeddings.get('ligand_ids', None)
    )
    
    print(f"✓ Embeddings saved to {output_path}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"  RMSE: {np.sqrt(np.mean((embeddings['predictions'] - embeddings['labels'])**2)):.4f}")
    print(f"  Pearson: {np.corrcoef(embeddings['predictions'], embeddings['labels'])[0,1]:.4f}")
    
    # Analyze embedding space
    pocket_emb = embeddings['pocket_embeddings']
    ligand_emb = embeddings['ligand_embeddings']
    
    # Normalize for cosine similarity
    pocket_norm = pocket_emb / (np.linalg.norm(pocket_emb, axis=1, keepdims=True) + 1e-8)
    ligand_norm = ligand_emb / (np.linalg.norm(ligand_emb, axis=1, keepdims=True) + 1e-8)
    
    # True pair similarities (diagonal)
    true_sims = np.sum(pocket_norm * ligand_norm, axis=1)
    
    # Random pair similarities (sample)
    n_random = min(1000, len(pocket_norm))
    random_sims = []
    for i in range(n_random):
        j = (i + np.random.randint(1, len(ligand_norm))) % len(ligand_norm)
        random_sims.append(np.dot(pocket_norm[i], ligand_norm[j]))
    random_sims = np.array(random_sims)
    
    print(f"\nEmbedding Space Analysis:")
    print(f"  True pair similarity: {true_sims.mean():.4f} ± {true_sims.std():.4f}")
    print(f"  Random pair similarity: {random_sims.mean():.4f} ± {random_sims.std():.4f}")
    print(f"  Separation: {true_sims.mean() - random_sims.mean():.4f}")


if __name__ == '__main__':
    main()
