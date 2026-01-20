"""
Test ESPCL2 generalization on Davis dataset.

Evaluates a model trained on PDBbind by testing on Davis protein-ligand pairs.
"""

import torch
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add ESPCL2 to path
sys.path.insert(0, '/home/patrick/Desktop/ESPCL2')

from models import ESP_JointNet
from utils.metrics import compute_recall_metrics, compute_auroc_bedroc, compute_similarity_stats


def load_model(checkpoint_path, device='cuda'):
    """Load trained ESPCL2 model."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize model
    model = ESP_JointNet(
        embedding_dim=256,
        dropout=0.3,
        pocket_channels=8,
        ligand_channels=8
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    
    # Handle DataParallel
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model


def evaluate_on_dataset(model, dataset_path, device='cuda', batch_size=100, save_embeddings=False):
    """Evaluate model on a dataset."""
    print(f"\nLoading dataset from {dataset_path}...")
    
    # Load dataset
    data = torch.load(dataset_path, weights_only=False, map_location='cpu')
    
    voxels = data['voxels']
    labels = data['labels']
    pdb_codes = data.get('pdb_codes', data.get('sample_ids', None))
    
    n_samples = len(voxels)
    n_channels = voxels.shape[1]
    channels_per_mol = n_channels // 2
    
    print(f"Dataset: {n_samples} samples, {n_channels} channels ({channels_per_mol} per molecule)")
    
    # Split into ligand and pocket
    ligand_voxels = voxels[:, :channels_per_mol]
    pocket_voxels = voxels[:, channels_per_mol:]
    
    # Compute embeddings in batches
    pocket_embeddings = []
    ligand_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Computing embeddings"):
            batch_end = min(i + batch_size, n_samples)
            
            pocket_batch = pocket_voxels[i:batch_end].to(device)
            ligand_batch = ligand_voxels[i:batch_end].to(device)
            
            z_pocket, z_ligand, _ = model(pocket_batch, ligand_batch)
            
            pocket_embeddings.append(z_pocket.cpu())
            ligand_embeddings.append(z_ligand.cpu())
    
    # Concatenate
    pocket_embeddings = torch.cat(pocket_embeddings, dim=0)
    ligand_embeddings = torch.cat(ligand_embeddings, dim=0)
    
    print(f"Embeddings shape: {pocket_embeddings.shape}")
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = torch.matmul(pocket_embeddings, ligand_embeddings.T)
    
    print(f"Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    
    # Compute metrics (use scaled logits for proper AUROC)
    print("\nComputing metrics...")
    B = similarity_matrix.shape[0]
    labels_idx = torch.arange(B)
    
    # Scale by temperature (same as training)
    temperature = 0.07
    logits = similarity_matrix / temperature
    
    recall_metrics = compute_recall_metrics(logits, labels_idx, B)
    auroc_metrics = compute_auroc_bedroc(logits)
    sim_stats = compute_similarity_stats(similarity_matrix)
    
    # Combine all metrics
    all_metrics = {**recall_metrics, **auroc_metrics, **sim_stats}
    
    if save_embeddings:
        return all_metrics, similarity_matrix, pdb_codes, pocket_embeddings, ligand_embeddings
    else:
        return all_metrics, similarity_matrix, pdb_codes


def print_results(metrics, dataset_name):
    """Print evaluation results."""
    print(f"\n{'='*80}")
    print(f"{dataset_name} Results (Zero-shot Generalization)")
    print('='*80)
    
    print(f"\nRecall Metrics:")
    print(f"  Pocket→Ligand:")
    print(f"    Recall@1%:  {metrics['recall1pct_p2l']*100:.2f}%")
    print(f"    Recall@5%:  {metrics['recall5pct_p2l']*100:.2f}%")
    print(f"    Recall@10%: {metrics['recall10pct_p2l']*100:.2f}%")
    print(f"  Ligand→Pocket:")
    print(f"    Recall@1%:  {metrics['recall1pct_l2p']*100:.2f}%")
    print(f"    Recall@5%:  {metrics['recall5pct_l2p']*100:.2f}%")
    print(f"    Recall@10%: {metrics['recall10pct_l2p']*100:.2f}%")
    
    print(f"\nEnrichment Factors:")
    print(f"  EF@1%:  P→L {metrics['ef1_p2l']:.1f}, L→P {metrics['ef1_l2p']:.1f}")
    print(f"  EF@5%:  P→L {metrics['ef5_p2l']:.1f}, L→P {metrics['ef5_l2p']:.1f}")
    print(f"  EF@10%: P→L {metrics['ef10_p2l']:.1f}, L→P {metrics['ef10_l2p']:.1f}")
    
    print(f"\nRanking Quality:")
    print(f"  AUROC: P→L {metrics['auroc_p2l']:.4f}, L→P {metrics['auroc_l2p']:.4f}")
    print(f"  BEDROC: P→L {metrics['bedroc_p2l']:.4f}, L→P {metrics['bedroc_l2p']:.4f}")
    
    if 'pos_similarity' in metrics:
        print(f"\nSimilarity Statistics:")
        print(f"  Positive pairs: {metrics['pos_similarity']:.3f}")
        print(f"  Negative pairs: {metrics['neg_similarity']:.3f}")
        print(f"  Separation: {metrics['separation']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Test ESPCL2 generalization on Davis dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--davis_dataset', type=str, default='/data/davis_field_based_v4.pt', help='Path to Davis dataset (.pt)')
    parser.add_argument('--output_dir', type=str, default='davis_evaluation', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save_embeddings', action='store_true', help='Save embeddings for visualization')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model (trained on PDBbind)
    model = load_model(args.checkpoint, args.device)
    
    # Evaluate on Davis
    results = evaluate_on_dataset(
        model, args.davis_dataset, args.device, args.batch_size, 
        save_embeddings=args.save_embeddings
    )
    
    if args.save_embeddings:
        davis_metrics, davis_similarity, davis_codes, pocket_embeds, ligand_embeds = results
    else:
        davis_metrics, davis_similarity, davis_codes = results
    
    # Print results
    print_results(davis_metrics, "Davis Dataset")
    
    # Save results (similarity matrix is too large for standard pickle)
    results_file = output_dir / 'davis_generalization_results.pt'
    save_dict = {
        'metrics': davis_metrics,
        'pdb_codes': davis_codes
    }
    
    if args.save_embeddings:
        save_dict['pocket_embeddings'] = pocket_embeds
        save_dict['ligand_embeddings'] = ligand_embeds
        print(f"\nSaving embeddings (shape: {pocket_embeds.shape})...")
    
    torch.save(save_dict, results_file, pickle_protocol=4)
    
    print(f"\nResults saved to {results_file}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([davis_metrics])
    metrics_df.to_csv(output_dir / 'davis_metrics.csv', index=False)
    
    print(f"Metrics saved to {output_dir / 'davis_metrics.csv'}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
