"""
Visualize embeddings grouped by target/protein identity
Shows how different targets cluster with their respective ligands
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import argparse
from collections import defaultdict


def plot_target_clusters(results_path, output_path, n_targets=20, perplexity=30, random_state=42):
    """
    Visualize embeddings grouped by target identity
    
    Args:
        results_path: Path to saved results file (.pt)
        output_path: Path to save the plot
        n_targets: Number of targets to visualize
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
    """
    print(f"Loading embeddings from {results_path}...")
    results = torch.load(results_path, weights_only=False)
    
    pocket_embeds = results['pocket_embeddings'].cpu().numpy()
    ligand_embeds = results['ligand_embeddings'].cpu().numpy()
    sample_ids = results['pdb_codes']
    
    print(f"Pocket embeddings: {pocket_embeds.shape}")
    print(f"Ligand embeddings: {ligand_embeds.shape}")
    print(f"Total samples: {len(sample_ids)}")
    
    # Check if dataset has protein_ids (Davis format)
    # Load the original dataset to get protein IDs
    dataset_path = Path(results_path).parent.parent / 'data'
    
    # Try to infer dataset from path
    if 'davis' in str(results_path).lower():
        try:
            # Load Davis dataset to get protein IDs
            davis_data = torch.load('/data/davis_field_based_v4_uniform.pt', weights_only=False)
            protein_ids = davis_data.get('protein_ids', None)
            if protein_ids is not None:
                target_ids = [f"P{pid}" for pid in protein_ids]
                print(f"Using Davis protein IDs")
            else:
                raise ValueError("No protein_ids found")
        except:
            print("Warning: Could not load protein IDs from Davis dataset")
            target_ids = [f"T{sid}" for sid in sample_ids]
    else:
        # Extract target identifiers from sample IDs (BindingDB format)
        # For BindingDB: id_3fqa -> 3fqa
        target_ids = []
        for sid in sample_ids:
            if isinstance(sid, str):
                # Remove id_ prefix if present
                tid = sid.replace('id_', '')
                # Take first 4 characters as target (PDB code)
                target_ids.append(tid[:4].upper())
            else:
                target_ids.append(f"T{sid}")
    
    # Count samples per target
    target_counts = defaultdict(int)
    for tid in target_ids:
        target_counts[tid] += 1
    
    print(f"\nUnique targets: {len(target_counts)}")
    print(f"Samples per target range: {min(target_counts.values())}-{max(target_counts.values())}")
    
    # Select top N most frequent targets
    top_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:n_targets]
    selected_targets = set([t[0] for t in top_targets])
    
    print(f"\nSelected top {n_targets} targets:")
    for tid, count in top_targets:
        print(f"  {tid}: {count} samples")
    
    # Filter to selected targets
    selected_indices = [i for i, tid in enumerate(target_ids) if tid in selected_targets]
    
    pocket_embeds_filtered = pocket_embeds[selected_indices]
    ligand_embeds_filtered = ligand_embeds[selected_indices]
    target_ids_filtered = [target_ids[i] for i in selected_indices]
    
    print(f"\nFiltered to {len(selected_indices)} samples from {len(selected_targets)} targets")
    
    # Combine pocket and ligand embeddings
    all_embeds = np.vstack([pocket_embeds_filtered, ligand_embeds_filtered])
    
    # Create target labels (repeated for pocket and ligand)
    all_targets = target_ids_filtered + target_ids_filtered
    
    # Create type labels (0=pocket, 1=ligand)
    type_labels = np.array([0] * len(pocket_embeds_filtered) + [1] * len(ligand_embeds_filtered))
    
    print(f"\nRunning t-SNE on {all_embeds.shape[0]} embeddings...")
    print(f"Perplexity: {perplexity}")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        verbose=1
    )
    
    embeddings_2d = tsne.fit_transform(all_embeds)
    
    # Split back into pocket and ligand
    pocket_2d = embeddings_2d[:len(pocket_embeds_filtered)]
    ligand_2d = embeddings_2d[len(pocket_embeds_filtered):]
    
    # Create color map for targets
    unique_targets = sorted(selected_targets)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_targets)))
    target_to_color = {tid: colors[i] for i, tid in enumerate(unique_targets)}
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Colored by target, with pockets and ligands distinguished
    ax = axes[0]
    
    for tid in unique_targets:
        # Get indices for this target
        pocket_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        ligand_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        
        color = target_to_color[tid]
        
        # Plot pockets as circles
        if pocket_idx:
            ax.scatter(pocket_2d[pocket_idx, 0], pocket_2d[pocket_idx, 1],
                      c=[color], s=80, alpha=0.7, marker='o', 
                      edgecolors='black', linewidths=0.5,
                      label=f'{tid} (pocket)')
        
        # Plot ligands as triangles
        if ligand_idx:
            ax.scatter(ligand_2d[ligand_idx, 0], ligand_2d[ligand_idx, 1],
                      c=[color], s=80, alpha=0.7, marker='^',
                      edgecolors='black', linewidths=0.5,
                      label=f'{tid} (ligand)')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Target Clustering (○=Pocket, △=Ligand)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: With connecting lines between matching pairs
    ax = axes[1]
    
    for tid in unique_targets:
        pocket_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        ligand_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        
        color = target_to_color[tid]
        
        # Draw lines first (background)
        for p_idx, l_idx in zip(pocket_idx, ligand_idx):
            ax.plot([pocket_2d[p_idx, 0], ligand_2d[l_idx, 0]], 
                   [pocket_2d[p_idx, 1], ligand_2d[l_idx, 1]], 
                   color=color, alpha=0.3, linewidth=1.5, zorder=1)
        
        # Plot points on top
        if pocket_idx:
            ax.scatter(pocket_2d[pocket_idx, 0], pocket_2d[pocket_idx, 1],
                      c=[color], s=80, alpha=0.8, marker='o',
                      edgecolors='black', linewidths=0.5, zorder=2)
        
        if ligand_idx:
            ax.scatter(ligand_2d[ligand_idx, 0], ligand_2d[ligand_idx, 1],
                      c=[color], s=80, alpha=0.8, marker='^',
                      edgecolors='black', linewidths=0.5, zorder=2)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Target Clustering with Matching Pairs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create legend with target names only (not every point)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=target_to_color[tid], 
                            edgecolor='black', label=tid) 
                      for tid in unique_targets]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
             ncol=1, fontsize=9, title='Targets')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to {pdf_path}")
    
    plt.close()
    
    # Create a simplified single-plot version
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    for tid in unique_targets:
        pocket_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        ligand_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        
        color = target_to_color[tid]
        
        # Draw lines
        for p_idx, l_idx in zip(pocket_idx, ligand_idx):
            ax.plot([pocket_2d[p_idx, 0], ligand_2d[l_idx, 0]], 
                   [pocket_2d[p_idx, 1], ligand_2d[l_idx, 1]], 
                   color=color, alpha=0.2, linewidth=1, zorder=1)
        
        # Plot points
        if pocket_idx:
            ax.scatter(pocket_2d[pocket_idx, 0], pocket_2d[pocket_idx, 1],
                      c=[color], s=60, alpha=0.8, marker='o',
                      edgecolors='black', linewidths=0.5, zorder=2)
        
        if ligand_idx:
            ax.scatter(ligand_2d[ligand_idx, 0], ligand_2d[ligand_idx, 1],
                      c=[color], s=60, alpha=0.8, marker='^',
                      edgecolors='black', linewidths=0.5, zorder=2)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title(f'Target-Ligand Clustering (Top {n_targets} Targets)\n○=Pocket  △=Ligand', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [Patch(facecolor=target_to_color[tid], 
                            edgecolor='black', label=tid) 
                      for tid in unique_targets]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
             ncol=2, fontsize=9, title='Target ID', title_fontsize=11)
    
    plt.tight_layout()
    
    single_path = output_path.parent / (output_path.stem + '_simple.png')
    plt.savefig(single_path, dpi=300, bbox_inches='tight')
    print(f"Simple plot saved to {single_path}")
    
    plt.close()
    
    # Compute clustering quality
    print("\n" + "="*80)
    print("Clustering Statistics")
    print("="*80)
    
    # Compute average distance between matching pairs for each target
    for tid in unique_targets[:5]:  # Show first 5
        pocket_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        ligand_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        
        if pocket_idx and ligand_idx:
            # Compute distances between matching pairs
            dists = []
            for p_idx, l_idx in zip(pocket_idx, ligand_idx):
                dist = np.linalg.norm(pocket_embeds_filtered[p_idx] - ligand_embeds_filtered[l_idx])
                dists.append(dist)
            
            print(f"{tid}: Mean pair distance = {np.mean(dists):.4f} ± {np.std(dists):.4f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Visualize target clustering with t-SNE')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results .pt file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for the plot')
    parser.add_argument('--n_targets', type=int, default=20,
                        help='Number of top targets to visualize')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    plot_target_clusters(
        args.results,
        args.output,
        n_targets=args.n_targets,
        perplexity=args.perplexity,
        random_state=args.random_state
    )


if __name__ == '__main__':
    main()
