"""
Create publication-quality t-SNE visualization of protein-ligand embeddings
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import argparse
from collections import defaultdict
import seaborn as sns

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})


def create_publication_tsne(results_path, output_path, n_targets=12, perplexity=30, random_state=42):
    """
    Create publication-quality t-SNE visualization
    """
    print(f"Loading embeddings from {results_path}...")
    results = torch.load(results_path, weights_only=False)
    
    pocket_embeds = results['pocket_embeddings'].cpu().numpy()
    ligand_embeds = results['ligand_embeddings'].cpu().numpy()
    sample_ids = results['pdb_codes']
    
    print(f"Total samples: {len(sample_ids)}")
    
    # Load Davis dataset to get protein IDs
    try:
        davis_data = torch.load('/data/davis_field_based_v4_uniform.pt', weights_only=False)
        protein_ids = davis_data.get('protein_ids', None)
        target_ids = [f"P{pid}" for pid in protein_ids]
        print("Using Davis protein IDs")
    except:
        target_ids = [f"T{sid}" for sid in sample_ids]
    
    # Count samples per target
    target_counts = defaultdict(int)
    for tid in target_ids:
        target_counts[tid] += 1
    
    print(f"Unique targets: {len(target_counts)}")
    
    # Select top N targets
    top_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:n_targets]
    selected_targets = set([t[0] for t in top_targets])
    
    print(f"\nSelected top {n_targets} targets (samples each):")
    for tid, count in top_targets:
        print(f"  {tid}: {count}")
    
    # Filter to selected targets
    selected_indices = [i for i, tid in enumerate(target_ids) if tid in selected_targets]
    
    pocket_embeds_filtered = pocket_embeds[selected_indices]
    ligand_embeds_filtered = ligand_embeds[selected_indices]
    target_ids_filtered = [target_ids[i] for i in selected_indices]
    
    print(f"\nFiltered to {len(selected_indices)} samples")
    
    # Combine embeddings
    all_embeds = np.vstack([pocket_embeds_filtered, ligand_embeds_filtered])
    all_targets = target_ids_filtered + target_ids_filtered
    
    # Run t-SNE
    print(f"\nRunning t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                max_iter=1000, verbose=1)
    embeddings_2d = tsne.fit_transform(all_embeds)
    
    pocket_2d = embeddings_2d[:len(pocket_embeds_filtered)]
    ligand_2d = embeddings_2d[len(pocket_embeds_filtered):]
    
    # Use professional color palette
    unique_targets = sorted(selected_targets)
    
    # Use tableau colors (more distinguishable)
    if len(unique_targets) <= 10:
        colors = sns.color_palette("tab10", len(unique_targets))
    elif len(unique_targets) <= 20:
        colors = sns.color_palette("tab20", len(unique_targets))
    else:
        colors = sns.color_palette("husl", len(unique_targets))
    
    target_to_color = {tid: colors[i] for i, tid in enumerate(unique_targets)}
    
    # Create main publication figure
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    
    # Plot with high-quality markers
    marker_size = 40
    edge_width = 0.3
    alpha = 0.7
    
    # Plot each target
    for tid in unique_targets:
        pocket_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        ligand_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        
        color = target_to_color[tid]
        
        # Plot pockets (circles)
        if pocket_idx:
            ax.scatter(pocket_2d[pocket_idx, 0], pocket_2d[pocket_idx, 1],
                      c=[color], s=marker_size, alpha=alpha, marker='o',
                      edgecolors='black', linewidths=edge_width, zorder=3,
                      label=tid)
        
        # Plot ligands (triangles)
        if ligand_idx:
            ax.scatter(ligand_2d[ligand_idx, 0], ligand_2d[ligand_idx, 1],
                      c=[color], s=marker_size, alpha=alpha, marker='^',
                      edgecolors='black', linewidths=edge_width, zorder=3)
        
        # Draw connecting lines (subtle)
        for p_idx, l_idx in zip(pocket_idx, ligand_idx):
            ax.plot([pocket_2d[p_idx, 0], ligand_2d[l_idx, 0]],
                   [pocket_2d[p_idx, 1], ligand_2d[l_idx, 1]],
                   color=color, alpha=0.15, linewidth=0.8, zorder=1)
    
    # Styling
    ax.set_xlabel('t-SNE Component 1', fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    
    # Legend with custom entries
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Add legend for markers
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label='Protein pocket', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label='Ligand', linestyle='None'),
        Line2D([0], [0], color='gray', alpha=0.3, linewidth=1,
               label='Binding pair')
    ]
    
    # Add target colors
    for tid in unique_targets[:6]:  # Show first 6 in main legend
        legend_elements.append(
            Patch(facecolor=target_to_color[tid], edgecolor='black',
                  linewidth=0.5, label=tid)
        )
    
    ax.legend(handles=legend_elements, loc='best', frameon=True,
             fancybox=False, shadow=False, ncol=1, fontsize=9)
    
    plt.tight_layout()
    
    # Save main figure
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved publication figure to {output_path}")
    
    # Save as PDF (vector format)
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved PDF to {pdf_path}")
    
    # Save as EPS (vector format for some journals)
    eps_path = output_path.with_suffix('.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"Saved EPS to {eps_path}")
    
    plt.close()
    
    # Create supplementary figure with all targets labeled
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    for tid in unique_targets:
        pocket_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        ligand_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        
        color = target_to_color[tid]
        
        if pocket_idx:
            ax.scatter(pocket_2d[pocket_idx, 0], pocket_2d[pocket_idx, 1],
                      c=[color], s=30, alpha=0.7, marker='o',
                      edgecolors='black', linewidths=0.3, zorder=3)
        
        if ligand_idx:
            ax.scatter(ligand_2d[ligand_idx, 0], ligand_2d[ligand_idx, 1],
                      c=[color], s=30, alpha=0.7, marker='^',
                      edgecolors='black', linewidths=0.3, zorder=3)
        
        for p_idx, l_idx in zip(pocket_idx, ligand_idx):
            ax.plot([pocket_2d[p_idx, 0], ligand_2d[l_idx, 0]],
                   [pocket_2d[p_idx, 1], ligand_2d[l_idx, 1]],
                   color=color, alpha=0.1, linewidth=0.5, zorder=1)
    
    ax.set_xlabel('t-SNE Component 1', fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    
    # Full legend
    legend_elements = [Patch(facecolor=target_to_color[tid], edgecolor='black',
                            linewidth=0.5, label=tid) for tid in unique_targets]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
             frameon=True, fancybox=False, ncol=2, fontsize=8,
             title='Target ID', title_fontsize=9)
    
    plt.tight_layout()
    
    supp_path = output_path.parent / (output_path.stem + '_supplementary.png')
    plt.savefig(supp_path, dpi=300, bbox_inches='tight')
    print(f"Saved supplementary figure to {supp_path}")
    
    plt.close()
    
    # Compute clustering metrics
    print(f"\n{'='*70}")
    print("Clustering Quality Metrics")
    print(f"{'='*70}")
    
    # Silhouette-like metric: intra-cluster vs inter-cluster distances
    target_metrics = []
    for tid in unique_targets[:5]:
        pocket_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        ligand_idx = [i for i, t in enumerate(target_ids_filtered) if t == tid]
        
        if len(pocket_idx) > 0 and len(ligand_idx) > 0:
            # Compute mean distance between matching pairs
            pair_dists = []
            for p_idx, l_idx in zip(pocket_idx, ligand_idx):
                dist = np.linalg.norm(pocket_embeds_filtered[p_idx] - 
                                    ligand_embeds_filtered[l_idx])
                pair_dists.append(dist)
            
            mean_dist = np.mean(pair_dists)
            std_dist = np.std(pair_dists)
            
            print(f"{tid:8s}: mean={mean_dist:.4f}, std={std_dist:.4f}, "
                  f"n_pairs={len(pair_dists)}")
            target_metrics.append(mean_dist)
    
    if target_metrics:
        print(f"\nOverall mean pair distance: {np.mean(target_metrics):.4f} ± {np.std(target_metrics):.4f}")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Create publication-quality t-SNE visualization')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results .pt file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for the plot')
    parser.add_argument('--n_targets', type=int, default=12,
                        help='Number of top targets to visualize')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    create_publication_tsne(
        args.results,
        args.output,
        n_targets=args.n_targets,
        perplexity=args.perplexity,
        random_state=args.random_state
    )


if __name__ == '__main__':
    main()
