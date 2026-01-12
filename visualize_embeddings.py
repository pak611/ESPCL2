"""
Visualize contrastive embeddings to show cluster separation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from sklearn.manifold import TSNE
import seaborn as sns


def plot_embeddings_2d(embeddings, labels, title, save_path, method='tsne'):
    """
    Plot 2D projection of embeddings colored by affinity
    
    Args:
        embeddings: [N, D] embedding vectors
        labels: [N] affinity labels
        title: plot title
        save_path: where to save figure
        method: 'tsne' or 'umap'
    """
    # Reduce to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
    
    print(f"Reducing {embeddings.shape[0]} embeddings to 2D with {method.upper()}...")
    coords_2d = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Colored by affinity (continuous)
    scatter = axes[0].scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=labels, cmap='viridis', s=20, alpha=0.6
    )
    axes[0].set_title(f'{title} - Colored by Affinity')
    axes[0].set_xlabel(f'{method.upper()} 1')
    axes[0].set_ylabel(f'{method.upper()} 2')
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Binding Affinity (pKd)')
    
    # Plot 2: Colored by affinity bins (categorical)
    # Bin affinities into weak/moderate/strong
    bins = np.percentile(labels, [0, 33, 66, 100])
    binned_labels = np.digitize(labels, bins[1:-1])
    colors = ['red', 'orange', 'green']
    bin_names = ['Weak', 'Moderate', 'Strong']
    
    for i, (color, name) in enumerate(zip(colors, bin_names)):
        mask = binned_labels == i
        axes[1].scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=color, label=name, s=20, alpha=0.6
        )
    
    axes[1].set_title(f'{title} - Binned by Affinity')
    axes[1].set_xlabel(f'{method.upper()} 1')
    axes[1].set_ylabel(f'{method.upper()} 2')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {save_path}")


def plot_pair_similarity(pocket_emb, ligand_emb, labels, save_path):
    """
    Plot cosine similarity distribution for true pairs vs random pairs
    """
    # Normalize embeddings
    pocket_emb = pocket_emb / np.linalg.norm(pocket_emb, axis=1, keepdims=True)
    ligand_emb = ligand_emb / np.linalg.norm(ligand_emb, axis=1, keepdims=True)
    
    # True pair similarities (diagonal)
    true_similarities = np.sum(pocket_emb * ligand_emb, axis=1)
    
    # Random pair similarities (off-diagonal samples)
    n_samples = min(1000, len(pocket_emb))
    random_idx = np.random.permutation(len(pocket_emb))[:n_samples]
    random_similarities = []
    
    for i in range(n_samples):
        # Pick random ligand different from pocket index
        rand_lig_idx = (random_idx[i] + np.random.randint(1, len(ligand_emb))) % len(ligand_emb)
        sim = np.dot(pocket_emb[random_idx[i]], ligand_emb[rand_lig_idx])
        random_similarities.append(sim)
    
    random_similarities = np.array(random_similarities)
    
    # Plot distributions
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(true_similarities, bins=50, alpha=0.6, label='True Pairs', color='green', density=True)
    ax.hist(random_similarities, bins=50, alpha=0.6, label='Random Pairs', color='red', density=True)
    
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Contrastive Embedding Similarity: True vs Random Pairs')
    ax.legend()
    ax.axvline(true_similarities.mean(), color='green', linestyle='--', 
               label=f'True mean: {true_similarities.mean():.3f}')
    ax.axvline(random_similarities.mean(), color='red', linestyle='--',
               label=f'Random mean: {random_similarities.mean():.3f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved similarity plot to {save_path}")


def plot_training_progression(embedding_dir, output_path, max_epochs=None):
    """
    Create a grid showing embedding evolution over training epochs
    """
    embedding_dir = Path(embedding_dir)
    
    # Find all embedding files
    emb_files = sorted(embedding_dir.glob('embeddings_epoch_*.npz'))
    
    if max_epochs:
        emb_files = emb_files[:max_epochs]
    
    if len(emb_files) == 0:
        print("No embedding files found!")
        return
    
    # Select evenly spaced epochs
    n_plots = min(6, len(emb_files))
    indices = np.linspace(0, len(emb_files)-1, n_plots, dtype=int)
    selected_files = [emb_files[i] for i in indices]
    
    # Create subplot grid
    n_rows = 2
    n_cols = (n_plots + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()
    
    # Fit TSNE on all data once for consistency
    print("Loading all embeddings...")
    all_embeddings = []
    all_labels = []
    epoch_splits = []
    
    for emb_file in selected_files:
        data = np.load(emb_file)
        # Use pocket embeddings
        emb = data['pocket_embeddings']
        labels = data['labels']
        
        epoch_splits.append(len(all_embeddings))
        all_embeddings.append(emb)
        all_labels.append(labels)
    
    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)
    
    print(f"Fitting TSNE on {all_embeddings.shape[0]} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    all_coords = tsne.fit_transform(all_embeddings)
    
    # Plot each epoch
    epoch_splits.append(len(all_embeddings))
    
    for idx, (emb_file, ax) in enumerate(zip(selected_files, axes)):
        # Extract epoch number from filename
        epoch = int(emb_file.stem.split('_')[-1])
        
        # Get coordinates for this epoch
        start_idx = epoch_splits[idx]
        end_idx = epoch_splits[idx + 1]
        coords = all_coords[start_idx:end_idx]
        labels = all_labels[start_idx:end_idx]
        
        # Plot
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap='viridis', s=10, alpha=0.5
        )
        ax.set_title(f'Epoch {epoch}')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Remove empty subplots
    for idx in range(len(selected_files), len(axes)):
        fig.delaxes(axes[idx])
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axes[:len(selected_files)], fraction=0.046, pad=0.04)
    cbar.set_label('Binding Affinity (pKd)')
    
    plt.suptitle('Embedding Space Evolution During Training', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved progression plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-dir', type=str,
                        help='Directory containing saved embeddings (for multi-epoch)')
    parser.add_argument('--embedding-file', type=str,
                        help='Single embedding file to visualize')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                        help='Where to save visualizations')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'umap'],
                        help='Dimensionality reduction method')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    if args.embedding_file:
        # Single file mode
        emb_file = Path(args.embedding_file)
        print(f"Loading embeddings from {emb_file}")
        data = np.load(emb_file)
        epoch = "final"
    else:
        # Directory mode - find latest
        if not args.embedding_dir:
            print("Error: Must provide either --embedding-file or --embedding-dir")
            return
        
        embedding_dir = Path(args.embedding_dir)
        emb_files = sorted(embedding_dir.glob('embeddings_epoch_*.npz'))
        
        if len(emb_files) == 0:
            print("No embedding files found!")
            return
        
        # Load latest embeddings
        latest_file = emb_files[-1]
        epoch = int(latest_file.stem.split('_')[-1])
        print(f"Loading embeddings from {latest_file.name} (Epoch {epoch})")
        data = np.load(latest_file)
    pocket_emb = data['pocket_embeddings']
    ligand_emb = data['ligand_embeddings']
    labels = data['labels']
    
    print(f"Pocket embeddings: {pocket_emb.shape}")
    print(f"Ligand embeddings: {ligand_emb.shape}")
    print(f"Labels: {labels.shape}")
    
    # Plot pocket embeddings
    plot_embeddings_2d(
        pocket_emb, labels,
        f'Pocket Embeddings (Epoch {epoch})',
        output_dir / f'pocket_embeddings_epoch{epoch}.png',
        method=args.method
    )
    
    # Plot ligand embeddings
    plot_embeddings_2d(
        ligand_emb, labels,
        f'Ligand Embeddings (Epoch {epoch})',
        output_dir / f'ligand_embeddings_epoch{epoch}.png',
        method=args.method
    )
    
    # Plot combined (concatenated pocket + ligand)
    combined_emb = np.concatenate([pocket_emb, ligand_emb], axis=1)
    plot_embeddings_2d(
        combined_emb, labels,
        f'Combined Embeddings (Epoch {epoch})',
        output_dir / f'combined_embeddings_epoch{epoch}.png',
        method=args.method
    )
    
    # Plot true vs random pair similarities
    plot_pair_similarity(
        pocket_emb, ligand_emb, labels,
        output_dir / f'pair_similarity_epoch{epoch}.png'
    )
    
    # Plot training progression if multiple epochs available (only in directory mode)
    if args.embedding_dir and not args.embedding_file:
        emb_files = sorted(Path(args.embedding_dir).glob('embeddings_epoch_*.npz'))
        if len(emb_files) > 1:
            plot_training_progression(
                args.embedding_dir,
                output_dir / 'training_progression.png'
            )
    
    print(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()
