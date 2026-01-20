"""
Visualize voxelized field-based representations of protein-ligand complexes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import argparse
from pathlib import Path


def visualize_sample(data_path, sample_idx=0, output_dir='voxel_visualizations', show_channels=None):
    """
    Visualize a voxelized protein-ligand complex
    
    Args:
        data_path: Path to dataset .pt file
        sample_idx: Index of sample to visualize
        output_dir: Directory to save visualizations
        show_channels: List of channel indices to visualize (default: all)
    """
    print(f"Loading dataset from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    
    voxels = data['voxels']
    labels = data['labels']
    sample_ids = data.get('sample_ids', data.get('pdb_codes', None))
    
    print(f"Dataset shape: {voxels.shape}")
    print(f"Number of samples: {len(voxels)}")
    print(f"Grid size: {voxels.shape[2:]}")
    print(f"Number of channels: {voxels.shape[1]}")
    
    # Get sample
    sample = voxels[sample_idx].cpu().numpy()
    label = labels[sample_idx].item()
    sample_id = sample_ids[sample_idx] if sample_ids is not None else sample_idx
    
    print(f"\nVisualizing sample {sample_idx}: {sample_id}")
    print(f"Label (binding affinity): {label:.3f}")
    print(f"Sample shape: {sample.shape}")
    
    # Channel names (8 ligand + 8 pocket - CORRECT ORDER)
    channel_names = [
        'Ligand ESP', 'Ligand Rotatable', 'Ligand Hydrophobic', 'Ligand Donor',
        'Ligand Acceptor', 'Ligand MolType', 'Ligand Aromatic', 'Ligand AtomType',
        'Pocket ESP', 'Pocket Rotatable', 'Pocket Hydrophobic', 'Pocket Donor',
        'Pocket Acceptor', 'Pocket MolType', 'Pocket Aromatic', 'Pocket AtomType'
    ]
    
    n_channels = sample.shape[0]
    pocket_channels = n_channels // 2
    
    # Determine which channels to show
    if show_channels is None:
        # Show first 4 of each (pocket and ligand)
        show_channels = list(range(4)) + list(range(pocket_channels, pocket_channels + 4))
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === Visualization 1: 2D slices of all channels ===
    print(f"\nCreating 2D slice visualization...")
    
    n_show = len(show_channels)
    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Get middle slice
    slice_idx = sample.shape[1] // 2
    
    for idx, ch_idx in enumerate(show_channels):
        ax = axes[idx]
        
        # Get 2D slice through middle of voxel grid
        slice_data = sample[ch_idx, slice_idx, :, :]
        
        # Plot
        im = ax.imshow(slice_data, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'{channel_names[ch_idx]}\nMax: {slice_data.max():.2f}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide extra subplots
    for idx in range(n_show, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Sample {sample_id} - 2D Slices (z={slice_idx})\nBinding Affinity: {label:.3f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    slice_path = output_dir / f'sample_{sample_idx}_slices.png'
    plt.savefig(slice_path, dpi=150, bbox_inches='tight')
    print(f"Saved 2D slices to {slice_path}")
    plt.close()
    
    # === Visualization 2: 3D isosurface visualization ===
    print(f"\nCreating 3D isosurface visualization...")
    
    # Select a few key channels for 3D viz (CORRECTED: 0-7=ligand, 8-15=pocket)
    key_channels = [
        (0, 'Ligand ESP', 'Reds'),
        (2, 'Ligand Hydrophobic', 'Oranges'),
        (8, 'Pocket ESP', 'Blues'),
        (10, 'Pocket Hydrophobic', 'Greens')
    ]
    
    fig = plt.figure(figsize=(16, 12))
    
    for plot_idx, (ch_idx, name, cmap) in enumerate(key_channels):
        ax = fig.add_subplot(2, 2, plot_idx + 1, projection='3d')
        
        channel_data = sample[ch_idx]
        
        # Get threshold for isosurface (show regions with significant values)
        threshold = np.percentile(np.abs(channel_data), 75)
        
        # Create meshgrid
        x, y, z = np.where(np.abs(channel_data) > threshold)
        
        if len(x) > 0:
            # Color by value
            colors = channel_data[x, y, z]
            
            # Scatter plot
            scatter = ax.scatter(x, y, z, c=colors, cmap=cmap, 
                               alpha=0.6, s=20, edgecolors='none')
            plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(name, fontweight='bold')
        
        # Set equal aspect ratio
        max_range = max(sample.shape[1:])
        ax.set_xlim([0, max_range])
        ax.set_ylim([0, max_range])
        ax.set_zlim([0, max_range])
    
    plt.suptitle(f'Sample {sample_id} - 3D Isosurfaces\nBinding Affinity: {label:.3f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    iso_path = output_dir / f'sample_{sample_idx}_3d.png'
    plt.savefig(iso_path, dpi=150, bbox_inches='tight')
    print(f"Saved 3D visualization to {iso_path}")
    plt.close()
    
    # === Visualization 3: Maximum intensity projections ===
    print(f"\nCreating maximum intensity projections...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    projections = [
        ('X (YZ plane)', 0),
        ('Y (XZ plane)', 1),
        ('Z (XY plane)', 2)
    ]
    
    for row, (ch_idx, ch_name) in enumerate([(0, 'Ligand ESP'), (8, 'Pocket ESP')]):
        channel_data = sample[ch_idx]
        
        # X projection (along axis 0)
        ax = axes[row, 0]
        proj = np.max(channel_data, axis=0)
        im = ax.imshow(proj, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'{ch_name}\nX-projection (max)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Y projection (along axis 1)
        ax = axes[row, 1]
        proj = np.max(channel_data, axis=1)
        im = ax.imshow(proj, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'{ch_name}\nY-projection (max)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Z projection (along axis 2)
        ax = axes[row, 2]
        proj = np.max(channel_data, axis=2)
        im = ax.imshow(proj, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'{ch_name}\nZ-projection (max)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Combined (mean of all projections)
        ax = axes[row, 3]
        proj = np.mean(channel_data, axis=0)
        im = ax.imshow(proj, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'{ch_name}\nMean intensity')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'Sample {sample_id} - Maximum Intensity Projections\nBinding Affinity: {label:.3f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    proj_path = output_dir / f'sample_{sample_idx}_projections.png'
    plt.savefig(proj_path, dpi=150, bbox_inches='tight')
    print(f"Saved projections to {proj_path}")
    plt.close()
    
    # === Print statistics ===
    print(f"\n{'='*80}")
    print("Channel Statistics")
    print(f"{'='*80}")
    
    for ch_idx in show_channels:
        ch_data = sample[ch_idx]
        print(f"{channel_names[ch_idx]:20s}: "
              f"min={ch_data.min():7.3f}, "
              f"max={ch_data.max():7.3f}, "
              f"mean={ch_data.mean():7.3f}, "
              f"std={ch_data.std():7.3f}, "
              f"nonzero={np.count_nonzero(ch_data):5d}/{ch_data.size:5d}")
    
    print(f"{'='*80}\n")
    
    print(f"All visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Visualize voxelized protein-ligand data')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset .pt file')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of sample to visualize')
    parser.add_argument('--output_dir', type=str, default='voxel_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--channels', type=int, nargs='+', default=None,
                        help='Specific channels to visualize (default: first 4 of pocket and ligand)')
    
    args = parser.parse_args()
    
    visualize_sample(
        args.data_path,
        sample_idx=args.sample_idx,
        output_dir=args.output_dir,
        show_channels=args.channels
    )


if __name__ == '__main__':
    main()
