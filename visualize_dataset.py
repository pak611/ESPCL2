"""
Quick visualization of voxelized dataset samples
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def visualize_sample(data_file, sample_idx=0, save_dir='dataset_viz'):
    """
    Visualize a sample from the voxelized dataset
    """
    print(f"Loading data from {data_file}")
    data = torch.load(data_file, map_location='cpu', weights_only=False)
    
    print(f"\nDataset keys: {data.keys()}")
    print(f"Number of samples: {len(data['labels'])}")
    
    # Get sample - check if unified or separate voxels
    if 'unified_voxels' in data:
        # Unified format: channels are interleaved [ligand_ch0-8, pocket_ch9-18]
        unified = data['unified_voxels'][sample_idx].numpy()
        print(f"Unified voxels shape: {unified.shape}")
        
        # Split into ligand and pocket based on channel count
        n_channels = unified.shape[0]
        if n_channels == 19:
            ligand_voxels = unified[:9]  # Channels 0-8
            pocket_voxels = unified[9:]  # Channels 9-18
        else:
            # Assume equal split
            mid = n_channels // 2
            ligand_voxels = unified[:mid]
            pocket_voxels = unified[mid:]
    else:
        # Separate format
        pocket_voxels = data['pocket_voxels'][sample_idx].numpy()
        ligand_voxels = data['ligand_voxels'][sample_idx].numpy()
    
    affinity = data['labels'][sample_idx].item()
    
    # Get IDs if available
    ligand_id = data.get('ligand_ids', ['unknown'] * len(data['labels']))[sample_idx]
    protein_id = data.get('protein_ids', ['unknown'] * len(data['labels']))[sample_idx]
    
    print(f"\nSample {sample_idx}:")
    print(f"Ligand: {ligand_id}, Protein: {protein_id}")
    print(f"Affinity: {affinity:.3f}")
    print(f"Pocket shape: {pocket_voxels.shape}")
    print(f"Ligand shape: {ligand_voxels.shape}")
    
    # Create output directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Find center of mass for visualization
    def get_center_slice(voxels):
        """Find the center of mass z-slice"""
        density = voxels.sum(axis=0)  # Sum over channels
        if density.sum() > 0:
            z_coords = np.where(density.sum(axis=(0, 1)) > 0)[0]
            if len(z_coords) > 0:
                return int(z_coords.mean())
        return voxels.shape[1] // 2
    
    pocket_center = get_center_slice(pocket_voxels)
    ligand_center = get_center_slice(ligand_voxels)
    
    print(f"Pocket center slice: {pocket_center}")
    print(f"Ligand center slice: {ligand_center}")
    
    # Channel names (based on common ESP feature organization)
    channel_names = {
        0: 'ESP',
        1: 'Hydrophobic',
        2: 'Donor',
        3: 'Acceptor',
        4: 'Aromatic',
        5: 'Positive',
        6: 'Negative',
        7: 'Polar',
        8: 'Other'
    }
    
    # Visualize pocket channels
    n_pocket_channels = pocket_voxels.shape[0]
    n_cols = min(5, n_pocket_channels)
    n_rows = (n_pocket_channels + n_cols - 1) // n_cols
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    fig1.suptitle(f'Pocket Channels - Sample {sample_idx} ({protein_id})\nAffinity: {affinity:.3f}', 
                  fontsize=14)
    axes1 = axes1.flatten() if n_pocket_channels > 1 else [axes1]
    
    for ch in range(n_pocket_channels):
        ax = axes1[ch]
        channel_slice = pocket_voxels[ch, :, :, pocket_center]
        im = ax.imshow(channel_slice, cmap='viridis', aspect='auto')
        ax.set_title(channel_names.get(ch, f'Ch {ch}'))
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide extra subplots
    for ch in range(n_pocket_channels, len(axes1)):
        axes1[ch].axis('off')
    
    plt.tight_layout()
    save_file1 = save_path / f'sample{sample_idx}_pocket.png'
    plt.savefig(save_file1, dpi=150, bbox_inches='tight')
    print(f"Saved pocket visualization to {save_file1}")
    
    # Visualize ligand channels
    n_ligand_channels = ligand_voxels.shape[0]
    n_cols = min(5, n_ligand_channels)
    n_rows = (n_ligand_channels + n_cols - 1) // n_cols
    
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    fig2.suptitle(f'Ligand Channels - Sample {sample_idx} ({ligand_id})\nAffinity: {affinity:.3f}', 
                  fontsize=14)
    axes2 = axes2.flatten() if n_ligand_channels > 1 else [axes2]
    
    for ch in range(n_ligand_channels):
        ax = axes2[ch]
        channel_slice = ligand_voxels[ch, :, :, ligand_center]
        im = ax.imshow(channel_slice, cmap='viridis', aspect='auto')
        ax.set_title(channel_names.get(ch, f'Ch {ch}'))
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide extra subplots
    for ch in range(n_ligand_channels, len(axes2)):
        axes2[ch].axis('off')
    
    plt.tight_layout()
    save_file2 = save_path / f'sample{sample_idx}_ligand.png'
    plt.savefig(save_file2, dpi=150, bbox_inches='tight')
    print(f"Saved ligand visualization to {save_file2}")
    
    # Create 3D visualization of channel overlap
    fig3 = plt.figure(figsize=(15, 5))
    
    # Pocket 3D projection
    ax1 = fig3.add_subplot(131, projection='3d')
    pocket_sum = pocket_voxels.sum(axis=0)
    z, y, x = np.where(pocket_sum > 0.1)
    if len(x) > 0:
        ax1.scatter(x, y, z, c=pocket_sum[z, y, x], cmap='Reds', alpha=0.3, s=1)
    ax1.set_title(f'Pocket 3D\n{protein_id}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Ligand 3D projection
    ax2 = fig3.add_subplot(132, projection='3d')
    ligand_sum = ligand_voxels.sum(axis=0)
    z, y, x = np.where(ligand_sum > 0.1)
    if len(x) > 0:
        ax2.scatter(x, y, z, c=ligand_sum[z, y, x], cmap='Blues', alpha=0.3, s=1)
    ax2.set_title(f'Ligand 3D\n{ligand_id}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Combined view
    ax3 = fig3.add_subplot(133, projection='3d')
    z, y, x = np.where(pocket_sum > 0.1)
    if len(x) > 0:
        ax3.scatter(x, y, z, c='red', alpha=0.2, s=1, label='Pocket')
    z, y, x = np.where(ligand_sum > 0.1)
    if len(x) > 0:
        ax3.scatter(x, y, z, c='blue', alpha=0.2, s=1, label='Ligand')
    ax3.set_title(f'Combined View\nAffinity: {affinity:.3f}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    fig3.suptitle(f'3D Voxel Density - Sample {sample_idx}', fontsize=16)
    plt.tight_layout()
    save_file3 = save_path / f'sample{sample_idx}_3d.png'
    plt.savefig(save_file3, dpi=150, bbox_inches='tight')
    print(f"Saved 3D visualization to {save_file3}")
    
    plt.show()
    
    # Print statistics
    print(f"\nPocket statistics:")
    print(f"  Non-zero voxels: {(pocket_voxels > 0).sum()}")
    print(f"  Mean value: {pocket_voxels.mean():.4f}")
    print(f"  Max value: {pocket_voxels.max():.4f}")
    print(f"  Per-channel non-zero counts:")
    for ch in range(n_pocket_channels):
        count = (pocket_voxels[ch] > 0).sum()
        print(f"    Ch {ch} ({channel_names.get(ch, 'Unknown')}): {count}")
    
    print(f"\nLigand statistics:")
    print(f"  Non-zero voxels: {(ligand_voxels > 0).sum()}")
    print(f"  Mean value: {ligand_voxels.mean():.4f}")
    print(f"  Max value: {ligand_voxels.max():.4f}")
    print(f"  Per-channel non-zero counts:")
    for ch in range(n_ligand_channels):
        count = (ligand_voxels[ch] > 0).sum()
        print(f"    Ch {ch} ({channel_names.get(ch, 'Unknown')}): {count}")


def main():
    parser = argparse.ArgumentParser(description='Visualize voxelized dataset samples')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to dataset .pt file')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--save-dir', type=str, default='dataset_viz',
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    visualize_sample(args.data_file, args.sample_idx, args.save_dir)


if __name__ == '__main__':
    main()
