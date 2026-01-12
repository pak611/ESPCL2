"""
Create publication-quality 3D visualizations of ligand-pocket features for academic papers
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import argparse
from pathlib import Path


# Channel mappings based on unified dataset format
CHANNEL_NAMES = {
    # Ligand channels (0-8)
    'ligand': {
        0: 'ESP',
        1: 'Hydrophobic',
        2: 'H-Donor',
        3: 'H-Acceptor',
        4: 'Aromatic',
        5: 'Positive',
        6: 'Negative',
        7: 'Polar',
        8: 'Moltype'
    },
    # Pocket channels (9-18)
    'pocket': {
        9: 'ESP',
        10: 'Hydrophobic',
        11: 'H-Donor',
        12: 'H-Acceptor',
        13: 'Aromatic',
        14: 'Positive',
        15: 'Negative',
        16: 'Polar',
        17: 'Backbone',
        18: 'Moltype'
    }
}


def create_isosurface(voxels, level=None, smoothing=1):
    """
    Create 3D isosurface from voxel data
    
    Args:
        voxels: 3D array
        level: isosurface threshold (if None, uses mean of non-zero values)
        smoothing: Gaussian smoothing factor
    
    Returns:
        verts, faces, normals, values
    """
    from scipy.ndimage import gaussian_filter
    
    if voxels.max() == 0:
        return None, None, None, None
    
    # Smooth the data
    if smoothing > 0:
        voxels_smooth = gaussian_filter(voxels, sigma=smoothing)
    else:
        voxels_smooth = voxels
    
    # Determine threshold
    if level is None:
        non_zero = voxels_smooth[voxels_smooth > 0]
        if len(non_zero) == 0:
            return None, None, None, None
        level = non_zero.mean() * 0.5  # Use 50% of mean
    
    # Generate isosurface using marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(
            voxels_smooth, level=level, spacing=(1.0, 1.0, 1.0)
        )
        return verts, faces, normals, values
    except:
        return None, None, None, None


def plot_dual_3d_isosurfaces(ligand_voxels, pocket_voxels, 
                             ligand_name, pocket_name,
                             title, save_path,
                             level=None, alpha=0.4):
    """
    Plot ligand and pocket isosurfaces side by side
    """
    fig = plt.figure(figsize=(16, 7))
    
    # Ligand plot
    ax1 = fig.add_subplot(121, projection='3d')
    verts, faces, normals, values = create_isosurface(ligand_voxels, level=level)
    
    if verts is not None and len(verts) > 0:
        mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidth=0.1, edgecolor='black')
        mesh.set_facecolor('blue')
        ax1.add_collection3d(mesh)
        
        # Set axis limits
        ax1.set_xlim(0, ligand_voxels.shape[0])
        ax1.set_ylim(0, ligand_voxels.shape[1])
        ax1.set_zlim(0, ligand_voxels.shape[2])
    
    ax1.set_xlabel('X (Å)', fontsize=10)
    ax1.set_ylabel('Y (Å)', fontsize=10)
    ax1.set_zlabel('Z (Å)', fontsize=10)
    ax1.set_title(f'Ligand - {ligand_name}', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=20, azim=45)
    
    # Pocket plot
    ax2 = fig.add_subplot(122, projection='3d')
    verts, faces, normals, values = create_isosurface(pocket_voxels, level=level)
    
    if verts is not None and len(verts) > 0:
        mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidth=0.1, edgecolor='black')
        mesh.set_facecolor('red')
        ax2.add_collection3d(mesh)
        
        # Set axis limits
        ax2.set_xlim(0, pocket_voxels.shape[0])
        ax2.set_ylim(0, pocket_voxels.shape[1])
        ax2.set_zlim(0, pocket_voxels.shape[2])
    
    ax2.set_xlabel('X (Å)', fontsize=10)
    ax2.set_ylabel('Y (Å)', fontsize=10)
    ax2.set_zlabel('Z (Å)', fontsize=10)
    ax2.set_title(f'Pocket - {pocket_name}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.view_init(elev=20, azim=45)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def plot_dual_3d_scatter(ligand_voxels, pocket_voxels,
                        ligand_name, pocket_name,
                        title, save_path,
                        threshold=0.1, alpha=0.3, s=1):
    """
    Plot ligand and pocket as 3D scatter plots side by side
    """
    fig = plt.figure(figsize=(16, 7))
    
    # Ligand plot
    ax1 = fig.add_subplot(121, projection='3d')
    z, y, x = np.where(ligand_voxels > threshold)
    if len(x) > 0:
        colors = ligand_voxels[z, y, x]
        scatter = ax1.scatter(x, y, z, c=colors, cmap='Blues', alpha=alpha, s=s, marker='o')
        plt.colorbar(scatter, ax=ax1, shrink=0.5, label='Intensity')
    
    ax1.set_xlabel('X (Å)', fontsize=10)
    ax1.set_ylabel('Y (Å)', fontsize=10)
    ax1.set_zlabel('Z (Å)', fontsize=10)
    ax1.set_title(f'Ligand - {ligand_name}', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=20, azim=45)
    
    # Pocket plot
    ax2 = fig.add_subplot(122, projection='3d')
    z, y, x = np.where(pocket_voxels > threshold)
    if len(x) > 0:
        colors = pocket_voxels[z, y, x]
        scatter = ax2.scatter(x, y, z, c=colors, cmap='Reds', alpha=alpha, s=s, marker='o')
        plt.colorbar(scatter, ax=ax2, shrink=0.5, label='Intensity')
    
    ax2.set_xlabel('X (Å)', fontsize=10)
    ax2.set_ylabel('Y (Å)', fontsize=10)
    ax2.set_zlabel('Z (Å)', fontsize=10)
    ax2.set_title(f'Pocket - {pocket_name}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.view_init(elev=20, azim=45)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def create_paper_visualizations(data_file, sample_idx=0, output_dir='paper_figures', 
                                use_isosurface=True, channels_to_plot=None):
    """
    Create publication-quality visualizations of all feature channels
    """
    print(f"Loading data from {data_file}")
    data = torch.load(data_file, map_location='cpu', weights_only=False)
    
    # Get sample
    unified = data['unified_voxels'][sample_idx].numpy()
    ligand_id = data.get('ligand_ids', ['unknown'])[sample_idx]
    protein_id = data.get('protein_ids', ['unknown'])[sample_idx]
    affinity = data['labels'][sample_idx].item()
    
    print(f"\nSample {sample_idx}: Ligand={ligand_id}, Protein={protein_id}, Affinity={affinity:.3f}")
    print(f"Unified voxels shape: {unified.shape}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Determine channels to plot
    if channels_to_plot is None:
        # Default: plot functional channels (skip moltype)
        channels_to_plot = [
            (0, 9, 'ESP'),
            (1, 10, 'Hydrophobic'),
            (2, 11, 'H-Donor'),
            (3, 12, 'H-Acceptor'),
            (4, 13, 'Aromatic'),
            (5, 14, 'Positive'),
            (6, 15, 'Negative'),
            (7, 16, 'Polar'),
        ]
    
    # Create visualizations for each channel pair
    for lig_ch, pock_ch, name in channels_to_plot:
        ligand_voxels = unified[lig_ch]
        pocket_voxels = unified[pock_ch]
        
        # Check if channels have data
        lig_nonzero = (ligand_voxels > 0).sum()
        pock_nonzero = (pocket_voxels > 0).sum()
        
        print(f"\n{name}:")
        print(f"  Ligand: {lig_nonzero} non-zero voxels")
        print(f"  Pocket: {pock_nonzero} non-zero voxels")
        
        if lig_nonzero == 0 and pock_nonzero == 0:
            print(f"  Skipping (no data)")
            continue
        
        # Create filename
        safe_name = name.replace('-', '').replace(' ', '_').lower()
        save_path = output_path / f'sample{sample_idx}_{safe_name}.png'
        
        title = f'{name} Features - {ligand_id} + {protein_id} (Affinity: {affinity:.2f})'
        
        if use_isosurface:
            plot_dual_3d_isosurfaces(
                ligand_voxels, pocket_voxels,
                name, name,
                title, save_path,
                alpha=0.6
            )
        else:
            plot_dual_3d_scatter(
                ligand_voxels, pocket_voxels,
                name, name,
                title, save_path,
                threshold=0.1, alpha=0.4, s=3
            )
    
    # Create combined overview figure
    create_multi_panel_figure(unified, ligand_id, protein_id, affinity, 
                             output_path / f'sample{sample_idx}_overview.png')
    
    print(f"\nAll visualizations saved to {output_dir}/")


def create_multi_panel_figure(unified, ligand_id, protein_id, affinity, save_path):
    """
    Create a multi-panel figure showing all channels in a grid
    """
    # Compute atom occupancy (sum of all functional channels)
    ligand_occupancy = unified[0:8].sum(axis=0)  # Sum channels 0-7 (exclude moltype)
    pocket_occupancy = unified[9:17].sum(axis=0)  # Sum channels 9-16 (exclude moltype and backbone)
    
    # Select key channels to display
    channels = [
        ('occupancy', 'occupancy', 'Atom Occupancy'),  # Special case for occupancy
        (0, 9, 'ESP'),
        (1, 10, 'Hydrophobic'),
        (2, 11, 'H-Donor'),
        (3, 12, 'H-Acceptor'),
    ]
    
    fig = plt.figure(figsize=(16, 15))
    
    for idx, (lig_ch, pock_ch, name) in enumerate(channels):
        # Handle special case for occupancy
        if lig_ch == 'occupancy':
            ligand_voxels = ligand_occupancy
            pocket_voxels = pocket_occupancy
        else:
            ligand_voxels = unified[lig_ch]
            pocket_voxels = unified[pock_ch]
        
        # Ligand - larger scatter points
        ax = fig.add_subplot(5, 4, idx*4 + 1, projection='3d')
        z, y, x = np.where(ligand_voxels > 0.05)
        if len(x) > 0:
            colors = ligand_voxels[z, y, x]
            ax.scatter(x, y, z, c=colors, cmap='Blues', alpha=0.8, s=40, marker='o', edgecolors='none')
        ax.set_title(f'Ligand {name}', fontsize=10)
        ax.view_init(elev=20, azim=45)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Pocket - larger scatter points
        ax = fig.add_subplot(5, 4, idx*4 + 2, projection='3d')
        z, y, x = np.where(pocket_voxels > 0.05)
        if len(x) > 0:
            colors = pocket_voxels[z, y, x]
            ax.scatter(x, y, z, c=colors, cmap='Reds', alpha=0.8, s=40, marker='o', edgecolors='none')
        ax.set_title(f'Pocket {name}', fontsize=10)
        ax.view_init(elev=20, azim=45)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Combined view 1 (angle 1) - larger points
        ax = fig.add_subplot(5, 4, idx*4 + 3, projection='3d')
        z, y, x = np.where(ligand_voxels > 0.05)
        if len(x) > 0:
            ax.scatter(x, y, z, c='blue', alpha=0.6, s=30, marker='o', edgecolors='none')
        z, y, x = np.where(pocket_voxels > 0.05)
        if len(x) > 0:
            ax.scatter(x, y, z, c='red', alpha=0.6, s=30, marker='o', edgecolors='none')
        ax.set_title(f'Combined {name}', fontsize=10)
        ax.view_init(elev=20, azim=45)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Combined view 2 (angle 2) - larger points
        ax = fig.add_subplot(5, 4, idx*4 + 4, projection='3d')
        z, y, x = np.where(ligand_voxels > 0.05)
        if len(x) > 0:
            ax.scatter(x, y, z, c='blue', alpha=0.6, s=30, marker='o', edgecolors='none')
        z, y, x = np.where(pocket_voxels > 0.05)
        if len(x) > 0:
            ax.scatter(x, y, z, c='red', alpha=0.6, s=30, marker='o', edgecolors='none')
        ax.set_title(f'Combined {name} (rotated)', fontsize=10)
        ax.view_init(elev=20, azim=135)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    fig.suptitle(f'{ligand_id} + {protein_id} - Affinity: {affinity:.2f} pKd', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved overview: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create publication-quality 3D visualizations')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to dataset .pt file')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--output-dir', type=str, default='paper_figures',
                        help='Output directory for figures')
    parser.add_argument('--use-scatter', action='store_true',
                        help='Use scatter plots instead of isosurfaces')
    
    args = parser.parse_args()
    
    create_paper_visualizations(
        args.data_file,
        args.sample_idx,
        args.output_dir,
        use_isosurface=not args.use_scatter
    )


if __name__ == '__main__':
    main()
