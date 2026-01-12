"""
Create publication-quality 3D visualizations of voxelized ligands for academic papers

Usage:
    python visualize_ligand_3d.py --data-file data/bindingdb_2016/voxelized_unified_48_32_normalized.pt \
                                   --sample-idx 0 \
                                   --style isosurface \
                                   --output paper_figures/ligand_3d.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import gaussian_filter
import argparse
from pathlib import Path


# Color schemes for different molecular features
FEATURE_COLORS = {
    'ESP': '#FF6B6B',           # Red for electrostatic
    'Hydrophobic': '#4ECDC4',   # Cyan for hydrophobic
    'H-Donor': '#95E1D3',       # Light green for donor
    'H-Acceptor': '#F38181',    # Pink for acceptor
    'Aromatic': '#FFD93D',      # Yellow for aromatic
    'Positive': '#6C5CE7',      # Purple for positive
    'Negative': '#FD79A8',      # Pink for negative
    'Polar': '#74B9FF',         # Light blue for polar
}


def load_ligand_from_dataset(data_file, sample_idx=0):
    """Load a single ligand voxel grid from dataset"""
    print(f"Loading data from {data_file}")
    data = torch.load(data_file, map_location='cpu', weights_only=False)
    
    print(f"Dataset keys: {list(data.keys())}")
    print(f"Number of samples: {len(data['labels'])}")
    
    # Get sample
    if 'unified_voxels' in data:
        unified = data['unified_voxels'][sample_idx].numpy()
        n_channels = unified.shape[0]
        # Extract ligand channels (typically 0-8)
        if n_channels == 19:
            ligand_voxels = unified[:9]
        else:
            mid = n_channels // 2
            ligand_voxels = unified[:mid]
    else:
        ligand_voxels = data['ligand_voxels'][sample_idx].numpy()
    
    affinity = data['labels'][sample_idx].item()
    ligand_id = data.get('ligand_ids', [f'ligand_{sample_idx}'])[sample_idx]
    
    print(f"\nLigand: {ligand_id}")
    print(f"Affinity: {affinity:.3f}")
    print(f"Voxel shape: {ligand_voxels.shape}")
    print(f"Channels: {ligand_voxels.shape[0]}")
    
    return ligand_voxels, ligand_id, affinity


def create_multi_channel_isosurface(voxels, channel_idx, level=None, smoothing=1.5):
    """Create isosurface from a specific channel"""
    channel_data = voxels[channel_idx]
    
    if channel_data.max() == 0:
        return None, None
    
    # Smooth
    if smoothing > 0:
        channel_smooth = gaussian_filter(channel_data, sigma=smoothing)
    else:
        channel_smooth = channel_data
    
    # Threshold
    if level is None:
        non_zero = channel_smooth[channel_smooth > 0]
        if len(non_zero) == 0:
            return None, None
        level = non_zero.mean() * 0.3
    
    try:
        verts, faces, normals, values = measure.marching_cubes(
            channel_smooth, level=level, spacing=(1.0, 1.0, 1.0)
        )
        return verts, faces
    except:
        return None, None


def plot_ligand_multi_layer(voxels, ligand_id, affinity, save_path, 
                           channels_to_plot=None, figsize=(18, 12), 
                           alpha=0.8, view_angle=(30, 45), use_voxels=True,
                           threshold=0.2):
    """
    Create multi-panel visualization showing different chemical features as discrete voxels
    """
    channel_names = ['ESP', 'Hydrophobic', 'H-Donor', 'H-Acceptor', 
                    'Aromatic', 'Positive', 'Negative', 'Polar']
    
    if channels_to_plot is None:
        channels_to_plot = list(range(min(8, voxels.shape[0])))
    
    n_channels = len(channels_to_plot)
    n_cols = 4
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=figsize)
    
    for idx, ch in enumerate(channels_to_plot):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        color = FEATURE_COLORS.get(channel_names[ch], '#95A5A6')
        
        if use_voxels:
            # Voxel representation
            channel_data = voxels[ch]
            coords = np.where(channel_data > threshold)
            
            if len(coords[0]) > 0:
                z, y, x = coords
                values = channel_data[coords]
                
                # Normalize for color intensity
                values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
                
                # Color with intensity
                from matplotlib.colors import to_rgba
                rgba = to_rgba(color)
                colors_array = np.array([list(rgba[:3]) + [alpha * v] for v in values_norm])
                
                ax.scatter(x, y, z, c=colors_array, s=30, marker='s',
                          edgecolors='black', linewidths=0.1)
        else:
            # Smooth isosurface
            verts, faces = create_multi_channel_isosurface(voxels, ch, smoothing=0.5)
            
            if verts is not None and len(verts) > 0:
                mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidth=0.1)
                mesh.set_facecolor(color)
                mesh.set_edgecolor('black')
                ax.add_collection3d(mesh)
        
        # Set limits
        ax.set_xlim(0, voxels.shape[1])
        ax.set_ylim(0, voxels.shape[2])
        ax.set_zlim(0, voxels.shape[3])
        
        # Styling
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_title(channel_names[ch], fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Minimal ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    
    fig.suptitle(f'Ligand: {ligand_id} | Affinity: pKd = {affinity:.2f}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved multi-layer visualization: {save_path}")
    plt.close()


def plot_ligand_composite(voxels, ligand_id, affinity, save_path,
                         channels=[0, 1, 2, 3], colors=None,
                         alpha=0.7, figsize=(12, 10), view_angle=(25, 45),
                         use_voxels=True, threshold=0.2):
    """
    Create composite overlay of multiple channels using discrete voxel cubes
    """
    channel_names = ['ESP', 'Hydrophobic', 'H-Donor', 'H-Acceptor', 
                    'Aromatic', 'Positive', 'Negative', 'Polar']
    
    if colors is None:
        colors = [FEATURE_COLORS.get(channel_names[ch], '#95A5A6') for ch in channels]
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if use_voxels:
        # Voxel-based representation - show discrete cubes
        from matplotlib.colors import to_rgba
        
        for ch, color in zip(channels, colors):
            channel_data = voxels[ch]
            
            # Get voxel coordinates above threshold
            coords = np.where(channel_data > threshold)
            if len(coords[0]) == 0:
                continue
                
            z, y, x = coords
            values = channel_data[coords]
            
            # Normalize values for color intensity
            values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
            
            # Create color array with varying intensity
            rgba = to_rgba(color)
            colors_array = np.array([list(rgba[:3]) + [alpha * v] for v in values_norm])
            
            # Plot as 3D scatter with larger markers for cube-like appearance
            ax.scatter(x, y, z, c=colors_array, s=50, marker='s', 
                      edgecolors='black', linewidths=0.2)
    else:
        # Original smooth isosurface approach
        for ch, color in zip(channels, colors):
            verts, faces = create_multi_channel_isosurface(voxels, ch, smoothing=0.5)
            
            if verts is not None and len(verts) > 0:
                mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidth=0.1)
                mesh.set_facecolor(color)
                mesh.set_edgecolor('black')
                ax.add_collection3d(mesh)
    
    # Set limits
    ax.set_xlim(0, voxels.shape[1])
    ax.set_ylim(0, voxels.shape[2])
    ax.set_zlim(0, voxels.shape[3])
    
    # Styling
    ax.set_xlabel('X (Å)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (Å)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (Å)', fontsize=12, labelpad=10)
    ax.set_title(f'{ligand_id}\nBinding Affinity: pKd = {affinity:.2f}', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.2, linewidth=0.8)
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Clean axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=alpha, 
                            label=channel_names[channels[i]]) 
                      for i in range(len(channels))]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
             framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved composite visualization: {save_path}")
    plt.close()


def plot_ligand_voxel_scatter(voxels, ligand_id, affinity, save_path,
                              channel=0, threshold=0.1, figsize=(10, 10),
                              cmap='viridis', view_angle=(25, 45)):
    """
    Create scatter plot representation of voxels
    """
    channel_names = ['ESP', 'Hydrophobic', 'H-Donor', 'H-Acceptor', 
                    'Aromatic', 'Positive', 'Negative', 'Polar']
    
    channel_data = voxels[channel]
    
    # Get non-zero voxels
    coords = np.where(channel_data > threshold)
    if len(coords[0]) == 0:
        print(f"No voxels above threshold {threshold} for channel {channel}")
        return
    
    z, y, x = coords
    values = channel_data[coords]
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(x, y, z, c=values, cmap=cmap, alpha=0.6, 
                        s=20, edgecolors='none')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Intensity', fontsize=11)
    
    # Styling
    ax.set_xlabel('X (Å)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (Å)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (Å)', fontsize=12, labelpad=10)
    ax.set_title(f'{ligand_id} - {channel_names[channel]}\npKd = {affinity:.2f}', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved voxel scatter plot: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create 3D visualizations of voxelized ligands')
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to dataset .pt file')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--output-dir', type=str, default='paper_figures',
                       help='Output directory for figures')
    parser.add_argument('--style', type=str, default='all',
                       choices=['all', 'multi-layer', 'composite', 'scatter'],
                       help='Visualization style')
    parser.add_argument('--channels', type=int, nargs='+', default=None,
                       help='Channels to visualize (default: 0-7)')
    parser.add_argument('--view-elevation', type=int, default=25,
                       help='Camera elevation angle')
    parser.add_argument('--view-azimuth', type=int, default=45,
                       help='Camera azimuth angle')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Voxel transparency (0-1)')
    parser.add_argument('--use-smooth', action='store_true',
                       help='Use smooth isosurfaces instead of discrete voxels')
    parser.add_argument('--threshold', type=float, default=0.2,
                       help='Voxel intensity threshold (0-1)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load ligand
    voxels, ligand_id, affinity = load_ligand_from_dataset(args.data_file, args.sample_idx)
    
    view_angle = (args.view_elevation, args.view_azimuth)
    
    # Generate visualizations
    use_voxels = not args.use_smooth
    
    if args.style in ['all', 'multi-layer']:
        save_path = output_dir / f'ligand_{ligand_id}_multi_layer.png'
        plot_ligand_multi_layer(voxels, ligand_id, affinity, save_path,
                               channels_to_plot=args.channels,
                               alpha=args.alpha, view_angle=view_angle,
                               use_voxels=use_voxels, threshold=args.threshold)
    
    if args.style in ['all', 'composite']:
        channels = args.channels if args.channels else [0, 1, 2, 3]
        save_path = output_dir / f'ligand_{ligand_id}_composite.png'
        plot_ligand_composite(voxels, ligand_id, affinity, save_path,
                             channels=channels, alpha=args.alpha,
                             view_angle=view_angle, use_voxels=use_voxels,
                             threshold=args.threshold)
    
    if args.style in ['all', 'scatter']:
        save_path = output_dir / f'ligand_{ligand_id}_scatter.png'
        plot_ligand_voxel_scatter(voxels, ligand_id, affinity, save_path,
                                 channel=0, view_angle=view_angle)
    
    print(f"\n✓ All visualizations completed!")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
