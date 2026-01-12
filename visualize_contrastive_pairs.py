"""
Visualize positive and negative contrastive samples for ligand-pocket pairs

This script loads a ligand-pocket pair and shows:
1. Positive sample: Original ligand + pocket (true binding pair)
2. Negative sample: Corrupted ligand/pocket (chemically impossible binding)

Usage:
    python visualize_contrastive_pairs.py --data-file data/bindingdb_2016/voxelized_unified_48_32_normalized.pt \
                                          --sample-idx 0 \
                                          --output pair_visualizations/contrastive_example.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from train import create_chemical_negatives


def visualize_single_channel(voxels, ax, title, color='cyan', alpha=0.5):
    """Visualize a single channel as scatter plot of high-intensity voxels"""
    if len(voxels.shape) == 3:  # [H, W, D]
        data = np.abs(voxels)
    else:
        data = np.abs(voxels)
    
    # Get threshold for top voxels
    threshold = np.percentile(data[data > 0], 70) if np.any(data > 0) else 0
    
    if threshold > 0:
        points = np.argwhere(data > threshold)
        values = data[data > threshold]
        
        if len(points) > 0:
            # Color by intensity
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=values, cmap='hot', alpha=alpha, s=10,
                               vmin=values.min(), vmax=values.max())
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.grid(False)
    ax.set_box_aspect([1,1,1])
    
    # Set limits
    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(0, data.shape[1])
    ax.set_zlim(0, data.shape[2])


def visualize_channel_comparison(original, corrupted, channel_names, title):
    """Create bar plot comparing channel intensities"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Calculate mean absolute value per channel
    orig_means = [np.abs(original[i]).mean() for i in range(len(channel_names))]
    corr_means = [np.abs(corrupted[i]).mean() for i in range(len(channel_names))]
    
    x = np.arange(len(channel_names))
    width = 0.35
    
    ax.bar(x - width/2, orig_means, width, label='Original', color='green', alpha=0.7)
    ax.bar(x + width/2, corr_means, width, label='Corrupted', color='red', alpha=0.7)
    
    ax.set_xlabel('Channel')
    ax.set_ylabel('Mean Absolute Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize positive and negative contrastive samples')
    parser.add_argument('--data-file', required=True, help='Path to voxelized dataset')
    parser.add_argument('--sample-idx', type=int, default=0, help='Index of sample to visualize')
    parser.add_argument('--output', default='pair_visualizations/contrastive_example.png',
                       help='Output file path')
    parser.add_argument('--corruption-type', type=int, default=None,
                       help='Specific corruption type (0-4), or random if not specified')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset: {args.data_file}")
    data = torch.load(args.data_file, map_location='cpu', weights_only=False)
    
    print(f"Dataset keys: {list(data.keys())}")
    
    # Get sample
    if 'unified_voxels' in data:
        unified = data['unified_voxels'][args.sample_idx].unsqueeze(0)  # Add batch dim
        n_channels = unified.shape[1]
        
        # Split into pocket and ligand
        if n_channels == 30:  # 22 pocket + 8 ligand
            pocket_esp = unified[:, :22]
            ligand_esp = unified[:, 22:]
            pocket_channels = ['ESP', 'Hydrophobic', 'Donor', 'Acceptor', 'Aromatic', 
                             'Pos', 'Neg', 'Polar'] + [f'P{i}' for i in range(14)]
            ligand_channels = ['ESP', 'Hydrophobic', 'Donor', 'Acceptor', 
                             'Aromatic', 'Pos', 'Neg', 'Polar']
        elif n_channels == 19:  # 11 pocket + 8 ligand
            pocket_esp = unified[:, :11]
            ligand_esp = unified[:, 11:]
            pocket_channels = ['P_ESP', 'P_Hydrophobic', 'P_Charged', 'P_Aromatic',
                             'P_Pos', 'P_Neg', 'P_Polar', 'P_Donor', 'P_Acceptor',
                             'P_Metal', 'P_Sulfur']
            ligand_channels = ['L_ESP', 'L_Hydrophobic', 'L_Donor', 'L_Acceptor',
                             'L_Aromatic', 'L_Pos', 'L_Neg', 'L_Polar']
        else:
            print(f"Warning: Expected 30 channels, got {n_channels}")
            # Assume split at midpoint
            mid = n_channels // 2
            pocket_esp = unified[:, :mid]
            ligand_esp = unified[:, mid:]
            pocket_channels = [f'P{i}' for i in range(mid)]
            ligand_channels = [f'L{i}' for i in range(n_channels - mid)]
    else:
        print("Error: 'unified_voxels' not found in dataset")
        return
    
    # Get label info
    label = data['labels'][args.sample_idx].item()
    protein_id = data.get('protein_ids', ['Unknown'])[args.sample_idx]
    ligand_id = data.get('ligand_ids', ['Unknown'])[args.sample_idx]
    
    print(f"\nSample {args.sample_idx}:")
    print(f"  Protein: {protein_id}")
    print(f"  Ligand: {ligand_id}")
    print(f"  Label (pKd): {label:.2f}")
    print(f"  Pocket shape: {pocket_esp.shape}")
    print(f"  Ligand shape: {ligand_esp.shape}")
    
    # Create negative sample
    if args.corruption_type is not None:
        # Force specific corruption
        torch.manual_seed(args.corruption_type)
    
    corrupted_pocket, corrupted_ligand, mask = create_chemical_negatives(
        pocket_esp, ligand_esp, corruption_rate=1.0
    )
    
    # Convert to numpy
    pocket_orig = pocket_esp[0].numpy()
    ligand_orig = ligand_esp[0].numpy()
    pocket_corr = corrupted_pocket[0].numpy()
    ligand_corr = corrupted_ligand[0].numpy()
    
    # Determine corruption type and affected channels
    corruption_desc = "Unknown"
    affected_channel_idx = None
    is_pocket_affected = False
    
    if np.allclose(pocket_corr[0], -pocket_orig[0]):
        corruption_desc = "ESP sign flip (destroys electrostatics)"
        affected_channel_idx = 0
        is_pocket_affected = True
    elif np.allclose(pocket_corr[1], 0) and not np.allclose(pocket_orig[1], 0):
        corruption_desc = "Zeroed hydrophobic channel"
        affected_channel_idx = 1
        is_pocket_affected = True
    elif not np.allclose(pocket_corr[1], pocket_orig[1]) and not np.allclose(pocket_corr[2], pocket_orig[2]):
        corruption_desc = "Swapped hydrophilic/hydrophobic channels"
        affected_channel_idx = 1
        is_pocket_affected = True
    elif not np.allclose(ligand_corr[2:4], ligand_orig[2:4]):
        corruption_desc = "Removed H-bond donors/acceptors"
        affected_channel_idx = 2
        is_pocket_affected = False
    else:
        # Check for zeroed ligand channels
        for i in range(ligand_corr.shape[0]):
            if np.allclose(ligand_corr[i], 0) and not np.allclose(ligand_orig[i], 0):
                corruption_desc = f"Zeroed ligand {ligand_channels[i]}"
                affected_channel_idx = i
                is_pocket_affected = False
                break
    
    print(f"\nCorruption applied: {corruption_desc}")
    if affected_channel_idx is not None:
        print(f"Affected channel: {affected_channel_idx} ({'Pocket' if is_pocket_affected else 'Ligand'})")
    
    # Create visualization - focus on affected channel
    if affected_channel_idx is not None:
        fig = plt.figure(figsize=(20, 12))
        
        # Show the specific affected channel in detail
        if is_pocket_affected:
            affected_orig = pocket_orig[affected_channel_idx]
            affected_corr = pocket_corr[affected_channel_idx]
            ch_name = pocket_channels[affected_channel_idx]
            other_orig = ligand_orig
            other_corr = ligand_corr
            other_name = "Ligand"
            other_channels = ligand_channels
        else:
            affected_orig = ligand_orig[affected_channel_idx]
            affected_corr = ligand_corr[affected_channel_idx]
            ch_name = ligand_channels[affected_channel_idx]
            other_orig = pocket_orig
            other_corr = pocket_corr
            other_name = "Pocket"
            other_channels = pocket_channels
        
        # Row 1: Affected channel before/after
        ax1 = fig.add_subplot(3, 4, 1, projection='3d')
        visualize_single_channel(affected_orig, ax1, 
                                f'ORIGINAL: {ch_name}\n(Before Corruption)',
                                color='green', alpha=0.6)
        
        ax2 = fig.add_subplot(3, 4, 2, projection='3d')
        visualize_single_channel(affected_corr, ax2,
                                f'CORRUPTED: {ch_name}\n(After Corruption)',
                                color='red', alpha=0.6)
        
        # Show ESP channel for context (most important)
        if is_pocket_affected and affected_channel_idx != 0:
            ax3 = fig.add_subplot(3, 4, 3, projection='3d')
            visualize_single_channel(pocket_orig[0], ax3,
                                    f'Context: Pocket ESP\n(Unchanged)',
                                    color='blue', alpha=0.6)
        elif not is_pocket_affected and affected_channel_idx != 0:
            ax3 = fig.add_subplot(3, 4, 3, projection='3d')
            visualize_single_channel(ligand_orig[0], ax3,
                                    f'Context: Ligand ESP\n(Unchanged)',
                                    color='blue', alpha=0.6)
        
        # Difference map
        ax4 = fig.add_subplot(3, 4, 4, projection='3d')
        diff = np.abs(affected_orig - affected_corr)
        visualize_single_channel(diff, ax4,
                                f'DIFFERENCE MAP\n(Shows Corruption)',
                                color='magenta', alpha=0.7)
        
        # Row 2: Show top 3 channels from each molecule
        for i in range(3):
            # Original channels
            ax_orig = fig.add_subplot(3, 4, 5 + i, projection='3d')
            if is_pocket_affected and i < len(pocket_channels):
                visualize_single_channel(pocket_orig[i], ax_orig,
                                        f'Pocket: {pocket_channels[i]}\n(Original)',
                                        alpha=0.5)
            elif not is_pocket_affected and i < len(ligand_channels):
                visualize_single_channel(ligand_orig[i], ax_orig,
                                        f'Ligand: {ligand_channels[i]}\n(Original)',
                                        alpha=0.5)
        
        # Corrupted version
        ax_corr = fig.add_subplot(3, 4, 8, projection='3d')
        if is_pocket_affected and affected_channel_idx < len(pocket_channels):
            visualize_single_channel(pocket_corr[affected_channel_idx], ax_corr,
                                    f'CORRUPTED\n{pocket_channels[affected_channel_idx]}',
                                    color='red', alpha=0.6)
        elif not is_pocket_affected and affected_channel_idx < len(ligand_channels):
            visualize_single_channel(ligand_corr[affected_channel_idx], ax_corr,
                                    f'CORRUPTED\n{ligand_channels[affected_channel_idx]}',
                                    color='red', alpha=0.6)
        
        # Row 3: Channel intensity comparisons
        ax5 = fig.add_subplot(3, 2, 5)
        if is_pocket_affected:
            data_orig = pocket_orig
            data_corr = pocket_corr
            ch_labels = pocket_channels
        else:
            data_orig = ligand_orig
            data_corr = ligand_corr
            ch_labels = ligand_channels
        
        orig_means = [np.abs(data_orig[i]).mean() for i in range(len(ch_labels))]
        corr_means = [np.abs(data_corr[i]).mean() for i in range(len(ch_labels))]
        x = np.arange(len(ch_labels))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, orig_means, width, label='Original', color='green', alpha=0.7)
        bars2 = ax5.bar(x + width/2, corr_means, width, label='Corrupted', color='red', alpha=0.7)
        
        # Highlight affected channel
        if affected_channel_idx < len(ch_labels):
            bars1[affected_channel_idx].set_edgecolor('black')
            bars1[affected_channel_idx].set_linewidth(3)
            bars2[affected_channel_idx].set_edgecolor('black')
            bars2[affected_channel_idx].set_linewidth(3)
        
        ax5.set_xlabel('Channel', fontsize=12)
        ax5.set_ylabel('Mean Absolute Value', fontsize=12)
        ax5.set_title(f'{"Pocket" if is_pocket_affected else "Ligand"} Channel Intensities\n(Black outline = corrupted)', fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(ch_labels, rotation=45, ha='right')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # Statistics panel
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.axis('off')
        
        stats_text = f"""
CORRUPTION STATISTICS

Type: {corruption_desc}
Affected: {"Pocket" if is_pocket_affected else "Ligand"} Channel {affected_channel_idx}
Channel Name: {ch_name}

Original Channel Stats:
  Mean: {np.abs(affected_orig).mean():.4f}
  Max: {np.abs(affected_orig).max():.4f}
  Non-zero voxels: {np.count_nonzero(affected_orig)}

Corrupted Channel Stats:
  Mean: {np.abs(affected_corr).mean():.4f}
  Max: {np.abs(affected_corr).max():.4f}
  Non-zero voxels: {np.count_nonzero(affected_corr)}

Change:
  Mean difference: {np.abs(affected_orig).mean() - np.abs(affected_corr).mean():.4f}
  Total voxels lost: {np.count_nonzero(affected_orig) - np.count_nonzero(affected_corr)}
        """
        
        ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    else:
        # Fallback: show error message
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, 'Could not determine corruption type', 
                ha='center', va='center', fontsize=16)
        ax.axis('off')
    
    # Add overall title
    fig.suptitle(f'Contrastive Sample Visualization\n'
                f'Protein: {protein_id} | Ligand: {ligand_id} | pKd: {label:.2f}\n'
                f'Corruption: {corruption_desc}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {args.output}")
    plt.close()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
