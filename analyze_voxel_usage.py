"""
Analyze voxel occupancy to determine optimal cropping strategy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm


def analyze_bounding_boxes(data_file, num_samples=100):
    """
    Analyze the distribution of occupied voxels to determine optimal cropping
    """
    print(f"Loading data from {data_file}")
    data = torch.load(data_file, map_location='cpu', weights_only=False)
    
    total_samples = len(data['labels'])
    num_samples = min(num_samples, total_samples)
    
    print(f"Analyzing {num_samples} samples from {total_samples} total")
    
    # Track bounding boxes
    bounding_boxes = []
    occupancy_ratios = []
    
    for idx in tqdm(range(num_samples), desc="Analyzing samples"):
        # Get unified voxels
        unified = data['unified_voxels'][idx].numpy()
        
        # Find occupied region (any channel has non-zero values)
        occupied = unified.sum(axis=0) > 0  # [H, W, D]
        
        if occupied.sum() == 0:
            print(f"Warning: Sample {idx} has no occupied voxels")
            continue
        
        # Find bounding box
        z_coords, y_coords, x_coords = np.where(occupied)
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()
        
        # Calculate bounding box size
        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1
        z_size = z_max - z_min + 1
        
        # Calculate occupancy ratio
        total_voxels = unified.shape[1] * unified.shape[2] * unified.shape[3]
        occupied_voxels = occupied.sum()
        occupancy_ratio = occupied_voxels / total_voxels
        
        bounding_boxes.append({
            'x_min': x_min, 'x_max': x_max, 'x_size': x_size,
            'y_min': y_min, 'y_max': y_max, 'y_size': y_size,
            'z_min': z_min, 'z_max': z_max, 'z_size': z_size,
            'volume': x_size * y_size * z_size,
            'occupied': occupied_voxels,
        })
        occupancy_ratios.append(occupancy_ratio)
    
    bounding_boxes = np.array([list(bb.values()) for bb in bounding_boxes])
    
    # Print statistics
    print("\n" + "="*60)
    print("BOUNDING BOX STATISTICS")
    print("="*60)
    
    original_size = unified.shape[1:]
    print(f"\nOriginal grid size: {original_size}")
    print(f"Original volume: {np.prod(original_size)} voxels")
    
    x_sizes = bounding_boxes[:, 2]
    y_sizes = bounding_boxes[:, 5]
    z_sizes = bounding_boxes[:, 8]
    volumes = bounding_boxes[:, 9]
    
    print(f"\nBounding box dimensions:")
    print(f"  X: min={x_sizes.min():.0f}, max={x_sizes.max():.0f}, mean={x_sizes.mean():.1f}, median={np.median(x_sizes):.1f}")
    print(f"  Y: min={y_sizes.min():.0f}, max={y_sizes.max():.0f}, mean={y_sizes.mean():.1f}, median={np.median(y_sizes):.1f}")
    print(f"  Z: min={z_sizes.min():.0f}, max={z_sizes.max():.0f}, mean={z_sizes.mean():.1f}, median={np.median(z_sizes):.1f}")
    
    print(f"\nBounding box volume:")
    print(f"  Min: {volumes.min():.0f} voxels")
    print(f"  Max: {volumes.max():.0f} voxels")
    print(f"  Mean: {volumes.mean():.1f} voxels")
    print(f"  Median: {np.median(volumes):.1f} voxels")
    
    print(f"\nOccupancy ratios:")
    print(f"  Min: {np.min(occupancy_ratios):.4f} ({np.min(occupancy_ratios)*100:.2f}%)")
    print(f"  Max: {np.max(occupancy_ratios):.4f} ({np.max(occupancy_ratios)*100:.2f}%)")
    print(f"  Mean: {np.mean(occupancy_ratios):.4f} ({np.mean(occupancy_ratios)*100:.2f}%)")
    print(f"  Median: {np.median(occupancy_ratios):.4f} ({np.median(occupancy_ratios)*100:.2f}%)")
    
    # Suggest optimal crop size
    percentiles = [75, 90, 95, 99]
    print(f"\nSuggested crop sizes (to fit X% of samples):")
    for p in percentiles:
        x_p = np.percentile(x_sizes, p)
        y_p = np.percentile(y_sizes, p)
        z_p = np.percentile(z_sizes, p)
        max_dim = max(x_p, y_p, z_p)
        
        # Round up to nearest power of 2 or multiple of 4
        crop_size = int(np.ceil(max_dim / 4) * 4)
        
        # Calculate volume reduction
        new_volume = crop_size ** 3
        old_volume = np.prod(original_size)
        reduction = (1 - new_volume / old_volume) * 100
        
        print(f"  {p}th percentile: {crop_size}×{crop_size}×{crop_size} "
              f"(max_dim={max_dim:.1f}, reduction={reduction:.1f}%)")
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Box size distributions
    ax = axes[0, 0]
    ax.hist(x_sizes, bins=20, alpha=0.5, label='X', color='red')
    ax.hist(y_sizes, bins=20, alpha=0.5, label='Y', color='green')
    ax.hist(z_sizes, bins=20, alpha=0.5, label='Z', color='blue')
    ax.axvline(original_size[0], color='black', linestyle='--', label='Original size')
    ax.set_xlabel('Bounding Box Dimension')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Bounding Box Dimensions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Volume distribution
    ax = axes[0, 1]
    ax.hist(volumes, bins=30, alpha=0.7, color='purple')
    ax.axvline(np.prod(original_size), color='black', linestyle='--', label='Original volume')
    ax.set_xlabel('Bounding Box Volume (voxels)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Bounding Box Volumes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Occupancy ratio
    ax = axes[1, 0]
    ax.hist(occupancy_ratios, bins=30, alpha=0.7, color='orange')
    ax.set_xlabel('Occupancy Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Voxel Occupancy')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Correlation between dimensions
    ax = axes[1, 1]
    max_dims = np.maximum(np.maximum(x_sizes, y_sizes), z_sizes)
    ax.scatter(max_dims, volumes, alpha=0.5)
    ax.set_xlabel('Max Dimension')
    ax.set_ylabel('Bounding Box Volume')
    ax.set_title('Max Dimension vs Volume')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('voxel_occupancy_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis plot to voxel_occupancy_analysis.png")
    
    return bounding_boxes, occupancy_ratios


def main():
    parser = argparse.ArgumentParser(description='Analyze voxel occupancy in dataset')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to dataset .pt file')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    analyze_bounding_boxes(args.data_file, args.num_samples)


if __name__ == '__main__':
    main()
