"""
Create a cropped version of the dataset by trimming empty voxels
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def crop_to_content(voxels, target_size=None, padding=2):
    """
    Crop voxels to bounding box of content with optional padding
    
    Args:
        voxels: [C, H, W, D] voxel grid
        target_size: if specified, crop to this cubic size (centered on content)
        padding: voxels of padding around content bounding box
    
    Returns:
        cropped_voxels: cropped grid
        crop_info: dict with crop parameters for reference
    """
    # Find occupied region
    occupied = voxels.sum(axis=0) > 0
    
    if occupied.sum() == 0:
        # No content - return center crop
        C, H, W, D = voxels.shape
        if target_size:
            start = [(s - target_size) // 2 for s in [H, W, D]]
            end = [s + target_size for s in start]
            return voxels[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]], None
        return voxels, None
    
    # Find bounding box
    z_coords, y_coords, x_coords = np.where(occupied)
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()
    
    # Add padding
    C, H, W, D = voxels.shape
    x_min = max(0, x_min - padding)
    x_max = min(W - 1, x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(H - 1, y_max + padding)
    z_min = max(0, z_min - padding)
    z_max = min(D - 1, z_max + padding)
    
    if target_size:
        # Calculate current size
        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1
        z_size = z_max - z_min + 1
        
        # Find center of content
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        z_center = (z_min + z_max) // 2
        
        # Crop centered on content with target size
        half_size = target_size // 2
        
        x_start = max(0, min(W - target_size, x_center - half_size))
        y_start = max(0, min(H - target_size, y_center - half_size))
        z_start = max(0, min(D - target_size, z_center - half_size))
        
        x_end = x_start + target_size
        y_end = y_start + target_size
        z_end = z_start + target_size
        
        # Handle edge cases where we can't fit target_size
        if x_end > W:
            x_end = W
            x_start = max(0, W - target_size)
        if y_end > H:
            y_end = H
            y_start = max(0, H - target_size)
        if z_end > D:
            z_end = D
            z_start = max(0, D - target_size)
    else:
        # Use bounding box
        x_start, x_end = x_min, x_max + 1
        y_start, y_end = y_min, y_max + 1
        z_start, z_end = z_min, z_max + 1
    
    cropped = voxels[:, z_start:z_end, y_start:y_end, x_start:x_end]
    
    crop_info = {
        'x_range': (x_start, x_end),
        'y_range': (y_start, y_end),
        'z_range': (z_start, z_end),
        'original_shape': voxels.shape,
        'cropped_shape': cropped.shape
    }
    
    return cropped, crop_info


def create_cropped_dataset(input_file, output_file, crop_size=24, padding=2):
    """
    Create a cropped version of the dataset
    
    Args:
        input_file: path to original dataset
        output_file: path to save cropped dataset
        crop_size: target cubic crop size (None for dynamic bounding box)
        padding: padding around content
    """
    print(f"Loading data from {input_file}")
    data = torch.load(input_file, map_location='cpu', weights_only=False)
    
    total_samples = len(data['labels'])
    print(f"Processing {total_samples} samples")
    print(f"Crop size: {crop_size}×{crop_size}×{crop_size}" if crop_size else "Dynamic bounding box")
    print(f"Padding: {padding} voxels")
    
    # Storage for cropped data
    cropped_voxels = []
    skipped = 0
    
    for idx in tqdm(range(total_samples), desc="Cropping samples"):
        voxels = data['unified_voxels'][idx]
        
        # Crop
        cropped, crop_info = crop_to_content(
            voxels.numpy(), 
            target_size=crop_size, 
            padding=padding
        )
        
        # Check if we got the target size
        if crop_size and cropped.shape[1] < crop_size:
            # Pad to target size if needed
            C = cropped.shape[0]
            padded = np.zeros((C, crop_size, crop_size, crop_size), dtype=cropped.dtype)
            
            # Center the content
            z_start = (crop_size - cropped.shape[1]) // 2
            y_start = (crop_size - cropped.shape[2]) // 2
            x_start = (crop_size - cropped.shape[3]) // 2
            
            padded[:, 
                   z_start:z_start+cropped.shape[1],
                   y_start:y_start+cropped.shape[2],
                   x_start:x_start+cropped.shape[3]] = cropped
            
            cropped = padded
        
        cropped_voxels.append(torch.from_numpy(cropped))
    
    print(f"\nSkipped {skipped} samples")
    
    # Stack into tensor
    print("Stacking tensors...")
    cropped_voxels = torch.stack(cropped_voxels)
    
    # Create new dataset
    cropped_data = {
        'unified_voxels': cropped_voxels,
        'labels': data['labels'],
        'ligand_ids': data['ligand_ids'],
        'protein_ids': data['protein_ids'],
        'ligand_smiles': data['ligand_smiles'],
        'pocket_sequences': data['pocket_sequences'],
        'crop_size': crop_size,
        'padding': padding,
        'original_file': str(input_file)
    }
    
    # Print statistics
    original_shape = data['unified_voxels'][0].shape
    new_shape = cropped_voxels[0].shape
    
    print(f"\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Original shape: {original_shape}")
    print(f"New shape: {new_shape}")
    print(f"Volume reduction: {(1 - np.prod(new_shape[1:]) / np.prod(original_shape[1:])) * 100:.1f}%")
    print(f"Memory reduction: {(1 - cropped_voxels.nbytes / data['unified_voxels'].nbytes) * 100:.1f}%")
    print(f"Original size: {data['unified_voxels'].nbytes / 1e9:.2f} GB")
    print(f"New size: {cropped_voxels.nbytes / 1e9:.2f} GB")
    
    # Save
    print(f"\nSaving cropped dataset to {output_file}")
    torch.save(cropped_data, output_file)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Create cropped version of voxel dataset')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to original dataset .pt file')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save cropped dataset')
    parser.add_argument('--crop-size', type=int, default=24,
                        help='Target crop size (cubic). Use 0 for dynamic bounding box.')
    parser.add_argument('--padding', type=int, default=2,
                        help='Padding around content in voxels')
    
    args = parser.parse_args()
    
    crop_size = args.crop_size if args.crop_size > 0 else None
    
    create_cropped_dataset(
        args.input_file,
        args.output_file,
        crop_size=crop_size,
        padding=args.padding
    )


if __name__ == '__main__':
    main()
