"""
Dataset utilities for ESP-CL
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def esp_to_voxel_grid(surface_points, surface_esp, grid_size=64, resolution=0.5):
    """
    Convert ESP point cloud to 3D voxel grid
    
    Args:
        surface_points: [N, 3] xyz coordinates (numpy array)
        surface_esp: [N] ESP values at each point (numpy array)
        grid_size: voxel grid dimension
        resolution: angstroms per voxel
    
    Returns:
        grid: [1, grid_size, grid_size, grid_size] voxelized ESP (numpy array)
    """
    # Center the points
    center = surface_points.mean(axis=0)
    points_centered = surface_points - center
    
    # Convert to voxel indices
    voxel_indices = (points_centered / resolution + grid_size / 2).astype(int)
    
    # Clip to grid bounds
    voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)
    
    # Create grid and assign ESP values
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    # Use maximum ESP value if multiple points map to same voxel
    for idx, esp_val in zip(voxel_indices, surface_esp):
        grid[idx[0], idx[1], idx[2]] = max(grid[idx[0], idx[1], idx[2]], esp_val)
    
    return grid[np.newaxis, ...]  # Add channel dimension


class ESPPairDataset(Dataset):
    """
    Dataset for paired ligand-pocket ESP data
    """
    
    def __init__(self, data_file, transform=None):
        """
        Args:
            data_file: path to .pt file with paired data
            transform: optional data augmentation transforms
        """
        self.data = torch.load(data_file, weights_only=False)
        self.transform = transform
        
        # Determine dataset format based on available keys
        if 'unified_voxels' in self.data:
            # Unified grid format: [19, 32, 32, 32] with ligand (0-8) and pocket (9-18) channels
            self.is_voxelized = True
            self.is_unified = True
            self.unified_key = 'unified_voxels'
            self.n_samples = len(self.data['ligand_ids'])
        elif 'ligand_voxels' in self.data:
            # Multi-channel voxelized dataset (separate grids)
            self.is_voxelized = True
            self.is_unified = False
            self.ligand_key = 'ligand_voxels'
            self.pocket_key = 'pocket_voxels'
            self.n_samples = len(self.data['ligand_ids'])
        elif 'ligand_esp_voxels' in self.data:
            # Single-channel (ESP only) voxelized dataset
            self.is_voxelized = True
            self.is_unified = False
            self.ligand_key = 'ligand_esp_voxels'
            self.pocket_key = 'pocket_esp_voxels'
            self.n_samples = len(self.data['ligand_ids'])
        elif 'ligand_esp' in self.data:
            # Raw dataset
            self.is_voxelized = False
            self.is_unified = False
            self.ligand_key = 'ligand_esp'
            self.pocket_key = 'pocket_esp'
            self.n_samples = len(self.data['ligand_ids'])
        else:
            raise ValueError(f"Unknown dataset format. Available keys: {list(self.data.keys())}")
        
        print(f"Loaded {self.n_samples} paired samples from {data_file}")
        print(f"Data format: {'voxelized' if self.is_voxelized else 'raw point cloud'}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Returns a single paired sample
        """
        if hasattr(self, 'is_unified') and self.is_unified:
            # Unified grid: split into ligand and pocket channels
            unified = self.data[self.unified_key][idx]
            ligand_esp = unified[:9]  # Channels 0-8: ligand
            pocket_esp = unified[9:]  # Channels 9-18: pocket
        else:
            # Separate grids
            ligand_esp = self.data[self.ligand_key][idx]
            pocket_esp = self.data[self.pocket_key][idx]
        
        label = self.data['labels'][idx]
        
        # Ensure tensors
        if not isinstance(ligand_esp, torch.Tensor):
            ligand_esp = torch.tensor(ligand_esp, dtype=torch.float32)
        if not isinstance(pocket_esp, torch.Tensor):
            pocket_esp = torch.tensor(pocket_esp, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)
        
        sample = {
            'ligand_esp': ligand_esp,
            'pocket_esp': pocket_esp,
            'label': label,
            'ligand_id': self.data['ligand_ids'][idx],
            'protein_id': self.data['protein_ids'][idx]
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class VoxelizeTransform:
    """
    Transform to convert ESP point cloud to voxel grid
    """
    
    def __init__(self, pocket_grid_size=64, ligand_grid_size=32, resolution=0.5):
        self.pocket_grid_size = pocket_grid_size
        self.ligand_grid_size = ligand_grid_size
        self.resolution = resolution
    
    def __call__(self, sample):
        """
        Voxelize ESP point clouds
        
        Note: This assumes ESP data comes with corresponding point coordinates
        Currently our data is just ESP values, we need point coordinates too
        """
        # TODO: Implement once we have point coordinates in the dataset
        return sample


class RandomRotation3D:
    """
    Random 3D rotation augmentation for voxel grids
    """
    
    def __init__(self, angle_range=180):
        self.angle_range = angle_range
    
    def __call__(self, sample):
        """
        Randomly rotate the voxel grids
        """
        # TODO: Implement 3D rotation
        return sample


class ESPNoise:
    """
    Add Gaussian noise to ESP values for data augmentation
    """
    
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std
    
    def __call__(self, sample):
        """
        Add noise to ESP values
        """
        if 'ligand_esp' in sample:
            noise = torch.randn_like(sample['ligand_esp']) * self.noise_std
            sample['ligand_esp'] = sample['ligand_esp'] + noise
        
        if 'pocket_esp' in sample:
            noise = torch.randn_like(sample['pocket_esp']) * self.noise_std
            sample['pocket_esp'] = sample['pocket_esp'] + noise
        
        return sample


def collate_variable_size(batch):
    """
    Custom collate function for batches with variable-size point clouds.
    Returns lists instead of stacked tensors for point cloud data.
    """
    # Check if we have voxelized or point cloud data
    first_sample = batch[0]
    
    # If ligand_esp is 4D or more, it's voxelized - we can stack
    if len(first_sample['ligand_esp'].shape) >= 3:
        # Voxelized data - stack normally
        return {
            'ligand_esp': torch.stack([item['ligand_esp'] for item in batch]),
            'pocket_esp': torch.stack([item['pocket_esp'] for item in batch]),
            'label': torch.stack([item['label'] for item in batch]),
            'ligand_id': [item['ligand_id'] for item in batch],
            'protein_id': [item['protein_id'] for item in batch]
        }
    else:
        # Point cloud data - keep as lists
        return {
            'ligand_esp': [item['ligand_esp'] for item in batch],
            'pocket_esp': [item['pocket_esp'] for item in batch],
            'label': torch.stack([item['label'] for item in batch]),
            'ligand_id': [item['ligand_id'] for item in batch],
            'protein_id': [item['protein_id'] for item in batch]
        }


def get_dataloaders(data_file, batch_size=128, train_split=0.7, val_split=0.15,
                    num_workers=4, seed=42, train_transform=None):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_file: path to paired dataset .pt file
        batch_size: batch size for training
        train_split: fraction of data for training (default: 0.7)
        val_split: fraction of data for validation (default: 0.15)
        num_workers: number of dataloader workers
        seed: random seed for reproducibility
        train_transform: optional transform for training data augmentation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load dataset ONCE
    dataset = ESPPairDataset(data_file)
    
    # Split into train/val/test
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val
    
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )
    
    # Wrap train dataset with transform if provided
    if train_transform is not None:
        from functools import partial
        original_getitem = train_dataset.__getitem__
        
        def augmented_getitem(idx):
            sample = original_getitem(idx)
            return train_transform(sample)
        
        train_dataset.__getitem__ = augmented_getitem
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_variable_size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_variable_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_variable_size
    )
    
    print(f"Created dataloaders: {n_train} train, {n_val} val, {n_test} test samples")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset loading
    data_file = '/home/patrick/Desktop/ESPCL/data/paired_dataset.pt'
    
    if Path(data_file).exists():
        print("Testing dataset loading...")
        dataset = ESPPairDataset(data_file)
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Ligand ESP shape: {sample['ligand_esp'].shape}")
        print(f"Pocket ESP shape: {sample['pocket_esp'].shape}")
        print(f"Label: {sample['label'].item():.4f}")
        print(f"Ligand ID: {sample['ligand_id']}")
        print(f"Protein ID: {sample['protein_id']}")
        
        # Test dataloader
        print("\nTesting dataloaders...")
        train_loader, val_loader = get_dataloaders(data_file, batch_size=8)
        
        batch = next(iter(train_loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch ligand ESP shape: {batch['ligand_esp'].shape}")
        print(f"Batch pocket ESP shape: {batch['pocket_esp'].shape}")
        print(f"Batch labels shape: {batch['label'].shape}")
    else:
        print(f"Dataset file not found: {data_file}")
