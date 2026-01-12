"""
Data augmentation for 3D voxelized molecular data
"""

import torch
import numpy as np
from scipy.ndimage import rotate


class RandomRotation3D:
    """
    Apply random 90-degree rotations to 3D voxel grids
    
    Rotations are limited to 90-degree increments to preserve voxel grid structure
    and avoid interpolation artifacts.
    """
    
    def __init__(self, axes=['x', 'y', 'z'], prob=0.5):
        """
        Args:
            axes: list of axes to rotate around ('x', 'y', 'z')
            prob: probability of applying rotation
        """
        self.axes = axes
        self.prob = prob
        self.axis_map = {
            'x': (1, 2),  # rotate in YZ plane
            'y': (0, 2),  # rotate in XZ plane
            'z': (0, 1)   # rotate in XY plane
        }
    
    def sample_params(self):
        """Sample rotation parameters to apply same transform to multiple inputs"""
        if np.random.rand() > self.prob:
            return None
        axis = np.random.choice(self.axes)
        k = np.random.randint(0, 4)
        return {'axis': axis, 'k': k}
    
    def apply_with_params(self, voxel_grid, params):
        """Apply rotation with pre-sampled parameters"""
        if params is None or params['k'] == 0:
            return voxel_grid
        
        plane = self.axis_map[params['axis']]
        k = params['k']
        
        # Handle batch dimension
        if voxel_grid.dim() == 5:
            batch_size = voxel_grid.shape[0]
            rotated = torch.zeros_like(voxel_grid)
            for b in range(batch_size):
                rotated[b] = self._rotate_single_with_params(voxel_grid[b], plane, k)
            return rotated
        else:
            return self._rotate_single_with_params(voxel_grid, plane, k)
    
    def _rotate_single_with_params(self, voxel_grid, plane, k):
        """Rotate with specified parameters"""
        device = voxel_grid.device
        is_cuda = voxel_grid.is_cuda
        
        if is_cuda:
            voxel_grid = voxel_grid.cpu()
        
        rotated = torch.zeros_like(voxel_grid)
        for c in range(voxel_grid.shape[0]):
            rotated[c] = torch.from_numpy(
                np.rot90(voxel_grid[c].numpy(), k=k, axes=plane).copy()
            )
        
        if is_cuda:
            rotated = rotated.to(device)
        
        return rotated
    
    def __call__(self, voxel_grid):
        """
        Apply random rotation to voxel grid
        
        Args:
            voxel_grid: [C, H, W, D] or [B, C, H, W, D] tensor
        
        Returns:
            rotated grid: same shape as input
        """
        if np.random.rand() > self.prob:
            return voxel_grid
        
        # Handle batch dimension
        if voxel_grid.dim() == 5:
            # Batch of voxel grids [B, C, H, W, D]
            batch_size = voxel_grid.shape[0]
            rotated = torch.zeros_like(voxel_grid)
            for b in range(batch_size):
                rotated[b] = self._rotate_single(voxel_grid[b])
            return rotated
        else:
            # Single voxel grid [C, H, W, D]
            return self._rotate_single(voxel_grid)
    
    def _rotate_single(self, voxel_grid):
        """Rotate a single voxel grid [C, H, W, D]"""
        # Choose random axis
        axis = np.random.choice(self.axes)
        plane = self.axis_map[axis]
        
        # Choose random 90-degree rotation (0, 90, 180, 270)
        k = np.random.randint(0, 4)  # number of 90-degree rotations
        
        if k == 0:
            return voxel_grid
        
        # Handle device (CPU or CUDA)
        device = voxel_grid.device
        is_cuda = voxel_grid.is_cuda
        
        # Move to CPU for numpy operations
        if is_cuda:
            voxel_grid = voxel_grid.cpu()
        
        # Apply rotation to each channel
        rotated = torch.zeros_like(voxel_grid)
        for c in range(voxel_grid.shape[0]):
            # rot90 rotates in the specified plane
            rotated[c] = torch.from_numpy(
                np.rot90(voxel_grid[c].numpy(), k=k, axes=plane).copy()
            )
        
        # Move back to original device
        if is_cuda:
            rotated = rotated.to(device)
        
        return rotated


class RandomFlip3D:
    """
    Apply random flips to 3D voxel grids
    """
    
    def __init__(self, axes=['x', 'y', 'z'], prob=0.5):
        """
        Args:
            axes: list of axes to flip along ('x', 'y', 'z')
            prob: probability of applying flip
        """
        self.axes = axes
        self.prob = prob
        self.axis_map = {
            'x': 1,  # flip along X axis (dim 1)
            'y': 2,  # flip along Y axis (dim 2)
            'z': 3   # flip along Z axis (dim 3)
        }
    
    def sample_params(self):
        """Sample flip parameters to apply same transform to multiple inputs"""
        if np.random.rand() > self.prob:
            return None
        axis = np.random.choice(self.axes)
        return {'axis': axis}
    
    def apply_with_params(self, voxel_grid, params):
        """Apply flip with pre-sampled parameters"""
        if params is None:
            return voxel_grid
        dim = self.axis_map[params['axis']]
        return torch.flip(voxel_grid, dims=[dim])
    
    def __call__(self, voxel_grid):
        """
        Apply random flip to voxel grid
        
        Args:
            voxel_grid: [C, H, W, D] tensor
        
        Returns:
            flipped grid: [C, H, W, D] tensor
        """
        if np.random.rand() > self.prob:
            return voxel_grid
        
        # Choose random axis
        axis = np.random.choice(self.axes)
        dim = self.axis_map[axis]
        
        return torch.flip(voxel_grid, dims=[dim])


class Compose:
    """
    Compose multiple augmentations
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class VoxelAugmentation:
    """
    Apply augmentation to pocket and ligand voxels
    
    For SHARED coordinate systems: applies SAME transformation to both
    For SEPARATE coordinate systems: applies INDEPENDENT transformations to each
    """
    
    def __init__(self, rotation_prob=0.5, flip_prob=0.5, shared_coords=True):
        """
        Args:
            rotation_prob: probability of rotation
            flip_prob: probability of flip
            shared_coords: If True, apply same transformation to both (shared coordinate system)
                          If False, apply independent transformations (separate coordinate systems)
        """
        self.shared_coords = shared_coords
        
        if shared_coords:
            # Single transformer applied to both
            self.rotation = RandomRotation3D(prob=rotation_prob)
            self.flip = RandomFlip3D(prob=flip_prob)
        else:
            # Independent transformers for each
            self.rotation_pocket = RandomRotation3D(prob=rotation_prob)
            self.flip_pocket = RandomFlip3D(prob=flip_prob)
            self.rotation_ligand = RandomRotation3D(prob=rotation_prob)
            self.flip_ligand = RandomFlip3D(prob=flip_prob)
    
    def __call__(self, sample):
        """
        Apply augmentation to pocket and ligand
        
        Args:
            sample: dict with 'pocket_esp' and 'ligand_esp' keys
        
        Returns:
            augmented sample
        """
        if self.shared_coords:
            # Apply SAME transformation to both (they share coordinate system)
            sample['pocket_esp'] = self.rotation(sample['pocket_esp'])
            sample['ligand_esp'] = self.rotation(sample['ligand_esp'])
            
            sample['pocket_esp'] = self.flip(sample['pocket_esp'])
            sample['ligand_esp'] = self.flip(sample['ligand_esp'])
        else:
            # Apply INDEPENDENT transformations (separate coordinate systems)
            sample['pocket_esp'] = self.rotation_pocket(sample['pocket_esp'])
            sample['pocket_esp'] = self.flip_pocket(sample['pocket_esp'])
            
            sample['ligand_esp'] = self.rotation_ligand(sample['ligand_esp'])
            sample['ligand_esp'] = self.flip_ligand(sample['ligand_esp'])
        
        return sample


def test_augmentation():
    """Test augmentation on dummy data"""
    print("Testing 3D augmentation...")
    
    # Create dummy voxel grid
    voxel = torch.randn(8, 32, 32, 32)
    
    # Test rotation
    rot = RandomRotation3D(prob=1.0)
    rotated = rot(voxel)
    print(f"Original shape: {voxel.shape}")
    print(f"Rotated shape: {rotated.shape}")
    print(f"Values changed: {not torch.allclose(voxel, rotated)}")
    
    # Test flip
    flip = RandomFlip3D(prob=1.0)
    flipped = flip(voxel)
    print(f"Flipped shape: {flipped.shape}")
    print(f"Values changed: {not torch.allclose(voxel, flipped)}")
    
    # Test full augmentation
    aug = VoxelAugmentation(rotation_prob=1.0, flip_prob=1.0)
    sample = {
        'pocket_esp': torch.randn(22, 64, 64, 64),
        'ligand_esp': torch.randn(8, 32, 32, 32),
        'label': torch.tensor(5.0)
    }
    augmented = aug(sample)
    print(f"\nFull augmentation test:")
    print(f"Pocket shape: {augmented['pocket_esp'].shape}")
    print(f"Ligand shape: {augmented['ligand_esp'].shape}")
    print(f"Label unchanged: {sample['label'] == augmented['label']}")
    print("\nAugmentation tests passed!")


if __name__ == '__main__':
    test_augmentation()
