"""
Dataset utilities for ESP-CL
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


def field_collate_fn(batch):
    """
    Custom collate function that handles string fields properly.
    Stacks tensors and skips string fields.
    """
    if not batch:
        return {}
    
    # Separate tensor and non-tensor fields
    collated = {}
    
    # Handle ligand_esp
    collated['ligand_esp'] = torch.stack([item['ligand_esp'] for item in batch])
    
    # Handle pocket_esp
    collated['pocket_esp'] = torch.stack([item['pocket_esp'] for item in batch])
    
    # Handle labels
    collated['label'] = torch.stack([item['label'] for item in batch])
    
    # Handle sample_id (keep as list, don't try to tensorize)
    if 'sample_id' in batch[0]:
        collated['sample_id'] = [item['sample_id'] for item in batch]
    
    return collated


class MaskLigandTransform:
    """Transform to mask all ligand channels (for ablation studies)"""
    
    def __init__(self, mask_value=0.0):
        self.mask_value = mask_value
    
    def __call__(self, sample):
        """Mask all ligand ESP channels"""
        sample['ligand_esp'] = torch.full_like(sample['ligand_esp'], self.mask_value)
        return sample


class MaskESPTransform:
    """Transform to mask partial charge channels (channel 0 for ligand, channel 8 for pocket)
    
    Note: This dataset uses Gasteiger partial charges, not ESP+/ESP-.
    Channel ordering:
        Ligand: 0=charges, 1=rotbonds, 2=hydro, 3=donor, 4=acceptor, 5=moltype, 6=aromatic, 7=atomtype
        Pocket: 8=charges, 9=rotbonds, 10=hydro, 11=donor, 12=acceptor, 13=moltype, 14=aromatic, 15=atomtype
    """
    
    def __init__(self, mask_value=0.0):
        self.mask_value = mask_value
    
    def __call__(self, sample):
        """Mask partial charge channels (channel 0 for ligand, channel 0 for pocket after split)"""
        # Mask channel 0 (partial charges) for ligand
        sample['ligand_esp'][0, :, :, :] = self.mask_value
        # Mask channel 0 (partial charges) for pocket (which is channel 8 in unified, but channel 0 after split)
        sample['pocket_esp'][0, :, :, :] = self.mask_value
        return sample


class MaskHydrophobicTransform:
    """Transform to mask hydrophobic potential channels (channel 2 for both ligand and pocket)
    
    Channel ordering:
        Ligand: 0=charges, 1=rotbonds, 2=hydro, 3=donor, 4=acceptor, 5=moltype, 6=aromatic, 7=atomtype
        Pocket: 8=charges, 9=rotbonds, 10=hydro, 11=donor, 12=acceptor, 13=moltype, 14=aromatic, 15=atomtype
    """
    
    def __init__(self, mask_value=0.0):
        self.mask_value = mask_value
    
    def __call__(self, sample):
        """Mask hydrophobic potential channels (channel 2 after split)"""
        # Mask channel 2 (hydrophobic) for ligand
        sample['ligand_esp'][2, :, :, :] = self.mask_value
        # Mask channel 2 (hydrophobic) for pocket (channel 10 in unified, channel 2 after split)
        sample['pocket_esp'][2, :, :, :] = self.mask_value
        return sample


class FieldBasedDataset(Dataset):
    """
    Dataset for field-based ligand-pocket voxelized data
    Handles multi-channel continuous field representations
    """
    
    def __init__(self, data_file, transform=None):
        """
        Args:
            data_file: path to .pt file with field-based data
            transform: optional data augmentation transforms
        """
        print(f"Loading field-based dataset from {data_file}")
        self.data = torch.load(data_file, weights_only=False, map_location='cpu')
        self.transform = transform
        
        # Check dataset format
        required_keys = ['voxels', 'labels']
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Missing required key '{key}' in dataset. Available keys: {list(self.data.keys())}")
        
        self.voxels = self.data['voxels']  # [N, C, H, W, D]
        self.labels = self.data['labels']  # [N]
        
        # Handle different identifier keys
        if 'pdb_codes' in self.data:
            self.identifiers = self.data['pdb_codes']
        elif 'sample_ids' in self.data:
            self.identifiers = self.data['sample_ids']
        else:
            self.identifiers = [f"sample_{i}" for i in range(len(self.labels))]
        
        # Load protein sequences and ligand SMILES if available
        self.protein_sequences = self.data.get('protein_sequences', None)
        self.ligand_smiles = self.data.get('ligand_smiles', None)
        
        self.n_samples = len(self.identifiers)
        
        # Validate shapes
        assert len(self.voxels) == self.n_samples, f"Voxels length {len(self.voxels)} != n_samples {self.n_samples}"
        assert len(self.labels) == self.n_samples, f"Labels length {len(self.labels)} != n_samples {self.n_samples}"
        n_channels = self.voxels.shape[1]
        
        channels_per_mol = n_channels // 2
        print(f"Loaded {self.n_samples} paired samples")
        print(f"Voxel shape: {self.voxels.shape}")
        print(f"Grid size: {self.voxels.shape[2]}³")
        print(f"Channels: {n_channels} ({channels_per_mol} ligand + {channels_per_mol} pocket fields)")
        print(f"Label range: [{self.labels.min():.2f}, {self.labels.max():.2f}]")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                'ligand_esp': [C/2, H, W, D] ligand fields
                'pocket_esp': [C/2, H, W, D] pocket fields
                'label': scalar affinity value
                'sample_id': string identifier
        """
        # Split unified voxels into ligand and pocket
        voxels = self.voxels[idx]  # [C, H, W, D]
        n_channels = voxels.shape[0]
        channels_per_mol = n_channels // 2
        
        ligand_fields = voxels[:channels_per_mol]  # First half: ligand channels
        pocket_fields = voxels[channels_per_mol:]  # Second half: pocket channels
        
        sample = {
            'ligand_esp': ligand_fields,
            'pocket_esp': pocket_fields,
            'label': self.labels[idx],
            'sample_id': self.identifiers[idx]
        }
        
        # Add sequence identifiers if available
        if self.protein_sequences is not None:
            sample['protein_seq'] = self.protein_sequences[idx]
        if self.ligand_smiles is not None:
            sample['ligand_smiles'] = self.ligand_smiles[idx]
        
        # Apply augmentation if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def get_field_dataloaders(data_file, batch_size=32, train_split=0.8, 
                          num_workers=4, seed=42, train_transform=None):
    """
    Create train/val/test dataloaders for field-based dataset
    
    Args:
        data_file: path to .pt file
        batch_size: batch size
        train_split: fraction for training (rest split equally for val/test)
        num_workers: number of dataloader workers
        seed: random seed for reproducibility
        train_transform: augmentation for training data only
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    full_dataset = FieldBasedDataset(data_file)
    
    # Split into train/val/test
    n_samples = len(full_dataset)
    n_train = int(train_split * n_samples)
    n_val = (n_samples - n_train) // 2
    n_test = n_samples - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Apply transform only to training data
    if train_transform:
        # Wrap train dataset to apply transforms
        class TransformedSubset(Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            
            def __len__(self):
                return len(self.subset)
            
            def __getitem__(self, idx):
                sample = self.subset[idx]
                if self.transform:
                    sample = self.transform(sample)
                return sample
        
        train_dataset = TransformedSubset(train_dataset, train_transform)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=field_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=field_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=field_collate_fn
    )
    
    return train_loader, val_loader, test_loader
