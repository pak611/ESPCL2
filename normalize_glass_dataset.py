"""
Normalize the glass functional dataset by converting raw IC50/Ki values 
from nM to pIC50 = -log10(M)
"""

import torch

# Load the dataset
input_path = '/home/patrick/Desktop/ESPCL2/data/glass/glass_functional_dataset.pt'
output_path = '/home/patrick/Desktop/ESPCL2/data/glass/glass_functional_dataset_normalized.pt'

print(f'Loading dataset from: {input_path}')
data = torch.load(input_path)

# Get original labels (in nM)
labels_nM = data['labels']

print('\n=== Original Labels (nM) ===')
print(f'Range: {labels_nM.min().item():.2e} to {labels_nM.max().item():.2e}')
print(f'Mean: {labels_nM.mean().item():.2e}, Std: {labels_nM.std().item():.2e}')

# Convert nM to M (multiply by 1e-9), then compute pIC50 = -log10(M)
labels_M = labels_nM * 1e-9
labels_pIC50 = -torch.log10(labels_M)

print('\n=== Transformed Labels (pIC50) ===')
print(f'Range: {labels_pIC50.min().item():.4f} to {labels_pIC50.max().item():.4f}')
print(f'Mean: {labels_pIC50.mean().item():.4f}, Std: {labels_pIC50.std().item():.4f}')

# Check for invalid values
inf_count = torch.isinf(labels_pIC50).sum().item()
nan_count = torch.isnan(labels_pIC50).sum().item()
print(f'Invalid values: {inf_count} inf, {nan_count} nan')

if inf_count > 0 or nan_count > 0:
    print('\nWARNING: Found invalid values! Filtering...')
    valid_mask = ~torch.isinf(labels_pIC50) & ~torch.isnan(labels_pIC50)
    print(f'Keeping {valid_mask.sum().item()} / {len(labels_pIC50)} samples')
    
    # Filter all data
    data['labels'] = labels_pIC50[valid_mask]
    data['unified_voxels'] = data['unified_voxels'][valid_mask]
    data['ligand_ids'] = [data['ligand_ids'][i] for i in range(len(valid_mask)) if valid_mask[i]]
    data['protein_ids'] = [data['protein_ids'][i] for i in range(len(valid_mask)) if valid_mask[i]]
    data['ligand_smiles'] = [data['ligand_smiles'][i] for i in range(len(valid_mask)) if valid_mask[i]]
    data['pocket_sequences'] = [data['pocket_sequences'][i] for i in range(len(valid_mask)) if valid_mask[i]]
    data['protein_sequences'] = [data['protein_sequences'][i] for i in range(len(valid_mask)) if valid_mask[i]]
    data['action_types'] = [data['action_types'][i] for i in range(len(valid_mask)) if valid_mask[i]]
else:
    # Update labels with pIC50 values
    data['labels'] = labels_pIC50

print('\n=== Sample Conversions ===')
for i in range(10):
    print(f'{labels_nM[i].item():>12.2f} nM -> pIC50 = {labels_pIC50[i].item():.4f}')

# Save normalized dataset
print(f'\nSaving normalized dataset to: {output_path}')
torch.save(data, output_path)

print('\n=== Final Dataset Info ===')
print(f'Number of samples: {len(data["labels"])}')
print(f'Voxel shape: {data["unified_voxels"].shape}')
print(f'Label range: {data["labels"].min().item():.4f} to {data["labels"].max().item():.4f}')
print(f'Label mean: {data["labels"].mean().item():.4f}, std: {data["labels"].std().item():.4f}')

print('\n✓ Dataset normalization complete!')
