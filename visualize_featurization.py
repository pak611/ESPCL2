"""
Visualize the featurization strategy for the paper.
Shows different feature channels for ligand and pocket side by side.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse

# SMARTS patterns for functional groups
LIGAND_SMARTS = {
    'h_donor': ['[#7H,#8H,#16H]', '[NH]', '[OH]'],
    'h_acceptor': ['[N,O,F]', '[n,o]'],
    'aromatic': ['a'],
    'hydrophobic': ['[CH3]', '[CH2][CH3]', 'c'],
    'positive': ['[+]', '[NH3+]', '[NH2+]'],
    'negative': ['[-]', '[O-]', 'C(=O)[O-]'],
    'polar': ['[OH]', '[NH2]', 'C=O', 'C(=O)O']
}

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--sample-idx', type=int, default=100, help='Which sample to visualize from dataset')
args = parser.parse_args()

print(f"Using sample index: {args.sample_idx}")

# Load actual pocket-ligand pair from dataset
print("Loading pocket-ligand pair from dataset...")
dataset_path = '/home/patrick/Desktop/ESPCL2/data/bindingdb_2016/voxelized_unified_48_32_normalized.pt'
dataset = torch.load(dataset_path, weights_only=False)

print(f"Dataset type: {type(dataset)}")
print(f"Dataset keys: {dataset.keys()}")

# Extract the unified voxels
unified_voxels = dataset['unified_voxels']
print(f"Unified voxels shape: {unified_voxels.shape}")
print(f"Number of samples: {unified_voxels.shape[0]}")

# Get sample [pocket_channels + ligand_channels, H, W, D]
sample_idx = args.sample_idx
sample_voxel = unified_voxels[sample_idx]
ligand_smiles = dataset['ligand_smiles'][sample_idx]
protein_seq = dataset['pocket_sequences'][sample_idx]
affinity = dataset['labels'][sample_idx]

print(f"\nSample {sample_idx}:")
print(f"  SMILES: {ligand_smiles}")
print(f"  Affinity: {affinity:.3f}")
print(f"  Pocket sequence length: {len(protein_seq)}")
print(f"  Sample voxel shape: {sample_voxel.shape}")

# Split into pocket (first 11 channels) and ligand (last 8 channels)
pocket = sample_voxel[:11].unsqueeze(0)  # [1, 11, H, W, D]
ligand = sample_voxel[11:].unsqueeze(0)  # [1, 8, H, W, D]
print(f"Pocket shape: {pocket.shape}")
print(f"Ligand shape: {ligand.shape}")

# Resize to 16x16x16 if needed
if pocket.shape[2:] != (16, 16, 16):
    import torch.nn.functional as F
    pocket = F.interpolate(pocket, size=(16, 16, 16), mode='trilinear', align_corners=False)
    ligand = F.interpolate(ligand, size=(16, 16, 16), mode='trilinear', align_corners=False)
    print(f"Resized to: {pocket.shape}")

print(f"Final shapes - Pocket: {pocket.shape}, Ligand: {ligand.shape}")

# For ligand visualization, we'll create a simple molecule structure overlay
# from the actual voxel data by finding the center of mass and creating bonds
ligand_combined = ligand[0].sum(dim=0).numpy()
ligand_points = np.argwhere(ligand_combined > np.percentile(ligand_combined[ligand_combined > 0], 70))
ligand_center = ligand_points.mean(axis=0) if len(ligand_points) > 0 else np.array([8, 8, 8])

# Use the SMILES to create a molecule for visualization
try:
    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=args.seed)
        AllChem.UFFOptimizeMolecule(mol)
        mol_heavy = Chem.RemoveHs(mol)
        
        # Get 3D coordinates
        conf_heavy = mol_heavy.GetConformer()
        positions = np.array([list(conf_heavy.GetAtomPosition(i)) for i in range(mol_heavy.GetNumAtoms())])
        positions -= positions.mean(axis=0)
        scale = 14.5 / (positions.max() - positions.min())
        positions = positions * scale + 8.0
        
        mol_name = ligand_smiles[:40] + "..." if len(ligand_smiles) > 40 else ligand_smiles
        use_molecule_overlay = True
    else:
        use_molecule_overlay = False
        mol_heavy = None
        positions = None
except:
    use_molecule_overlay = False
    mol_heavy = None
    positions = None

# Get atom colors for molecule overlay if available
atom_colors = []
if use_molecule_overlay and mol_heavy is not None:
    for i in range(mol_heavy.GetNumAtoms()):
        atom = mol_heavy.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        if symbol == 'C':
            atom_colors.append('#909090')
        elif symbol == 'N':
            atom_colors.append('#3050F8')
        elif symbol == 'O':
            atom_colors.append('#FF0D0D')
        else:
            atom_colors.append('#FF1493')

# Channel names and indices
ligand_channels = [
    ('ESP', 0, 'RdBu_r'),
    ('Hydrophobic', 1, 'Greens'),
    ('H-Donor', 2, 'Blues'),
    ('H-Acceptor', 3, 'Oranges'),
    ('Aromatic', 4, 'Purples'),
]

pocket_channels = [
    ('ESP', 0, 'RdBu_r'),
    ('Hydrophobic', 1, 'Greens'),
    ('H-Donor', 2, 'Blues'),
    ('H-Acceptor', 3, 'Oranges'),
    ('Aromatic', 4, 'Purples'),
]

# Create figure with 5 rows x 2 columns
fig = plt.figure(figsize=(10, 20))
threshold = 0.01

for row_idx, (name, ch_idx, cmap_name) in enumerate(ligand_channels):
    # Ligand column (left)
    ax_lig = fig.add_subplot(5, 2, row_idx*2 + 1, projection='3d')
    ligand_data = ligand[0, ch_idx].numpy()
    ligand_abs = np.abs(ligand_data)
    
    if np.any(ligand_abs > threshold):
        points = np.argwhere(ligand_abs > threshold)
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=ligand_data.min(), vmax=ligand_data.max())
        colors = [cmap(norm(ligand_data[p[0], p[1], p[2]]))[:3] for p in points]
        ax_lig.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors, s=20, alpha=0.7)
    
    # Overlay molecule
    for bond in mol_heavy.GetBonds():
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        p1, p2 = positions[idx1], positions[idx2]
        c1 = atom_colors[idx1]
        ax_lig.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   color=c1, linewidth=3, alpha=0.8, solid_capstyle='round')
    
    ax_lig.set_title(f'Ligand: {name}', fontsize=11, fontweight='bold')
    ax_lig.set_xlim(0, 16)
    ax_lig.set_ylim(0, 16)
    ax_lig.set_zlim(0, 16)
    ax_lig.set_box_aspect([1,1,1])
    ax_lig.axis('off')
    
    # Pocket column (right)
    ax_pkt = fig.add_subplot(5, 2, row_idx*2 + 2, projection='3d')
    pocket_data = pocket[0, ch_idx].numpy()
    pocket_abs = np.abs(pocket_data)
    
    if np.any(pocket_abs > threshold):
        points = np.argwhere(pocket_abs > threshold)
        # Subsample to avoid overcrowding
        if len(points) > 500:
            indices = np.random.choice(len(points), 500, replace=False)
            points = points[indices]
        
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=pocket_data.min(), vmax=pocket_data.max())
        colors = [cmap(norm(pocket_data[p[0], p[1], p[2]]))[:3] for p in points]
        ax_pkt.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors, s=20, alpha=0.7)
    
    # Overlay pocket surface (show all channels combined as gray outline)
    pocket_combined = np.abs(pocket[0].numpy()).sum(axis=0)
    pocket_surface_threshold = np.percentile(pocket_combined[pocket_combined > 0], 80) if np.any(pocket_combined > 0) else 0
    if pocket_surface_threshold > 0:
        surface_points = np.argwhere(pocket_combined > pocket_surface_threshold)
        if len(surface_points) > 300:
            indices = np.random.choice(len(surface_points), 300, replace=False)
            surface_points = surface_points[indices]
        ax_pkt.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2],
                      c='gray', s=10, alpha=0.2, edgecolors='none')
    
    ax_pkt.set_title(f'Pocket: {name}', fontsize=11, fontweight='bold')
    ax_pkt.set_xlim(0, 16)
    ax_pkt.set_ylim(0, 16)
    ax_pkt.set_zlim(0, 16)
    ax_pkt.set_box_aspect([1,1,1])
    ax_pkt.axis('off')

plt.tight_layout()
plt.savefig('paper_figures/featurization_strategy.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\nSaved to: paper_figures/featurization_strategy.png")
