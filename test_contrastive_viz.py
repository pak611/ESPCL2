"""Quick test of contrastive visualization with synthetic data"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append(str(Path(__file__).parent))
from train import create_chemical_negatives

# Ligand functional role SMARTS patterns (from create_functional_dataset.py)
LIGAND_SMARTS = {
    'h_donor': [
        '[NH,NH2]',       # Primary/secondary amines
        '[NH3+]',         # Ammonium
        '[OH]',           # Hydroxyl
        '[SH]',           # Thiol
        '[nH]',           # Aromatic NH (pyrrole, indole)
    ],
    'h_acceptor': [
        '[N;H0;v3]',      # Tertiary amine with lone pair
        '[N;H0;v2]',      # Imine nitrogen
        '[O;H0]',         # Oxygen with lone pairs (not hydroxyl)
        '[o,n;H0]',       # Aromatic O, N with lone pairs
        '[#16;H0;v2]',    # Sulfur with lone pairs (not thiol)
    ],
    'aromatic': [
        '[a]',             # Any aromatic atom
    ],
    'hydrophobic': [
        '[CH3,CH2]',       # Aliphatic methyl/methylene
        '[CH;X4]',         # Aliphatic methine (sp3)
        '[F,Cl,Br,I]',     # Halogens
    ],
    'positive': [
        '[+1,+2,+3]',      # Explicit positive charge
        '[NH3+,NH2+]',     # Protonated amines
        '[NX4+]',          # Quaternary ammonium
        '[n+]',            # Protonated aromatic nitrogen
    ],
    'negative': [
        '[-1,-2]',         # Explicit negative charge
        '[O-]',            # Oxyanion
        '[CX3](=O)[O-]',   # Carboxylate
        '[PX4](=O)(=O)[O-]',  # Phosphate
        '[SX4](=O)(=O)[O-]',  # Sulfonate
    ],
    'polar': [
        '[N;!H0;+0]',      # Neutral nitrogen with H (not donor, e.g. amide N)
        '[O;!H0;+0]',      # Neutral oxygen with H (hydroxyl/water, overlaps donor)
        '[CX3]=O',         # Carbonyl
        '[SX2;!H0]',       # Sulfur with H (overlaps donor)
    ]
}

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# Create a real small molecule for visualization
# Choose molecule with diverse functionality
options = {
    'captopril': ('CC(CS)C(=O)N1CCCC1C(=O)O', 'Captopril (ACE inhibitor)'),  # thiol, carboxyl, amide, proline
    'aspirin': ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),  # ester, carboxyl, aromatic
    'imatinib': ('Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(n4)c5cccnc5', 'Imatinib (Gleevec)'),  # very complex
    'losartan': ('CCCCc1nc(Cl)c(CO)n1Cc2ccc(cc2)c3ccccc3C4=NNN=N4', 'Losartan'),  # imidazole, tetrazole, chlorine
}

chosen = 'imatinib'
smiles, mol_name = options[chosen]
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=args.seed)
AllChem.UFFOptimizeMolecule(mol)

# Remove Hs for functional role assignment
mol_heavy = Chem.RemoveHs(mol)

# Get 3D coordinates
conf = mol.GetConformer()
positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

# Only keep heavy atoms for positions matching mol_heavy
conf_heavy = mol_heavy.GetConformer()
positions = np.array([list(conf_heavy.GetAtomPosition(i)) for i in range(mol_heavy.GetNumAtoms())])
positions -= positions.mean(axis=0)  # Center at origin

# Scale to fit in 16x16x16 grid (map to voxel space) - use larger scale to fill box
scale = 14.5 / (positions.max() - positions.min())  # Larger scale for bigger appearance
positions = positions * scale + 8.0  # Center at voxel 8,8,8

# Assign functional roles using SMARTS patterns
role_order = ['h_donor', 'h_acceptor', 'aromatic', 'hydrophobic', 'positive', 'negative', 'polar']
n_atoms = mol_heavy.GetNumAtoms()
atom_roles = np.zeros((n_atoms, 7), dtype=np.float32)

for role_idx, role_name in enumerate(role_order):
    patterns = LIGAND_SMARTS[role_name]
    matched_atoms = set()
    
    for smarts in patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            matches = mol_heavy.GetSubstructMatches(pattern)
            for match in matches:
                matched_atoms.update(match)
    
    # Mark matched atoms
    for atom_idx in matched_atoms:
        atom_roles[atom_idx, role_idx] = 1.0

print(f"\nMolecule: {mol_name}")
print(f"Heavy atoms: {n_atoms}")
print(f"Functional roles per atom:")
for i in range(n_atoms):
    atom = mol_heavy.GetAtomWithIdx(i)
    roles_active = [role_order[j] for j in range(7) if atom_roles[i, j] > 0]
    print(f"  Atom {i} ({atom.GetSymbol()}): {roles_active if roles_active else ['none']}")

# Create voxel grids from molecule functional roles
torch.manual_seed(args.seed)
pocket = torch.randn(1, 11, 16, 16, 16) * 0.2
ligand = torch.zeros(1, 8, 16, 16, 16)  # Start with zeros - no noise

# Channel mapping: [ESP, hydrophobic, donor, acceptor, aromatic, positive, negative, polar]
# Role order:      [donor, acceptor, aromatic, hydrophobic, positive, negative, polar]
role_to_channel = {
    0: 2,  # h_donor -> channel 2
    1: 3,  # h_acceptor -> channel 3
    2: 4,  # aromatic -> channel 4
    3: 1,  # hydrophobic -> channel 1
    4: 5,  # positive -> channel 5
    5: 6,  # negative -> channel 6
    6: 7,  # polar -> channel 7
}

# Populate voxel features based on functional roles - ONLY where atoms are
for i in range(n_atoms):
    pos = positions[i]
    x, y, z = int(np.round(pos[0])), int(np.round(pos[1])), int(np.round(pos[2]))
    
    # Only place features in immediate vicinity of atoms (very tight)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                nx, ny, nz = x+dx, y+dy, z+dz
                if 0 <= nx < 10 and 0 <= ny < 10 and 0 <= nz < 10:
                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    weight = np.exp(-dist**2 / 0.3)  # Very tight Gaussian falloff
                    
                    # Apply weight to each active role
                    for role_idx in range(7):
                        if atom_roles[i, role_idx] > 0:
                            channel_idx = role_to_channel[role_idx]
                            ligand[0, channel_idx, nx, ny, nz] += weight * atom_roles[i, role_idx]
    
    # Channel 0: ESP (simple approximation from atom type)
    atom = mol_heavy.GetAtomWithIdx(i)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                nx, ny, nz = x+dx, y+dy, z+dz
                if 0 <= nx < 16 and 0 <= ny < 16 and 0 <= nz < 16:
                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    weight = np.exp(-dist**2 / 0.3)  # Very tight Gaussian falloff
                    
                    if atom.GetSymbol() == 'O':
                        ligand[0, 0, nx, ny, nz] -= weight * 0.5  # Negative
                    elif atom.GetSymbol() == 'N':
                        ligand[0, 0, nx, ny, nz] -= weight * 0.3
                    elif atom.GetSymbol() == 'C' and atom.GetIsAromatic():
                        ligand[0, 0, nx, ny, nz] += weight * 0.1

# NO NOISE - features only where molecule is

# Create negative - replace features with random Gaussian noise centered at 0
torch.manual_seed(args.seed + 1)  # Different seed for corruption
corrupted_pocket = pocket.clone()
corrupted_ligand = ligand.clone()

# Replace features with Gaussian noise (mean=0) only where features exist
noise_std = 0.3  # Standard deviation for noise
for channel_idx in range(8):
    # Create mask of non-zero voxels
    mask = ligand[:, channel_idx, :, :, :] != 0
    # Generate noise
    noise = torch.randn_like(corrupted_ligand[:, channel_idx, :, :, :]) * noise_std
    # Replace original features with noise only where original features exist
    corrupted_ligand[:, channel_idx, :, :, :] = torch.where(
        mask, 
        noise,
        torch.zeros_like(ligand[:, channel_idx, :, :, :])
    )

# Find what was corrupted
pocket_np = pocket[0].numpy()
ligand_np = ligand[0].numpy()
pocket_corr = corrupted_pocket[0].numpy()
ligand_corr = corrupted_ligand[0].numpy()

# Create simple visualization
channel_names = ['ESP', 'Hydrophobic', 'Charged', 'Aromatic', 'Pos', 'Neg', 'Polar', 'Donor', 'Acceptor', 'Metal', 'Sulfur']
ligand_names = ['ESP', 'Hydrophobic', 'Donor', 'Acceptor', 'Aromatic', 'Pos', 'Neg', 'Polar']

print("Checking corruptions...")
# Check multiple channels to see the effects
for ch_idx, ch_name in enumerate(ligand_names):
    orig_mean = np.abs(ligand_np[ch_idx]).mean()
    corr_mean = np.abs(ligand_corr[ch_idx]).mean()
    orig_nonzero = np.count_nonzero(ligand_np[ch_idx])
    corr_nonzero = np.count_nonzero(ligand_corr[ch_idx])
    if orig_nonzero > 0:
        print(f"Channel {ch_idx} ({ch_name}): mean {orig_mean:.4f} → {corr_mean:.4f}, nonzero {orig_nonzero} → {corr_nonzero}")

# Visualize acceptor (channel 3) and hydrophobic (channel 1)
affected_idx = 3  # Acceptor

# Force visualization of ligand channel 3 (acceptor)
is_pocket = False
orig = ligand_np[affected_idx]
corr = ligand_corr[affected_idx]
name = f"Ligand: {ligand_names[affected_idx]}"

# 3D visualization - 2x3 grid (acceptor, hydrophobic, aromatic)
fig = plt.figure(figsize=(20, 12))

# Atom colors (CPK coloring)
atom_colors = []
for atom in mol_heavy.GetAtoms():
    symbol = atom.GetSymbol()
    if symbol == 'C':
        atom_colors.append('#909090')  # Gray
    elif symbol == 'N':
        atom_colors.append('#3050F8')  # Blue
    elif symbol == 'O':
        atom_colors.append('#FF0D0D')  # Red
    elif symbol == 'H':
        atom_colors.append('#FFFFFF')  # White
    else:
        atom_colors.append('#FF1493')  # Pink for others

# Get threshold for visualization (show all non-zero values)
orig_abs = np.abs(orig)
threshold = 0.01  # Very low threshold to show all actual features

# Original - Acceptor channel
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
if threshold > 0:
    points = np.argwhere(orig_abs > threshold)
    if len(points) > 0:
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(vmin=orig.min(), vmax=orig.max())
        colors = [cmap(norm(orig[p[0], p[1], p[2]]))[:3] for p in points]
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=colors, s=30, alpha=0.6)

# Overlay molecule as thick sticks (bonds only)
for bond in mol_heavy.GetBonds():
    idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    p1, p2 = positions[idx1], positions[idx2]
    
    # Get colors for the two atoms
    c1 = atom_colors[idx1]
    c2 = atom_colors[idx2]
    
    # Draw as thick line
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            color=c1, linewidth=6, alpha=0.9, solid_capstyle='round')

ax1.set_title(f'ORIGINAL - ACCEPTOR\n{name}', fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_box_aspect([1,1,1])
ax1.set_xlim(0, 16)
ax1.set_ylim(0, 16)
ax1.set_zlim(0, 16)

# Corrupted - Acceptor channel
ax2 = fig.add_subplot(2, 3, 4, projection='3d')
corr_abs = np.abs(corr)
if np.any(corr_abs > threshold):
    points = np.argwhere(corr_abs > threshold)
    if len(points) > 0:
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(vmin=orig.min(), vmax=orig.max())
        colors = [cmap(norm(corr[p[0], p[1], p[2]]))[:3] for p in points]
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=colors, s=30, alpha=0.6)
else:
    points = np.argwhere(orig_abs > threshold)
    if len(points) > 0:
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='gray', s=30, alpha=0.3)

# Overlay molecule as thick sticks
for bond in mol_heavy.GetBonds():
    idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    p1, p2 = positions[idx1], positions[idx2]
    c1 = atom_colors[idx1]
    c2 = atom_colors[idx2]
    
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            color=c1, linewidth=6, alpha=0.9, solid_capstyle='round')

ax2.set_title(f'CORRUPTED - ACCEPTOR\n(Negative Sample)', fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_box_aspect([1,1,1])
ax2.set_xlim(0, 16)
ax2.set_ylim(0, 16)
ax2.set_zlim(0, 16)

# Third plot - Original Hydrophobic channel
ax3 = fig.add_subplot(2, 3, 2, projection='3d')
orig_hydrophobic = ligand_np[1]
orig_hydrophobic_abs = np.abs(orig_hydrophobic)
if np.any(orig_hydrophobic_abs > threshold):
    points = np.argwhere(orig_hydrophobic_abs > threshold)
    if len(points) > 0:
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(vmin=orig_hydrophobic.min(), vmax=orig_hydrophobic.max())
        colors = [cmap(norm(orig_hydrophobic[p[0], p[1], p[2]]))[:3] for p in points]
        ax3.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=colors, s=30, alpha=0.6)

# Overlay molecule
for bond in mol_heavy.GetBonds():
    idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    p1, p2 = positions[idx1], positions[idx2]
    c1 = atom_colors[idx1]
    ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            color=c1, linewidth=6, alpha=0.9, solid_capstyle='round')

ax3.set_title(f'ORIGINAL - HYDROPHOBIC\nLigand: Hydrophobic', fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_box_aspect([1,1,1])
ax3.set_xlim(0, 16)
ax3.set_ylim(0, 16)
ax3.set_zlim(0, 16)

# Fourth plot - Corrupted Hydrophobic channel
ax4 = fig.add_subplot(2, 3, 5, projection='3d')
corr_hydrophobic = ligand_corr[1]
corr_hydrophobic_abs = np.abs(corr_hydrophobic)
if np.any(corr_hydrophobic_abs > threshold):
    points = np.argwhere(corr_hydrophobic_abs > threshold)
    if len(points) > 0:
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(vmin=orig_hydrophobic.min(), vmax=orig_hydrophobic.max())
        colors = [cmap(norm(corr_hydrophobic[p[0], p[1], p[2]]))[:3] for p in points]
        ax4.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=colors, s=30, alpha=0.6)
else:
    points = np.argwhere(orig_hydrophobic_abs > threshold)
    if len(points) > 0:
        ax4.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='gray', s=30, alpha=0.3)

# Overlay molecule
for bond in mol_heavy.GetBonds():
    idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    p1, p2 = positions[idx1], positions[idx2]
    c1 = atom_colors[idx1]
    ax4.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            color=c1, linewidth=6, alpha=0.9, solid_capstyle='round')

ax4.set_title(f'CORRUPTED - HYDROPHOBIC\n(Negative Sample)', fontsize=11, fontweight='bold', pad=10)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_box_aspect([1,1,1])
ax4.set_xlim(0, 16)
ax4.set_ylim(0, 16)
ax4.set_zlim(0, 16)

# Fifth plot - Original Aromatic channel
ax5 = fig.add_subplot(2, 3, 3, projection='3d')
orig_aromatic = ligand_np[4]  # Channel 4 = Aromatic
orig_aromatic_abs = np.abs(orig_aromatic)
if np.any(orig_aromatic_abs > threshold):
    points = np.argwhere(orig_aromatic_abs > threshold)
    if len(points) > 0:
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(vmin=orig_aromatic.min(), vmax=orig_aromatic.max())
        colors = [cmap(norm(orig_aromatic[p[0], p[1], p[2]]))[:3] for p in points]
        ax5.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=colors, s=30, alpha=0.6)

# Overlay molecule
for bond in mol_heavy.GetBonds():
    idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    p1, p2 = positions[idx1], positions[idx2]
    c1 = atom_colors[idx1]
    ax5.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            color=c1, linewidth=6, alpha=0.9, solid_capstyle='round')

ax5.set_title(f'ORIGINAL - AROMATIC\nLigand: Aromatic', fontsize=11, fontweight='bold', pad=10)
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
ax5.set_box_aspect([1,1,1])
ax5.set_xlim(0, 16)
ax5.set_ylim(0, 16)
ax5.set_zlim(0, 16)

# Sixth plot - Corrupted Aromatic channel
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
corr_aromatic = ligand_corr[4]
corr_aromatic_abs = np.abs(corr_aromatic)
if np.any(corr_aromatic_abs > threshold):
    points = np.argwhere(corr_aromatic_abs > threshold)
    if len(points) > 0:
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(vmin=orig_aromatic.min(), vmax=orig_aromatic.max())
        colors = [cmap(norm(corr_aromatic[p[0], p[1], p[2]]))[:3] for p in points]
        ax6.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=colors, s=30, alpha=0.6)
else:
    points = np.argwhere(orig_aromatic_abs > threshold)
    if len(points) > 0:
        ax6.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='gray', s=30, alpha=0.3)

# Overlay molecule
for bond in mol_heavy.GetBonds():
    idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    p1, p2 = positions[idx1], positions[idx2]
    c1 = atom_colors[idx1]
    ax6.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            color=c1, linewidth=6, alpha=0.9, solid_capstyle='round')

ax6.set_title(f'CORRUPTED - AROMATIC\n(Negative Sample)', fontsize=11, fontweight='bold', pad=10)
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_zlabel('Z')
ax6.set_box_aspect([1,1,1])
ax6.set_xlim(0, 16)
ax6.set_ylim(0, 16)
ax6.set_zlim(0, 16)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.2)
plt.savefig('pair_visualizations/contrastive_test.png', dpi=150, bbox_inches='tight')
print("\nSaved to: pair_visualizations/contrastive_test.png")
