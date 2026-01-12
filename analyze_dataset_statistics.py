"""
Comprehensive analysis of dataset sources, processing, and statistics
"""

import torch
import pandas as pd
from pathlib import Path
import json

print("=" * 80)
print("DATASET ANALYSIS: Sources, Processing, and Statistics")
print("=" * 80)

# Dataset information with sources
datasets_info = {
    "BindingDB 2016": {
        "source": "https://www.bindingdb.org/bind/index.jsp",
        "description": "PDBbind includes a general set (~19k complexes) and core set (~195 diverse complexes)",
        "original_csv": "/home/patrick/Desktop/ESP/datasets/bindingdb/BindingDB2016.csv",
        "paired_dataset": "/home/patrick/Desktop/ESP/datasets/bindingdb/paired_dataset.pt",
        "functional_dataset": "/home/patrick/Desktop/ESP/datasets/bindingdb/bindingdb_functional_dataset.pt",
        "successful_pockets": "/home/patrick/Desktop/ESP/datasets/bindingdb/successful_pockets.csv"
    },
    "DAVIS": {
        "source": "https://doi.org/10.1038/nbt.1990",
        "description": "442 kinases vs 68 ligands (~30k interactions, 10nm-10um range)",
        "original_csv": "/home/patrick/Desktop/Extract_Pocket/datasets/DAVIS/davis_full_formatted.csv",
        "train_csv": "/home/patrick/Desktop/Extract_Pocket/datasets/DAVIS/davis_train_formatted.csv",
        "test_csv": "/home/patrick/Desktop/Extract_Pocket/datasets/DAVIS/davis_test_formatted.csv",
        "paired_dataset": "/home/patrick/Desktop/ESP/datasets/davis/paired_dataset.pt",
        "functional_dataset": "/home/patrick/Desktop/ESP/datasets/davis/davis_functional_dataset.pt",
        "successful_pockets": "/home/patrick/Desktop/ESP/datasets/davis/successful_pockets.csv"
    },
    "KIBA": {
        "source": "https://doi.org/10.1021/ci400709d",
        "description": "518 kinases vs 612 ligands (~230k interactions, unified KIBA scores)",
        "original_csv": "/home/patrick/Desktop/Extract_Pocket/datasets/KIBA/kiba_full.csv",
        "train_csv": "/home/patrick/Desktop/Extract_Pocket/datasets/KIBA/kiba_train.csv",
        "test_csv": "/home/patrick/Desktop/Extract_Pocket/datasets/KIBA/kiba_test.csv",
        "paired_dataset": "/home/patrick/Desktop/ESP/datasets/kiba/paired_dataset.pt",
        "functional_dataset": "/home/patrick/Desktop/ESP/datasets/kiba/kiba_functional_dataset.pt",
        "successful_pockets": "/home/patrick/Desktop/ESP/datasets/kiba/successful_pockets.csv"
    },
    "GLASS": {
        "source": "https://zhanggroup.org/GLASS/",
        "description": "1,147,227 protein-ligand pairs with diverse affinities and annotations",
        "original_csv": "/home/patrick/Desktop/ESP/datasets/glass/glass2_with_msa_filtered.csv",
        "paired_dataset": "/home/patrick/Desktop/ESP/datasets/glass/paired_dataset.pt",
        "functional_dataset": "/home/patrick/Desktop/ESP/datasets/glass/glass_functional_dataset.pt",
        "functional_normalized": "/home/patrick/Desktop/ESPCL2/data/glass/glass_functional_dataset_normalized.pt",
        "successful_pockets": "/home/patrick/Desktop/ESP/datasets/glass/successful_pockets.csv"
    }
}

print("\n" + "=" * 80)
print("DATASET SOURCES AND DESCRIPTIONS")
print("=" * 80)

for name, info in datasets_info.items():
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Source: {info['source']}")
    print(f"Description: {info['description']}")

print("\n" + "=" * 80)
print("PROCESSING PIPELINE STATISTICS")
print("=" * 80)

for name, info in datasets_info.items():
    print(f"\n{'=' * 80}")
    print(f"{name} DATASET")
    print(f"{'=' * 80}")
    
    # Original CSV statistics
    if Path(info.get('original_csv', '')).exists():
        try:
            df = pd.read_csv(info['original_csv'])
            print(f"\n1. Original CSV: {Path(info['original_csv']).name}")
            print(f"   Total rows: {len(df):,}")
            if 'regression_label' in df.columns:
                print(f"   Affinity range: {df['regression_label'].min():.2f} to {df['regression_label'].max():.2f}")
            print(f"   Columns: {', '.join(df.columns[:5])}...")
        except Exception as e:
            print(f"   Could not read: {e}")
    
    # Paired dataset (after ESP extraction)
    if Path(info.get('paired_dataset', '')).exists():
        try:
            data = torch.load(info['paired_dataset'])
            print(f"\n2. Paired Dataset (after ESP/pocket extraction): {Path(info['paired_dataset']).name}")
            if isinstance(data, dict):
                if 'affinities' in data:
                    print(f"   Number of pairs: {len(data['affinities']):,}")
                    print(f"   Affinity stats: min={data['affinities'].min():.2f}, max={data['affinities'].max():.2f}, mean={data['affinities'].mean():.2f}")
                elif 'labels' in data:
                    print(f"   Number of pairs: {len(data['labels']):,}")
                    print(f"   Label stats: min={data['labels'].min():.2e}, max={data['labels'].max():.2e}")
                print(f"   Keys: {', '.join(data.keys())}")
        except Exception as e:
            print(f"   Could not load: {e}")
    
    # Successful pockets mapping
    if Path(info.get('successful_pockets', '')).exists():
        try:
            pockets_df = pd.read_csv(info['successful_pockets'])
            print(f"\n3. Successful Pocket Extraction: {Path(info['successful_pockets']).name}")
            print(f"   Proteins with extracted pockets: {len(pockets_df):,}")
            if 'source_pocket' in pockets_df.columns:
                print(f"   Sample pocket IDs: {pockets_df['source_pocket'].head(3).tolist()}")
        except Exception as e:
            print(f"   Could not read: {e}")
    
    # Functional dataset (final voxelized)
    if Path(info.get('functional_dataset', '')).exists():
        try:
            data = torch.load(info['functional_dataset'])
            print(f"\n4. Functional Dataset (final voxelized): {Path(info['functional_dataset']).name}")
            if isinstance(data, dict):
                if 'unified_voxels' in data:
                    print(f"   Number of samples: {len(data['unified_voxels']):,}")
                    print(f"   Voxel shape: {data['unified_voxels'].shape}")
                    print(f"   Channels: {data['unified_voxels'].shape[1]} (ESP + functional groups)")
                if 'labels' in data:
                    print(f"   Label type: {'Raw IC50/Ki (nM)' if data['labels'].max() > 1000 else 'pIC50'}")
                    print(f"   Label range: {data['labels'].min():.4f} to {data['labels'].max():.4f}")
                elif 'affinities' in data:
                    print(f"   Affinity range: {data['affinities'].min():.4f} to {data['affinities'].max():.4f}")
        except Exception as e:
            print(f"   Could not load: {e}")
    
    # Check for normalized version
    if info.get('functional_normalized') and Path(info['functional_normalized']).exists():
        try:
            data = torch.load(info['functional_normalized'])
            print(f"\n5. Normalized Dataset: {Path(info['functional_normalized']).name}")
            print(f"   Number of samples: {len(data['labels']):,}")
            print(f"   pIC50 range: {data['labels'].min():.4f} to {data['labels'].max():.4f}")
            print(f"   pIC50 mean ± std: {data['labels'].mean():.4f} ± {data['labels'].std():.4f}")
        except Exception as e:
            print(f"   Could not load: {e}")

print("\n" + "=" * 80)
print("SUCCESS RATE ANALYSIS")
print("=" * 80)

for name, info in datasets_info.items():
    original_path = info.get('original_csv', '')
    final_path = info.get('functional_dataset', '')
    
    if Path(original_path).exists() and Path(final_path).exists():
        try:
            original_df = pd.read_csv(original_path)
            final_data = torch.load(final_path)
            
            original_count = len(original_df)
            if 'unified_voxels' in final_data:
                final_count = len(final_data['unified_voxels'])
            elif 'labels' in final_data:
                final_count = len(final_data['labels'])
            else:
                final_count = 0
            
            success_rate = (final_count / original_count * 100) if original_count > 0 else 0
            
            print(f"\n{name}:")
            print(f"  Original entries: {original_count:,}")
            print(f"  Final voxelized: {final_count:,}")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Failed/filtered: {original_count - final_count:,}")
        except Exception as e:
            print(f"\n{name}: Could not compute - {e}")

print("\n" + "=" * 80)
print("ONE-TO-ONE RELATIONSHIP ANALYSIS")
print("=" * 80)
print("""
Dataset Structure:
- BindingDB: One-to-one ligand-protein-pocket complexes (each row = unique complex)
- DAVIS: Matrix format - 442 kinases × 68 ligands = multiple measurements per pair
- KIBA: Matrix format - 518 kinases × 612 ligands = aggregated from multiple sources
- GLASS: One-to-one pairs with diverse annotations

After Processing:
All datasets are converted to one-to-one ligand-pocket pairs for training.
Each sample has:
  - Unique ligand (SMILES)
  - Unique protein sequence
  - Extracted binding pocket (PDB)
  - Unified voxel grid (19 channels: ESP + functional groups)
  - Binding affinity (pIC50 or normalized)
""")

print("\n" + "=" * 80)
print("HOMOGENEITY ANALYSIS")
print("=" * 80)

for name, info in datasets_info.items():
    functional_path = info.get('functional_dataset', '')
    if Path(functional_path).exists():
        try:
            data = torch.load(functional_path)
            
            print(f"\n{name}:")
            
            # Check unique proteins and ligands
            if 'protein_ids' in data:
                unique_proteins = len(set(data['protein_ids']))
                total_samples = len(data['protein_ids'])
                print(f"  Unique proteins: {unique_proteins:,}")
                print(f"  Avg samples per protein: {total_samples/unique_proteins:.1f}")
            
            if 'ligand_ids' in data:
                unique_ligands = len(set(data['ligand_ids']))
                print(f"  Unique ligands: {unique_ligands:,}")
                print(f"  Avg samples per ligand: {total_samples/unique_ligands:.1f}")
            
            # Affinity distribution
            if 'labels' in data:
                labels = data['labels']
                print(f"  Affinity distribution:")
                print(f"    Mean: {labels.mean():.4f}")
                print(f"    Median: {labels.median():.4f}")
                print(f"    Std: {labels.std():.4f}")
                print(f"    Range: [{labels.min():.4f}, {labels.max():.4f}]")
            
        except Exception as e:
            print(f"  Could not analyze: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Key Points:
1. All datasets come from established benchmarks in drug discovery
2. Processing pipeline: CSV → ESP extraction → Pocket detection → Voxelization
3. Success rates vary based on:
   - Availability of PDB structures for proteins
   - fpocket's ability to detect binding sites
   - Ligand conformer generation success
   - Voxelization quality checks
4. Final datasets are standardized:
   - 19-channel unified voxel grids (32×32×32)
   - One-to-one ligand-pocket pairs
   - Normalized affinity values (pIC50)
5. Homogeneity varies:
   - BindingDB: Diverse proteins, high structural variety
   - DAVIS/KIBA: Focused on kinases, more homogeneous
   - GLASS: Large-scale, diverse annotations
""")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
