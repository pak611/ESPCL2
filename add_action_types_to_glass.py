"""
Regenerate GLASS functional dataset WITH action_type field properly included
"""

import torch
import pandas as pd
from pathlib import Path

print("Loading GLASS CSV with action types...")
csv_path = "/home/patrick/Desktop/ESP/datasets/glass/glass2_with_msa_filtered.csv"
df = pd.read_csv(csv_path)

print(f"Total CSV entries: {len(df)}")
print(f"Action type distribution:")
action_counts = df['action_type'].value_counts()
for action, count in action_counts.head(20).items():
    print(f"  {action}: {count}")

# Load the existing functional dataset
func_path = "/home/patrick/Desktop/ESPCL2/data/glass/glass_functional_dataset_normalized.pt"
data = torch.load(func_path)

print(f"\nCurrent dataset samples: {len(data['labels'])}")
print(f"Keys: {data.keys()}")

# Need to match protein_ids to action_types from CSV
# The dataset has protein_ids that should match target_uniprot_id in CSV
print("\nMatching action types to dataset...")

# Create a mapping from protein_id + ligand_smiles to action_type
csv_mapping = {}
for _, row in df.iterrows():
    key = (row['target_uniprot_id'], row['Ligand'])
    csv_mapping[key] = row['action_type'] if pd.notna(row['action_type']) else ''

# Match to dataset
matched_actions = []
for i in range(len(data['protein_ids'])):
    key = (data['protein_ids'][i], data['ligand_smiles'][i])
    action = csv_mapping.get(key, '')
    matched_actions.append(action)

# Update dataset
data['action_types'] = matched_actions

# Save updated dataset
output_path = "/home/patrick/Desktop/ESPCL2/data/glass/glass_functional_dataset_normalized_with_actions.pt"
torch.save(data, output_path)

print(f"\nSaved updated dataset to: {output_path}")
print(f"Action type distribution in final dataset:")
from collections import Counter
action_counts = Counter(matched_actions)
for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  '{action}': {count}")
