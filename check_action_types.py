import torch
from collections import Counter

# Load dataset
data = torch.load('data/glass/glass_functional_dataset_normalized.pt')
action_types = data['action_types']

# Count action types
counts = Counter(action_types)
print('Action type distribution:')
for action, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'  {repr(action):20s}: {count:6d} samples')

print(f'\nTotal samples: {len(action_types)}')

# Show some examples
print('\nSample protein-action pairs:')
for i in range(min(20, len(action_types))):
    if action_types[i]:  # Only show non-empty
        print(f'  Protein: {data["protein_ids"][i][:40]:40s} | Action: {action_types[i]:15s} | Affinity: {data["labels"][i]:.3f}')
