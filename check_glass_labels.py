import torch
import numpy as np

# Load dataset
data = torch.load('/home/patrick/Desktop/ESPCL2/data/glass/glass_functional_dataset.pt')
labels = data['labels']

print('Original label distribution:')
print(f'  Range: {labels.min().item():.2e} to {labels.max().item():.2e}')
print(f'  Mean: {labels.mean().item():.2e}, Std: {labels.std().item():.2e}')

print('\nLog-transformed (pIC50 style = -log10(M)):')
# Convert nM to M, then take -log10
log_labels = -torch.log10(labels * 1e-9)
print(f'  Range: {log_labels.min().item():.4f} to {log_labels.max().item():.4f}')
print(f'  Mean: {log_labels.mean().item():.4f}, Std: {log_labels.std().item():.4f}')

print('\nSample conversions:')
for i in range(15):
    print(f'  {labels[i].item():>12.2f} nM -> pIC50 = {log_labels[i].item():.4f}')

print('\nCheck for inf/nan after transformation:')
print(f'  inf count: {torch.isinf(log_labels).sum().item()}')
print(f'  nan count: {torch.isnan(log_labels).sum().item()}')
