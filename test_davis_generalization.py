"""Test cross-dataset generalization: BindingDB → DAVIS"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

from models.esp_jointnet import ESP_JointNet
from utils.dataset import ESPPairDataset

def evaluate_on_davis(checkpoint_path, davis_path, max_samples=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load DAVIS dataset
    print(f"Loading DAVIS dataset from {davis_path}")
    davis_dataset = ESPPairDataset(davis_path)
    
    if max_samples:
        davis_dataset = torch.utils.data.Subset(davis_dataset, list(range(max_samples)))
        print(f"Limited to {max_samples} samples")
    
    davis_loader = torch.utils.data.DataLoader(
        davis_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with cross-attention
    model = ESP_JointNet(
        pocket_channels=10,
        ligand_channels=9,
        embedding_dim=256,
        use_cross_attention=True
    ).to(device)
    
    # Remove 'module.' prefix from DataParallel
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"Training loss: {checkpoint.get('train_loss', '?'):.4f}")
    print(f"Validation loss: {checkpoint.get('val_loss', '?'):.4f}\n")
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating on DAVIS...")
    with torch.no_grad():
        for batch in tqdm(davis_loader):
            pocket = batch['pocket_esp'].to(device)
            ligand = batch['ligand_esp'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass (returns predictions, z_pocket, z_ligand, mask_info)
            predictions, _, _, _ = model(pocket, ligand)
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate results
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    # Compute metrics
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    pearson_r, pearson_p = pearsonr(all_labels, all_preds)
    
    print("\n" + "="*70)
    print("CROSS-DATASET GENERALIZATION: BindingDB → DAVIS")
    print("="*70)
    print(f"Samples: {len(all_labels)}")
    print(f"\nMSE:       {mse:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"MAE:       {mae:.4f}")
    print(f"R²:        {r2:.4f}")
    print(f"Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"\nLabel statistics:")
    print(f"  Mean: {all_labels.mean():.4f}")
    print(f"  Std:  {all_labels.std():.4f}")
    print(f"  Range: [{all_labels.min():.4f}, {all_labels.max():.4f}]")
    print(f"\nPrediction statistics:")
    print(f"  Mean: {all_preds.mean():.4f}")
    print(f"  Std:  {all_preds.std():.4f}")
    print(f"  Range: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
    print("="*70)
    
    # Show some examples
    print("\nExample predictions (first 10):")
    print("True Label | Prediction | Error")
    print("-" * 40)
    for i in range(min(10, len(all_labels))):
        error = all_preds[i] - all_labels[i]
        print(f"{all_labels[i]:10.4f} | {all_preds[i]:10.4f} | {error:+.4f}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test cross-dataset generalization')
    parser.add_argument('--checkpoint', type=str, default='results/run_20260109_123001/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--davis-data', type=str, default='data/davis/davis_voxelized.pt',
                        help='Path to DAVIS dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to test on')
    
    args = parser.parse_args()
    
    evaluate_on_davis(args.checkpoint, args.davis_data, max_samples=args.max_samples)
