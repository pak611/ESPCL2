"""
One-shot evaluation: measure random initialization performance as baseline
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import sys

# Add models to path
sys.path.append(str(Path(__file__).parent))
from models.esp_jointnet_shared_proj import ESP_JointNet
from utils.dataset import ESPPairDataset


def evaluate_oneshot(data_path, model, device='cuda', batch_size=32, num_samples=1000):
    """
    Evaluate untrained model on test data
    
    Args:
        data_path: Path to dataset
        model: Untrained model
        device: Device to run on
        batch_size: Batch size for evaluation
        num_samples: Number of samples to evaluate (for speed)
    """
    print(f"Loading dataset from {data_path}")
    dataset = ESPPairDataset(data_path)
    
    # Use last num_samples as test set
    test_size = min(num_samples, len(dataset))
    test_indices = list(range(len(dataset) - test_size, len(dataset)))
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Evaluating on {len(test_dataset)} samples with random weights...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Get voxels from batch
            ligand_voxels = batch['ligand_esp'].to(device)
            pocket_voxels = batch['pocket_esp'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            predictions, _, _, _ = model(pocket_voxels, ligand_voxels)
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions).flatten()
    labels = np.concatenate(all_labels).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    pearson_r, _ = pearsonr(labels, predictions)
    
    print("\n" + "="*60)
    print("ONE-SHOT PERFORMANCE (Random Initialization Baseline)")
    print("="*60)
    print(f"Samples evaluated: {len(predictions)}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Pearson r: {pearson_r:.4f}")
    print("="*60)
    
    # Print some example predictions
    print("\nExample predictions (first 10):")
    print("True Label | Prediction | Error")
    print("-" * 40)
    for i in range(min(10, len(predictions))):
        error = predictions[i] - labels[i]
        print(f"{labels[i]:10.4f} | {predictions[i]:10.4f} | {error:+.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r,
        'predictions': predictions,
        'labels': labels
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='One-shot evaluation on BindingDB')
    parser.add_argument('--data-path', type=str,
                        default='/home/patrick/Desktop/ESPCL2/data/bindingdb_2016/voxelized_unified_48_32_colocated.pt',
                        help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to evaluate')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--use-cross-attention', action='store_true',
                        help='Use cross-attention between pocket and ligand')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create untrained model
    print("\nInitializing model with random weights...")
    model = ESP_JointNet(
        embedding_dim=args.embedding_dim,
        dropout=0.3,
        pocket_channels=10,  # Unified format: channels 9-18
        ligand_channels=9,   # Unified format: channels 0-8
        use_cross_attention=args.use_cross_attention
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Evaluate
    results = evaluate_oneshot(
        args.data_path,
        model,
        device=device,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    print("\n✓ One-shot evaluation complete!")
    print("This baseline shows what random initialization achieves.")
    print("Training should improve significantly beyond these metrics.")


if __name__ == '__main__':
    main()
