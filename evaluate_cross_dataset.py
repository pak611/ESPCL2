"""
Cross-dataset evaluation (one-shot/zero-shot):
Train on BindingDB → Test on DAVIS (no DAVIS training)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import sys

sys.path.append(str(Path(__file__).parent))
from models.esp_jointnet_shared_proj import ESP_JointNet
from utils.dataset import ESPPairDataset


def evaluate_cross_dataset(checkpoint_path, test_data_path, device='cuda', batch_size=32, max_samples=None, use_cross_attention=False):
    """
    Load model trained on BindingDB and test on DAVIS
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        test_data_path: Path to DAVIS dataset
        device: Device to run on
        batch_size: Batch size for evaluation
        max_samples: Optional limit on number of samples to test
    """
    print(f"\n{'='*70}")
    print("CROSS-DATASET EVALUATION (One-Shot Generalization)")
    print(f"{'='*70}")
    
    # Load test dataset
    print(f"\nLoading test dataset from {test_data_path}")
    test_dataset = ESPPairDataset(test_data_path)
    
    if max_samples and max_samples < len(test_dataset):
        indices = list(range(max_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        print(f"Limited to {max_samples} samples for faster evaluation")
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load trained model
    print(f"\nLoading trained model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = ESP_JointNet(
        embedding_dim=256,
        dropout=0.3,
        pocket_channels=10,
        ligand_channels=9,
        use_cross_attention=use_cross_attention
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if present (from DataParallel)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        epoch = checkpoint.get('epoch', 'unknown')
        train_loss = checkpoint.get('train_loss', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}, train loss: {train_loss}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Evaluate
    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            ligand_voxels = batch['ligand_esp'].to(device)
            pocket_voxels = batch['pocket_esp'].to(device)
            labels = batch['label'].to(device)
            
            predictions, _, _, _ = model(pocket_voxels, ligand_voxels)
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {(i+1)*batch_size}/{len(test_dataset)} samples...")
    
    # Concatenate results
    predictions = np.concatenate(all_predictions).flatten()
    labels = np.concatenate(all_labels).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    pearson_r, p_value = pearsonr(labels, predictions)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS: Cross-Dataset Generalization Performance")
    print("="*70)
    print(f"Test samples:  {len(predictions)}")
    print(f"\nRegression Metrics:")
    print(f"  MSE:        {mse:.4f}")
    print(f"  RMSE:       {rmse:.4f}")
    print(f"  MAE:        {mae:.4f}")
    print(f"  R²:         {r2:.4f}")
    print(f"  Pearson r:  {pearson_r:.4f} (p={p_value:.2e})")
    print(f"\nLabel statistics:")
    print(f"  Mean:       {labels.mean():.4f}")
    print(f"  Std:        {labels.std():.4f}")
    print(f"  Range:      [{labels.min():.4f}, {labels.max():.4f}]")
    print(f"\nPrediction statistics:")
    print(f"  Mean:       {predictions.mean():.4f}")
    print(f"  Std:        {predictions.std():.4f}")
    print(f"  Range:      [{predictions.min():.4f}, {predictions.max():.4f}]")
    print("="*70)
    
    # Show examples
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
    parser = argparse.ArgumentParser(description='Cross-dataset evaluation (one-shot)')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/patrick/Desktop/ESPCL2/results/run_20260109_101228/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--test-data', type=str,
                        default='/home/patrick/Desktop/ESPCL2/data/davis/davis_voxelized.pt',
                        help='Path to test dataset (DAVIS)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to test (for speed)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = evaluate_cross_dataset(
        args.checkpoint,
        args.test_data,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    print("\n✓ Cross-dataset evaluation complete!")
    print("\nInterpretation:")
    print("  - This measures how well BindingDB training generalizes to DAVIS")
    print("  - Good one-shot performance (R²>0.3, Pearson>0.5) indicates")
    print("    the model learned transferable binding affinity patterns")
    print("  - Compare to within-dataset test performance to assess generalization")


if __name__ == '__main__':
    main()
