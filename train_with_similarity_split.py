"""Train model with similarity-based train/test splits."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

from models.esp_jointnet import ESP_JointNet
from utils.dataset import ESPPairDataset
from utils.augmentation import VoxelAugmentation


def collate_batch(batch):
    """Collate function for batching samples."""
    return {
        'ligand_esp': torch.stack([item['ligand_esp'] for item in batch]),
        'pocket_esp': torch.stack([item['pocket_esp'] for item in batch]),
        'label': torch.tensor([item['label'].item() for item in batch], dtype=torch.float32),
        'ligand_id': [item['ligand_id'] for item in batch],
        'protein_id': [item['protein_id'] for item in batch]
    }


def evaluate_split(model, dataset, split_indices, batch_size=32, device='cuda'):
    """Evaluate model on a data split."""
    model.eval()
    
    subset = Subset(dataset, split_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0,
                       collate_fn=collate_batch)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            pocket_voxels = batch['pocket_esp'].to(device)
            ligand_voxels = batch['ligand_esp'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(pocket_voxels, ligand_voxels)
            predictions = outputs[0] if isinstance(outputs, tuple) else outputs
            predictions = predictions.squeeze(-1)  # [B, 1] -> [B]
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    pearson_r, pearson_p = pearsonr(all_labels, all_preds)
    
    return {
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'n_samples': len(all_labels)
    }


def train_with_similarity_split(
    data_file: str,
    split_file: str,
    output_dir: str = 'results',
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 0.0001,
    use_cross_attention: bool = True,
    num_heads: int = 8,
    dropout: float = 0.3,
    patience: int = 15,
    save_every: int = 10
):
    """Train model using similarity-based splits."""
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(output_dir) / f'run_similarity_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Load dataset and splits
    print(f"\nLoading dataset from {data_file}")
    dataset = ESPPairDataset(data_file)
    
    print(f"Loading splits from {split_file}")
    splits = torch.load(split_file)
    
    print(f"\nSplit summary:")
    print(f"  Train: {splits['train']['n_samples']} samples, {splits['train']['n_proteins']} proteins")
    for key in splits:
        if key.startswith('test_'):
            split = splits[key]
            print(f"  {key}: {split['n_samples']} samples, {split['n_proteins']} proteins ({split['description']})")
    
    # Create data loaders
    train_indices = splits['train']['indices']
    
    # Shuffle train indices before splitting into train/val to avoid label distribution bias
    np.random.seed(42)
    train_indices = np.array(train_indices)
    np.random.shuffle(train_indices)
    train_indices = train_indices.tolist()
    
    train_subset = Subset(dataset, train_indices)
    
    # Split train into train/val (80/20)
    n_train = len(train_indices)
    n_val = int(0.2 * n_train)
    val_indices = train_indices[:n_val]
    train_indices = train_indices[n_val:]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    # Wrap train dataset with augmentation
    original_getitem = train_subset.__getitem__
    
    def augmented_getitem(idx):
        sample = original_getitem(idx)
        return augmentation(sample)
    
    train_subset.__getitem__ = augmented_getitem
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_batch
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    
    # Create model
    model = ESP_JointNet(
        pocket_channels=10,
        ligand_channels=9,
        embedding_dim=256,
        use_cross_attention=use_cross_attention,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Save config
    config = {
        'data_file': data_file,
        'split_file': split_file,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'use_cross_attention': use_cross_attention,
        'num_heads': num_heads,
        'dropout': dropout,
        'patience': patience,
        'save_every': save_every
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Starting training - results will be saved to {run_dir}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [Train]')
        for batch in pbar:
            pocket_voxels = batch['pocket_esp'].to(device)
            ligand_voxels = batch['ligand_esp'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(pocket_voxels, ligand_voxels)
            predictions = outputs[0] if isinstance(outputs, tuple) else outputs
            predictions = predictions.squeeze(-1)  # [B, 1] -> [B]
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = np.mean(train_losses)
        
        # Validate
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                pocket_voxels = batch['pocket_esp'].to(device)
                ligand_voxels = batch['ligand_esp'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(pocket_voxels, ligand_voxels)
                predictions = outputs[0] if isinstance(outputs, tuple) else outputs
                predictions = predictions.squeeze(-1)  # [B, 1] -> [B]
                loss = criterion(predictions, labels)
                val_losses.append(loss.item())
                
                val_preds.extend(predictions.squeeze().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        
        # Compute validation metrics
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_r2 = r2_score(val_labels, val_preds)
        val_pearson_r, val_pearson_p = pearsonr(val_labels, val_preds)
        val_mae = mean_absolute_error(val_labels, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_labels, val_preds))
        
        print(f'Epoch {epoch}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, R²: {val_r2:.4f}, Pearson r: {val_pearson_r:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}')
        
        # Save checkpoint
        if epoch % save_every == 0:
            checkpoint_path = run_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = run_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_model_path)
            print(f'  → Saved best model (val_loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch}')
                break
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"{'='*70}\n")
    
    # Load best model
    print("Loading best model for evaluation...")
    checkpoint = torch.load(run_dir / 'best_model.pt', map_location=device, weights_only=False)
    
    # Handle DataParallel state dict
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if present (from DataParallel)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    # Evaluate on all splits
    print("\nEvaluating on all splits...")
    results = {}
    
    # Validation set
    print("\n" + "="*70)
    print("VALIDATION SET (same proteins as training)")
    print("="*70)
    val_metrics = evaluate_split(model, dataset, val_indices, batch_size, device)
    print(f"Samples: {val_metrics['n_samples']}")
    print(f"R²:        {val_metrics['r2']:.4f}")
    print(f"Pearson r: {val_metrics['pearson_r']:.4f} (p={val_metrics['pearson_p']:.2e})")
    print(f"RMSE:      {val_metrics['rmse']:.4f}")
    print(f"MAE:       {val_metrics['mae']:.4f}")
    results['validation'] = val_metrics
    
    # Test splits
    for key in splits:
        if key.startswith('test_'):
            split = splits[key]
            print("\n" + "="*70)
            print(f"{key.upper()}: {split['description']}")
            print(f"Proteins: {split['n_proteins']}")
            print("="*70)
            test_metrics = evaluate_split(model, dataset, split['indices'], batch_size, device)
            print(f"Samples: {test_metrics['n_samples']}")
            print(f"R²:        {test_metrics['r2']:.4f}")
            print(f"Pearson r: {test_metrics['pearson_r']:.4f} (p={test_metrics['pearson_p']:.2e})")
            print(f"RMSE:      {test_metrics['rmse']:.4f}")
            print(f"MAE:       {test_metrics['mae']:.4f}")
            results[key] = test_metrics
    
    # Save results
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {run_dir / 'results.json'}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with similarity-based splits')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to dataset file')
    parser.add_argument('--split-file', type=str, required=True,
                        help='Path to split indices file')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--use-cross-attention', action='store_true', default=True)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save-every', type=int, default=10)
    
    args = parser.parse_args()
    
    train_with_similarity_split(
        data_file=args.data_file,
        split_file=args.split_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        use_cross_attention=args.use_cross_attention,
        num_heads=args.num_heads,
        dropout=args.dropout,
        patience=args.patience,
        save_every=args.save_every
    )
