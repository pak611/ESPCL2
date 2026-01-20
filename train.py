"""
Clean training script for ESP-JointNet pocket-ligand matching
"""

import torch
import torch.nn as nn
import argparse
import logging
from pathlib import Path
import sys
import json

from models import ESP_JointNet
from utils.dataset import get_field_dataloaders, MaskLigandTransform, MaskESPTransform, MaskHydrophobicTransform
from utils.losses import InfoNCELoss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        pocket_esp = batch['pocket_esp'].to(device)
        ligand_esp = batch['ligand_esp'].to(device)
        
        # Forward pass
        z_pocket, z_ligand, _ = model(pocket_esp, ligand_esp)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z_pocket, z_ligand.T)
        
        # Compute loss
        loss, metrics = criterion(similarity_matrix)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += metrics['loss']
        for key, value in metrics.items():
            if key not in total_metrics:
                total_metrics[key] = 0
            total_metrics[key] += value
        num_batches += 1
        
        # Log progress
        if (batch_idx + 1) % 20 == 0:
            logging.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"R@1%: {metrics['recall1pct_p2l']:.3f}/{metrics['recall1pct_l2p']:.3f} | "
                f"EF1: {metrics['ef1_p2l']:.1f}/{metrics['ef1_l2p']:.1f}"
            )
    
    # Average metrics
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_metrics


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            pocket_esp = batch['pocket_esp'].to(device)
            ligand_esp = batch['ligand_esp'].to(device)
            
            z_pocket, z_ligand, _ = model(pocket_esp, ligand_esp)
            similarity_matrix = torch.matmul(z_pocket, z_ligand.T)
            
            loss, metrics = criterion(similarity_matrix)
            
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value
            num_batches += 1
    
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train ESP-JointNet')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Model
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    
    # Ablation
    parser.add_argument('--mask_ligand', action='store_true', help='Mask all ligand channels (ablation study)')
    parser.add_argument('--mask_esp', action='store_true', help='Mask partial charge channels for both ligand and pocket (ablation study)')
    parser.add_argument('--mask_hydrophobic', action='store_true', help='Mask hydrophobic potential for both ligand and pocket (ablation study)')
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_file', type=str, default='training.log')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("="*80)
    logging.info("ESP-JointNet Training")
    logging.info("="*80)
    logging.info(f"Config: {args}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")
    
    # Load data
    logging.info(f"Loading data from {args.data_path}...")
    
    # Setup ablation transform if needed
    train_transform = None
    if args.mask_ligand:
        logging.info("⚠️  ABLATION: Masking all ligand channels")
        train_transform = MaskLigandTransform(mask_value=0.0)
    elif args.mask_esp:
        logging.info("⚠️  ABLATION: Masking partial charge channels (ligand + pocket)")
        train_transform = MaskESPTransform(mask_value=0.0)
    elif args.mask_hydrophobic:
        logging.info("⚠️  ABLATION: Masking hydrophobic potential (ligand + pocket)")
        train_transform = MaskHydrophobicTransform(mask_value=0.0)
    
    train_loader, val_loader, test_loader = get_field_dataloaders(
        data_file=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
        train_transform=train_transform
    )
    logging.info(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}")
    
    # Detect channel counts from data
    sample = next(iter(train_loader))
    pocket_channels = sample['pocket_esp'].shape[1]
    ligand_channels = sample['ligand_esp'].shape[1]
    logging.info(f"Channels: Pocket={pocket_channels}, Ligand={ligand_channels}")
    
    # Initialize model
    logging.info("Initializing model...")
    model = ESP_JointNet(
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        pocket_channels=pocket_channels,
        ligand_channels=ligand_channels
    )
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = InfoNCELoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Training loop
    logging.info("="*80)
    logging.info("Starting Training")
    logging.info("="*80)
    
    best_val_recall = 0.0
    
    for epoch in range(1, args.epochs + 1):
        logging.info(f"\nEpoch {epoch}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        logging.info(
            f"Train | Loss: {train_metrics['loss']:.4f} | "
            f"R@1%: {train_metrics['recall1pct_p2l']:.3f}/{train_metrics['recall1pct_l2p']:.3f} | "
            f"R@5%: {train_metrics['recall5pct_p2l']:.3f}/{train_metrics['recall5pct_l2p']:.3f} | "
            f"R@10%: {train_metrics['recall10pct_p2l']:.3f}/{train_metrics['recall10pct_l2p']:.3f}"
        )
        logging.info(
            f"      | EF1: {train_metrics['ef1_p2l']:.1f}/{train_metrics['ef1_l2p']:.1f} | "
            f"EF5: {train_metrics['ef5_p2l']:.1f}/{train_metrics['ef5_l2p']:.1f} | "
            f"EF10: {train_metrics['ef10_p2l']:.1f}/{train_metrics['ef10_l2p']:.1f} | "
            f"AUROC: {train_metrics['auroc_p2l']:.3f}/{train_metrics['auroc_l2p']:.3f} | "
            f"Sep: {train_metrics['separation']:.3f}"
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logging.info(
            f"Val   | Loss: {val_metrics['loss']:.4f} | "
            f"R@1%: {val_metrics['recall1pct_p2l']:.3f}/{val_metrics['recall1pct_l2p']:.3f} | "
            f"R@5%: {val_metrics['recall5pct_p2l']:.3f}/{val_metrics['recall5pct_l2p']:.3f} | "
            f"R@10%: {val_metrics['recall10pct_p2l']:.3f}/{val_metrics['recall10pct_l2p']:.3f}"
        )
        logging.info(
            f"      | EF1: {val_metrics['ef1_p2l']:.1f}/{val_metrics['ef1_l2p']:.1f} | "
            f"EF5: {val_metrics['ef5_p2l']:.1f}/{val_metrics['ef5_l2p']:.1f} | "
            f"EF10: {val_metrics['ef10_p2l']:.1f}/{val_metrics['ef10_l2p']:.1f} | "
            f"AUROC: {val_metrics['auroc_p2l']:.3f}/{val_metrics['auroc_l2p']:.3f} | "
            f"Sep: {val_metrics['separation']:.3f}"
        )
        
        scheduler.step()
        
        # Save best model
        avg_recall = (val_metrics['recall1pct_p2l'] + val_metrics['recall1pct_l2p']) / 2
        if avg_recall > best_val_recall:
            best_val_recall = avg_recall
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, save_dir / 'best_model.pt')
            logging.info(f"→ Saved best model (Recall@1%: {best_val_recall:.3f})")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, save_dir / f'checkpoint_epoch{epoch}.pt')
    
    # Test evaluation
    logging.info("\n" + "="*80)
    logging.info("Testing")
    logging.info("="*80)
    
    checkpoint = torch.load(save_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    logging.info(
        f"Test | Loss: {test_metrics['loss']:.4f} | "
        f"R@1%: {test_metrics['recall1pct_p2l']:.3f}/{test_metrics['recall1pct_l2p']:.3f} | "
        f"R@5%: {test_metrics['recall5pct_p2l']:.3f}/{test_metrics['recall5pct_l2p']:.3f} | "
        f"R@10%: {test_metrics['recall10pct_p2l']:.3f}/{test_metrics['recall10pct_l2p']:.3f}"
    )
    logging.info(
        f"     | EF1: {test_metrics['ef1_p2l']:.1f}/{test_metrics['ef1_l2p']:.1f} | "
        f"EF5: {test_metrics['ef5_p2l']:.1f}/{test_metrics['ef5_l2p']:.1f} | "
        f"EF10: {test_metrics['ef10_p2l']:.1f}/{test_metrics['ef10_l2p']:.1f} | "
        f"AUROC: {test_metrics['auroc_p2l']:.3f}/{test_metrics['auroc_l2p']:.3f}"
    )
    
    # Show sample predictions
    logging.info("\n" + "="*80)
    logging.info("Sample Predictions (Pocket→Ligand retrieval)")
    logging.info("="*80)
    
    model.eval()
    with torch.no_grad():
        # Get first batch from test set
        batch = next(iter(test_loader))
        pocket_esp = batch['pocket_esp'].to(device)
        ligand_esp = batch['ligand_esp'].to(device)
        sample_ids = batch['sample_id']
        
        # Get embeddings
        z_pocket, z_ligand, _ = model(pocket_esp, ligand_esp)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z_pocket, z_ligand.T).cpu()
        
        # Get top predictions for each pocket
        _, top_indices = similarity_matrix.topk(5, dim=1)
        
        # Show first 10 samples
        logging.info(f"{'True ID':<15} | {'Rank':<6} | {'Predicted IDs (Top 5)':<60} | {'Similarities'}")
        logging.info("-" * 120)
        
        for i in range(min(10, len(sample_ids))):
            true_id = sample_ids[i]
            predicted_ids = [sample_ids[idx] for idx in top_indices[i]]
            similarities = [f"{similarity_matrix[i, idx].item():.3f}" for idx in top_indices[i]]
            
            # Find rank of correct match
            rank = (top_indices[i] == i).nonzero(as_tuple=True)[0]
            rank_str = f"{rank.item() + 1}" if len(rank) > 0 else ">5"
            
            # Mark correct prediction with ✓
            pred_str = ", ".join([f"{pid}{'✓' if pid == true_id else ''}" for pid in predicted_ids])
            sim_str = ", ".join(similarities)
            
            logging.info(f"{true_id:<15} | {rank_str:<6} | {pred_str:<60} | {sim_str}")
    
    torch.save(test_metrics, save_dir / 'test_results.pt')
    logging.info("\n" + "="*80)
    logging.info("Training Complete!")


if __name__ == '__main__':
    main()
