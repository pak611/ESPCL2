"""
Training script for ESP supervised binding affinity prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.esp_jointnet import ESP_JointNet
from utils.dataset import get_dataloaders
from utils.augmentation import VoxelAugmentation


def create_chemical_negatives(pocket_esp, ligand_esp, corruption_rate=0.1):
    """
    Create hard chemical negatives by corrupting 10% of batch
    
    Corruptions (applied randomly):
    0. Flip ESP sign in pocket (destroys electrostatics)
    1. Zero out hydrophobic channel in pocket (removes key binding feature)
    2. Zero out random channels in ligand (corrupts ligand features)
    3. Swap hydrophilic/hydrophobic channels in pocket (reverses polarity preferences)
    4. Remove donor/acceptor regions in ligand (destroys H-bond capability)
    
    Args:
        pocket_esp: [B, C, H, W, D] pocket voxels (10 channels: ESP, hydrophobic, charged, aromatic, etc.)
        ligand_esp: [B, C, H, W, D] ligand voxels (9 channels: ESP, hydrophobic, donor, acceptor, etc.)
        corruption_rate: fraction of batch to corrupt (default 0.1)
    
    Returns:
        corrupted_pocket: [B, C, H, W, D] with some samples corrupted
        corrupted_ligand: [B, C, H, W, D] with some samples corrupted
        corruption_mask: [B] boolean mask indicating corrupted samples
    """
    B = pocket_esp.size(0)
    device = pocket_esp.device
    
    # Determine which samples to corrupt
    n_corrupt = max(1, int(B * corruption_rate))
    corrupt_idx = torch.randperm(B, device=device)[:n_corrupt]
    corruption_mask = torch.zeros(B, dtype=torch.bool, device=device)
    corruption_mask[corrupt_idx] = True
    
    corrupted_pocket = pocket_esp.clone()
    corrupted_ligand = ligand_esp.clone()
    
    for idx in corrupt_idx:
        corruption_type = torch.randint(0, 5, (1,)).item()
        
        if corruption_type == 0:
            # Flip ESP sign in pocket (channel 0 is ESP)
            corrupted_pocket[idx, 0] = -corrupted_pocket[idx, 0]
        
        elif corruption_type == 1:
            # Zero hydrophobic channel in pocket (channel 1 is hydrophobic)
            if corrupted_pocket.size(1) > 1:
                corrupted_pocket[idx, 1] = 0
        
        elif corruption_type == 2:
            # Zero random channels in ligand
            if corrupted_ligand.size(1) > 2:
                rand_ch = torch.randint(0, corrupted_ligand.size(1), (1,)).item()
                corrupted_ligand[idx, rand_ch] = 0
        
        elif corruption_type == 3:
            # Swap hydrophilic/hydrophobic channels in pocket (reverses polarity preferences)
            # Assuming: channel 1 = hydrophobic, channel 2 = charged/hydrophilic
            if corrupted_pocket.size(1) > 2:
                temp = corrupted_pocket[idx, 1].clone()
                corrupted_pocket[idx, 1] = corrupted_pocket[idx, 2]
                corrupted_pocket[idx, 2] = temp
        
        elif corruption_type == 4:
            # Remove donor/acceptor regions in ligand (destroys H-bond capability)
            # Assuming: channel 2 = donor, channel 3 = acceptor (after ESP and hydrophobic)
            if corrupted_ligand.size(1) > 3:
                corrupted_ligand[idx, 2] = 0  # Zero donor
                corrupted_ligand[idx, 3] = 0  # Zero acceptor
    
    return corrupted_pocket, corrupted_ligand, corruption_mask


def train_epoch(model, train_loader, optimizer, device, epoch, use_contrastive=False, contrastive_alpha=0.1, 
                positive_augmentation='masking', chemical_negatives_rate=0.1):
    """Train for one epoch with optional contrastive learning"""
    import time
    model.train()
    total_loss = 0
    total_mse_loss = 0
    total_contrastive_loss = 0
    total_samples = 0
    start_time = time.time()
    
    # Diagnostic tracking
    gradient_norms = {'encoder': [], 'attention': [], 'projection': [], 'contrastive_proj': []}
    cl_diagnostics = {'pos_sim_p': [], 'neg_sim_p': [], 'pos_sim_l': [], 'neg_sim_l': []}
    activation_stats = {'encoder_norm': [], 'pooled_norm': [], 'embedding_norm': []}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        # Handle both tensor batches and list batches
        if isinstance(batch['ligand_esp'], list):
            # Point cloud data - stack the list of tensors
            ligand_esp = torch.stack(batch['ligand_esp']).to(device)
            pocket_esp = torch.stack(batch['pocket_esp']).to(device)
        else:
            # Voxelized data - already stacked
            ligand_esp = batch['ligand_esp'].to(device)
            pocket_esp = batch['pocket_esp'].to(device)
        
        labels = batch['label'].to(device)
        
        # Handle DataParallel wrapper
        model_module = model.module if isinstance(model, nn.DataParallel) else model
        
        if use_contrastive and (model_module.use_cross_attention or model_module.use_pairwise_attention):
            # Dual-view contrastive learning with configurable augmentation
            
            if positive_augmentation == 'masking':
                # View 1 & 2: Same input, different masking patterns
                predictions_v1, z_pocket_v1, z_ligand_v1, mask_info_v1 = model(pocket_esp, ligand_esp)
                predictions_v2, z_pocket_v2, z_ligand_v2, mask_info_v2 = model(pocket_esp, ligand_esp)
                
            elif positive_augmentation == 'rotation':
                # View 1 & 2: Different rotations, no masking (disable masking temporarily)
                # Apply SAME rotation to both pocket and ligand to preserve relative geometry
                original_masking = model_module.use_masking
                model_module.use_masking = False
                
                # Apply random rotations/flips
                from utils.augmentation import RandomRotation3D, RandomFlip3D
                rot_augment = RandomRotation3D(prob=1.0)
                flip_augment = RandomFlip3D(prob=0.5)
                
                # View 1: Sample transform once, apply to both pocket and ligand
                rot_params_v1 = rot_augment.sample_params()
                flip_params_v1 = flip_augment.sample_params()
                pocket_v1 = flip_augment.apply_with_params(
                    rot_augment.apply_with_params(pocket_esp, rot_params_v1), 
                    flip_params_v1
                )
                ligand_v1 = flip_augment.apply_with_params(
                    rot_augment.apply_with_params(ligand_esp, rot_params_v1),
                    flip_params_v1
                )
                
                # View 2: Sample different transform, apply to both pocket and ligand
                rot_params_v2 = rot_augment.sample_params()
                flip_params_v2 = flip_augment.sample_params()
                pocket_v2 = flip_augment.apply_with_params(
                    rot_augment.apply_with_params(pocket_esp, rot_params_v2),
                    flip_params_v2
                )
                ligand_v2 = flip_augment.apply_with_params(
                    rot_augment.apply_with_params(ligand_esp, rot_params_v2),
                    flip_params_v2
                )
                
                predictions_v1, z_pocket_v1, z_ligand_v1, mask_info_v1 = model(pocket_v1, ligand_v1)
                predictions_v2, z_pocket_v2, z_ligand_v2, mask_info_v2 = model(pocket_v2, ligand_v2)
                
                model_module.use_masking = original_masking
                
            elif positive_augmentation == 'rotation+masking':
                # View 1 & 2: Different rotations + different masking
                # Apply SAME rotation to both pocket and ligand to preserve relative geometry
                from utils.augmentation import RandomRotation3D, RandomFlip3D
                rot_augment = RandomRotation3D(prob=1.0)
                flip_augment = RandomFlip3D(prob=0.5)
                
                # View 1: Sample transform once, apply to both pocket and ligand
                rot_params_v1 = rot_augment.sample_params()
                flip_params_v1 = flip_augment.sample_params()
                pocket_v1 = flip_augment.apply_with_params(
                    rot_augment.apply_with_params(pocket_esp, rot_params_v1),
                    flip_params_v1
                )
                ligand_v1 = flip_augment.apply_with_params(
                    rot_augment.apply_with_params(ligand_esp, rot_params_v1),
                    flip_params_v1
                )
                
                # View 2: Sample different transform, apply to both pocket and ligand
                rot_params_v2 = rot_augment.sample_params()
                flip_params_v2 = flip_augment.sample_params()
                pocket_v2 = flip_augment.apply_with_params(
                    rot_augment.apply_with_params(pocket_esp, rot_params_v2),
                    flip_params_v2
                )
                ligand_v2 = flip_augment.apply_with_params(
                    rot_augment.apply_with_params(ligand_esp, rot_params_v2),
                    flip_params_v2
                )
                
                predictions_v1, z_pocket_v1, z_ligand_v1, mask_info_v1 = model(pocket_v1, ligand_v1)
                predictions_v2, z_pocket_v2, z_ligand_v2, mask_info_v2 = model(pocket_v2, ligand_v2)
            
            # Create chemical negatives (hard negatives that shouldn't bind)
            pocket_neg, ligand_neg, neg_mask = create_chemical_negatives(
                pocket_esp, ligand_esp, corruption_rate=chemical_negatives_rate
            )
            predictions_neg, z_pocket_neg, z_ligand_neg, mask_info_neg = model(pocket_neg, ligand_neg)
            
            # Average predictions from both views for regression loss
            predictions = (predictions_v1 + predictions_v2) / 2
            mse_loss = model_module.regression_loss(predictions, labels)
            
            # Hard Contrastive Loss with Chemical Negatives
            # Positive pairs: same molecule across views (rotation invariance)
            # In-batch negatives: different molecules (random)
            # Chemical negatives: corrupted pairs that shouldn't bind (hard negatives)
            contrastive_loss = model_module.contrastive_loss_with_negatives(
                z_pocket_v1, z_ligand_v1, z_pocket_v2, z_ligand_v2,
                z_pocket_neg, z_ligand_neg, neg_mask, temperature=0.07
            )
            
            # Embedding geometry diagnostics
            with torch.no_grad():
                # View-pair similarities (augmentation invariance)
                pos_sim_p = torch.cosine_similarity(z_pocket_v1, z_pocket_v2, dim=-1).mean().item()
                pos_sim_l = torch.cosine_similarity(z_ligand_v1, z_ligand_v2, dim=-1).mean().item()
                cl_diagnostics['pos_sim_p'].append(pos_sim_p)
                cl_diagnostics['pos_sim_l'].append(pos_sim_l)
                
                # Label-based similarities (soft supervised CL target)
                # Sort by labels and compare nearest vs farthest neighbors
                sorted_idx = torch.argsort(labels.squeeze())
                if len(sorted_idx) >= 4:
                    # Similar labels (close in sorted order)
                    near_idx1, near_idx2 = sorted_idx[0], sorted_idx[1]
                    z_near1 = torch.cat([z_pocket_v1[near_idx1:near_idx1+1], z_ligand_v1[near_idx1:near_idx1+1]], dim=-1)
                    z_near2 = torch.cat([z_pocket_v1[near_idx2:near_idx2+1], z_ligand_v1[near_idx2:near_idx2+1]], dim=-1)
                    near_sim = F.cosine_similarity(z_near1, z_near2, dim=-1).item()
                    
                    # Dissimilar labels (far in sorted order)
                    far_idx1, far_idx2 = sorted_idx[0], sorted_idx[-1]
                    z_far1 = torch.cat([z_pocket_v1[far_idx1:far_idx1+1], z_ligand_v1[far_idx1:far_idx1+1]], dim=-1)
                    z_far2 = torch.cat([z_pocket_v1[far_idx2:far_idx2+1], z_ligand_v1[far_idx2:far_idx2+1]], dim=-1)
                    far_sim = F.cosine_similarity(z_far1, z_far2, dim=-1).item()
                    
                    cl_diagnostics['neg_sim_p'].append(far_sim)  # "Negative" = dissimilar labels
                    cl_diagnostics['neg_sim_l'].append(near_sim)  # Store near_sim in neg_sim_l for tracking
                    
                    # Embedding statistics
                    activation_stats['embedding_norm'].append(z_pocket_v1.norm(dim=-1).mean().item())
            
            # Combined loss
            loss = mse_loss + contrastive_alpha * contrastive_loss
            
            # Track individual losses
            total_mse_loss += mse_loss.item() * ligand_esp.size(0)
            total_contrastive_loss += contrastive_loss.item() * ligand_esp.size(0)
        else:
            # Standard supervised training
            output = model(pocket_esp, ligand_esp)
            
            # Handle different return formats
            if isinstance(output, tuple):
                predictions = output[0]  # First element is predictions
            else:
                predictions = output
            
            # Compute regression loss (MSE / L2)
            loss = model_module.regression_loss(predictions, labels)
            total_mse_loss += loss.item() * ligand_esp.size(0)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Track gradient norms before clipping
        model_module = model.module if isinstance(model, nn.DataParallel) else model
        if hasattr(model_module, 'pocket_encoder'):
            encoder_grad = sum(p.grad.norm().item() for p in model_module.pocket_encoder.parameters() if p.grad is not None)
            gradient_norms['encoder'].append(encoder_grad)
        
        if hasattr(model_module, 'spatial_cross_attn_p'):
            attn_grad = sum(p.grad.norm().item() for p in model_module.spatial_cross_attn_p.parameters() if p.grad is not None)
            gradient_norms['attention'].append(attn_grad)
        
        if hasattr(model_module, 'projection_p'):
            proj_grad = sum(p.grad.norm().item() for p in model_module.projection_p.parameters() if p.grad is not None)
            gradient_norms['projection'].append(proj_grad)
        
        # For post-attention CL, projection_p/l ARE the contrastive projections
        # Track the same gradient (since they're shared for both CL and regression)
        if hasattr(model_module, 'projection_p'):
            gradient_norms['contrastive_proj'].append(proj_grad)
        elif hasattr(model_module, 'contrastive_proj_p'):
            # Fallback for pre-attention CL (if used)
            cl_proj_grad = sum(p.grad.norm().item() for p in model_module.contrastive_proj_p.parameters() if p.grad is not None)
            gradient_norms['contrastive_proj'].append(cl_proj_grad)
        
        # Gradient clipping to stabilize contrastive learning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Update metrics
        batch_size = ligand_esp.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Update progress bar
        if use_contrastive and model_module.use_cross_attention:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse_loss.item():.4f}',
                'cl': f'{contrastive_loss.item():.4f}'
            })
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_time = time.time() - start_time
    avg_loss = total_loss / total_samples
    avg_mse = total_mse_loss / total_samples
    avg_cl = total_contrastive_loss / total_samples if use_contrastive else 0
    
    # Compute diagnostic statistics
    diagnostics = {
        'grad_norms': {k: sum(v) / len(v) if v else 0 for k, v in gradient_norms.items()},
        'cl_diagnostics': {k: sum(v) / len(v) if v else 0 for k, v in cl_diagnostics.items()},
        'activation_stats': {k: sum(v) / len(v) if v else 0 for k, v in activation_stats.items()}
    }
    
    return avg_loss, avg_mse, avg_cl, train_time, diagnostics


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # Metrics for regression
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(val_loader, desc='Validation'):
        # Handle both tensor batches and list batches
        if isinstance(batch['ligand_esp'], list):
            # Point cloud data - stack the list of tensors
            ligand_esp = torch.stack(batch['ligand_esp']).to(device)
            pocket_esp = torch.stack(batch['pocket_esp']).to(device)
        else:
            # Voxelized data - already stacked
            ligand_esp = batch['ligand_esp'].to(device)
            pocket_esp = batch['pocket_esp'].to(device)
        
        labels = batch['label'].to(device)
        
        # Forward pass
        output = model(pocket_esp, ligand_esp)
        
        # Handle different return formats (with or without contrastive embeddings)
        if isinstance(output, tuple):
            predictions = output[0]  # First element is predictions
        else:
            predictions = output
        
        # Compute loss
        # Handle DataParallel wrapper
        model_module = model.module if isinstance(model, nn.DataParallel) else model
        loss = model_module.regression_loss(predictions, labels)
        
        # Store predictions and labels for metrics
        pred_list = predictions.squeeze().cpu().tolist()
        if isinstance(pred_list, float):  # Single sample batch
            pred_list = [pred_list]
        all_predictions.extend(pred_list)
        
        label_list = labels.cpu().tolist() if labels.dim() == 1 else labels.squeeze().cpu().tolist()
        if isinstance(label_list, float):  # Single sample batch
            label_list = [label_list]
        all_labels.extend(label_list)
        
        # Update metrics
        batch_size = ligand_esp.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    
    # Compute correlation metrics
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    
    pred_array = np.array(all_predictions)
    label_array = np.array(all_labels)
    
    pearson_r, _ = pearsonr(pred_array, label_array)
    spearman_r, _ = spearmanr(pred_array, label_array)
    mae = np.mean(np.abs(pred_array - label_array))
    rmse = np.sqrt(np.mean((pred_array - label_array) ** 2))
    sd = np.std(pred_array)  # Standard deviation of predictions
    
    # Compute concordance index (C-index)
    def concordance_index(y_true, y_pred):
        """Calculate concordance index for ranking quality"""
        concordant = 0
        discordant = 0
        for i in range(len(y_true)):
            for j in range(i + 1, len(y_true)):
                if y_true[i] != y_true[j]:
                    if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                       (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                        concordant += 1
                    else:
                        discordant += 1
        if concordant + discordant == 0:
            return 0.5
        return concordant / (concordant + discordant)
    
    c_index = concordance_index(label_array, pred_array)
    
    metrics = {
        'loss': avg_loss,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'mae': mae,
        'rmse': rmse,
        'sd': sd,
        'c_index': c_index
    }
    
    return metrics


def train(args):
    """Main training function"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {n_gpus}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'logs')
    
    # Setup data augmentation for training with shared coordinate system
    if args.no_augment:
        augmentation = None
        print("Data augmentation disabled")
    else:
        augmentation = VoxelAugmentation(rotation_prob=0.5, flip_prob=0.5, shared_coords=True)
        print("Data augmentation enabled: Shared coordinate transformations (same rotation/flip for both)")
    
    # Load data
    print(f"\nLoading data from {args.data_file}")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_file,
        batch_size=args.batch_size,
        train_split=args.train_split,
        num_workers=args.num_workers,
        seed=args.seed,
        train_transform=augmentation
    )
    
    # Detect number of channels from data
    sample_pocket = train_loader.dataset[0]['pocket_esp']
    sample_ligand = train_loader.dataset[0]['ligand_esp']
    pocket_channels = sample_pocket.shape[0] if len(sample_pocket.shape) == 4 else 1
    ligand_channels = sample_ligand.shape[0] if len(sample_ligand.shape) == 4 else 1
    
    # Create model
    print(f"\nCreating ESP-JointNet model (embedding_dim={args.embedding_dim}, temperature={args.temperature}, dropout={args.dropout})")
    print(f"  Pocket channels: {pocket_channels}")
    print(f"  Ligand channels: {ligand_channels}")
    if args.use_cross_attention:
        print(f"  Using cross-attention with {args.num_heads} heads")
    if args.use_pairwise_attention:
        print(f"  Using pairwise attention (AlphaFold-style)")
    if args.channel_mask_ratio > 0 or args.spatial_mask_ratio > 0:
        print(f"  Channel masking: {args.channel_mask_ratio:.1%}, Spatial masking: {args.spatial_mask_ratio:.1%} (patch size: {args.mask_patch_size})")
    
    model = ESP_JointNet(
        embedding_dim=args.embedding_dim,
        temperature=args.temperature,
        dropout=args.dropout,
        pocket_channels=pocket_channels,
        ligand_channels=ligand_channels,
        use_cross_attention=args.use_cross_attention,
        use_pairwise_attention=args.use_pairwise_attention,
        num_heads=args.num_heads,
        channel_mask_ratio=args.channel_mask_ratio,
        spatial_mask_ratio=args.spatial_mask_ratio,
        mask_patch_size=args.mask_patch_size
    )

    # Use DataParallel if multiple GPUs available
    if n_gpus > 1:
        print(f"Using DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Optionally load checkpoint
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Load encoder weights only (skip regressor and cross-attention if architecture changed)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        # Filter: only load encoder weights
        encoder_dict = {k: v for k, v in pretrained_dict.items() 
                       if 'pocket_encoder' in k or 'ligand_encoder' in k}
        
        # Update model dict with encoder weights
        model_dict.update(encoder_dict)
        
        # Handle DataParallel
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_dict, strict=False)
        else:
            model.load_state_dict(model_dict, strict=False)
        
        print(f"Loaded encoder weights from checkpoint ({len(encoder_dict)} keys).")
        
        # Optionally freeze encoders after loading
        if args.freeze_encoders:
            if isinstance(model, nn.DataParallel):
                model.module.freeze_encoders()
            else:
                model.freeze_encoders()
        
        # Don't load optimizer/scheduler state when architecture changed
        # (freeze_encoders or cross_attention means different trainable params)
        optimizer_state = None
        scheduler_state = None
        print("Skipping optimizer/scheduler state (architecture changed).")
    else:
        optimizer_state = None
        scheduler_state = None

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Early stopping setup
    patience = args.patience
    patience_counter = 0
    
    # Setup optimizer and scheduler
    # Use higher learning rate for unfrozen parameters (cross-attention, regressor)
    if args.freeze_encoders:
        # Only train cross-attention and regressor with higher LR
        if isinstance(model, nn.DataParallel):
            trainable_params = [p for p in model.module.parameters() if p.requires_grad]
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=args.lr * 10,  # 10x higher LR for unfrozen parts
            weight_decay=args.weight_decay
        )
        print(f"Using higher learning rate for unfrozen parameters: {args.lr * 10}")
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    # Don't load optimizer state when fine-tuning with different architecture

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    # Don't load scheduler state when fine-tuning with different architecture
    
    # Training loop
    import time
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train (with optional contrastive learning)
        train_loss, train_mse, train_cl, train_time, diagnostics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            use_contrastive=args.use_contrastive,
            contrastive_alpha=args.contrastive_alpha,
            positive_augmentation=args.positive_augmentation,
            chemical_negatives_rate=args.chemical_negatives_rate
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_mse', train_mse, epoch)
        if args.use_contrastive:
            writer.add_scalar('Loss/train_contrastive', train_cl, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Metrics/pearson_r', val_metrics['pearson_r'], epoch)
        writer.add_scalar('Metrics/spearman_r', val_metrics['spearman_r'], epoch)
        writer.add_scalar('Metrics/mae', val_metrics['mae'], epoch)
        writer.add_scalar('Metrics/rmse', val_metrics['rmse'], epoch)
        writer.add_scalar('Metrics/sd', val_metrics['sd'], epoch)
        writer.add_scalar('Metrics/c_index', val_metrics['c_index'], epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        writer.add_scalar('Time/train_epoch', train_time, epoch)
        
        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        if args.use_contrastive:
            print(f"  Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, CL: {train_cl:.4f}) | Time: {train_time:.1f}s")
        else:
            print(f"  Train Loss: {train_loss:.4f} | Time: {train_time:.1f}s")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  R (Pearson): {val_metrics['pearson_r']:.4f} | R (Spearman): {val_metrics['spearman_r']:.4f}")
        print(f"  RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f} | SD: {val_metrics['sd']:.4f}")
        print(f"  CI (C-Index): {val_metrics['c_index']:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Print diagnostics
        if args.use_contrastive and diagnostics:
            print(f"  Gradient Norms: Encoder={diagnostics['grad_norms']['encoder']:.2f}, "
                  f"Attention={diagnostics['grad_norms']['attention']:.2f}, "
                  f"Proj={diagnostics['grad_norms']['projection']:.2f}, "
                  f"CL_Proj={diagnostics['grad_norms']['contrastive_proj']:.2f}")
            print(f"  Embedding Geometry: Pos_sim={diagnostics['cl_diagnostics']['pos_sim_p']:.3f}/"
                  f"{diagnostics['cl_diagnostics']['pos_sim_l']:.3f}, "
                  f"Neg_sim={diagnostics['cl_diagnostics']['neg_sim_p']:.3f}/"
                  f"{diagnostics['cl_diagnostics']['neg_sim_l']:.3f}, "
                  f"Norm={diagnostics['activation_stats']['embedding_norm']:.3f}")
        
        # Save best model and check early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'embedding_dim': args.embedding_dim,
                'temperature': args.temperature,
                'config': vars(args)
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'embedding_dim': args.embedding_dim,
                'temperature': args.temperature,
                'config': vars(args)
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    writer.close()
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    # Evaluate on test set with best model
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    # Load best model
    best_checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    
    # Handle DataParallel state dict (remove 'module.' prefix if present)
    state_dict = best_checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    print(f"Loaded best model from epoch {best_checkpoint['epoch']}")
    
    # Evaluate on test set
    test_metrics = validate(model, test_loader, device)
    
    # Print test results
    print("\nTest Set Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  R (Pearson): {test_metrics['pearson_r']:.4f}")
    print(f"  R (Spearman): {test_metrics['spearman_r']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  SD: {test_metrics['sd']:.4f}")
    print(f"  CI (C-Index): {test_metrics['c_index']:.4f}")
    
    # Save comprehensive test results
    test_results = {
        'timestamp': timestamp,
        'best_epoch': int(best_checkpoint['epoch']),
        'dataset': {
            'file': args.data_file,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset)
        },
        'model_config': {
            'embedding_dim': args.embedding_dim,
            'dropout': args.dropout,
            'use_cross_attention': args.use_cross_attention,
            'num_heads': args.num_heads,
            'pocket_channels': pocket_channels,
            'ligand_channels': ligand_channels,
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'training_config': {
            'epochs_trained': int(best_checkpoint['epoch']),
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'freeze_encoders': args.freeze_encoders,
            'total_training_time_seconds': float(total_training_time),
            'training_time_formatted': f"{hours}h {minutes}m {seconds}s"
        },
        'test_metrics': {
            'loss': float(test_metrics['loss']),
            'pearson_r': float(test_metrics['pearson_r']),
            'spearman_r': float(test_metrics['spearman_r']),
            'rmse': float(test_metrics['rmse']),
            'mae': float(test_metrics['mae']),
            'sd': float(test_metrics['sd']),
            'c_index': float(test_metrics['c_index'])
        }
    }
    
    # Save to run-specific file
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Append to global results log for tracking across runs
    global_log = Path(args.output_dir) / 'all_test_results.txt'
    with open(global_log, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Run: {timestamp}\n")
        f.write(f"Dataset: {args.data_file}\n")
        f.write(f"Model: embedding_dim={args.embedding_dim}, dropout={args.dropout}, cross_attn={args.use_cross_attention}, heads={args.num_heads}\n")
        f.write(f"Training: epochs={best_checkpoint['epoch']}, batch_size={args.batch_size}, lr={args.lr}\n")
        f.write(f"Training Time: {hours}h {minutes}m {seconds}s ({total_training_time:.1f}s)\n")
        f.write(f"Parameters: {trainable_params:,} trainable / {total_params:,} total\n")
        f.write(f"\nTest Results:\n")
        f.write(f"  R (Pearson):  {test_metrics['pearson_r']:.4f}\n")
        f.write(f"  R (Spearman): {test_metrics['spearman_r']:.4f}\n")
        f.write(f"  RMSE:         {test_metrics['rmse']:.4f}\n")
        f.write(f"  MAE:          {test_metrics['mae']:.4f}\n")
        f.write(f"  SD:           {test_metrics['sd']:.4f}\n")
        f.write(f"  CI:           {test_metrics['c_index']:.4f}\n")
        f.write(f"  Loss:         {test_metrics['loss']:.4f}\n")
    
    print(f"\nTraining complete! Results saved to {output_dir}")
    print(f"Test results saved to {output_dir / 'test_results.json'}")
    print(f"Results appended to {global_log}")


def main():
    parser = argparse.ArgumentParser(description='Train ESP supervised binding affinity prediction model')
    
    # Data parameters
    parser.add_argument('--data-file', type=str, 
                        default='/home/patrick/Desktop/ESPCL/data/voxelized_aa_32_16_vdw2.5_filtered.pt',
                        help='Path to paired dataset file')
    parser.add_argument('--batch-size', type=int, default=120,
                        help='Batch size for training')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Fraction of data for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model parameters
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature parameter (not used in regression mode)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate for regularization')
    parser.add_argument('--freeze-encoders', action='store_true',
                        help='Freeze encoder weights (for fine-tuning pretrained models)')
    parser.add_argument('--use-cross-attention', action='store_true',
                        help='Use cross-attention between pocket and ligand embeddings')
    parser.add_argument('--use-pairwise-attention', action='store_true',
                        help='Use pairwise attention (AlphaFold-style) instead of cross-attention')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads for cross-attention')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation (no rotation/flip)')
    
    # Masking parameters
    parser.add_argument('--channel-mask-ratio', type=float, default=0.0,
                        help='Ratio of channels to mask during training (0-1)')
    parser.add_argument('--spatial-mask-ratio', type=float, default=0.0,
                        help='Ratio of spatial patches to mask during training (0-1)')
    parser.add_argument('--mask-patch-size', type=int, default=4,
                        help='Size of cubic patches for spatial masking')
    
    # Contrastive learning parameters
    parser.add_argument('--use-contrastive', action='store_true',
                        help='Use contrastive learning with dual-view masking')
    parser.add_argument('--contrastive-alpha', type=float, default=0.1,
                        help='Weight for contrastive loss in combined loss (MSE + alpha*CL)')
    parser.add_argument('--contrastive-temperature', type=float, default=0.1,
                        help='Temperature for InfoNCE contrastive loss')
    parser.add_argument('--positive-augmentation', type=str, default='masking',
                        choices=['masking', 'rotation', 'rotation+masking'],
                        help='Type of augmentation for positive pairs: masking (channel+spatial), rotation (90deg rotations+flips), or rotation+masking (both)')
    parser.add_argument('--chemical-negatives-rate', type=float, default=0.1,
                        help='Fraction of batch to corrupt for chemical hard negatives (0-1)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, 
                        default='/home/patrick/Desktop/ESPCL2/results',
                        help='Output directory for results')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Pretrained/Resume options
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume or initialize model weights')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
