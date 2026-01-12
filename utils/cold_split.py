"""
Cold-start and few-shot split utilities for drug-target binding affinity prediction
"""

import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json


def create_cold_protein_split(data, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    Create cold protein split - test on completely unseen proteins
    
    Args:
        data: Dict with keys ['protein_ids', 'ligand_ids', 'labels', ...]
        test_ratio: Fraction of proteins for testing
        val_ratio: Fraction of proteins for validation
        seed: Random seed
    
    Returns:
        train_idx, val_idx, test_idx: Lists of sample indices
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Get unique proteins
    unique_proteins = sorted(list(set(data['protein_ids'])))
    n_proteins = len(unique_proteins)
    
    # Split proteins into train/val/test
    n_test = int(n_proteins * test_ratio)
    n_val = int(n_proteins * val_ratio)
    n_train = n_proteins - n_test - n_val
    
    # Shuffle and split
    shuffled_proteins = unique_proteins.copy()
    random.shuffle(shuffled_proteins)
    
    train_proteins = set(shuffled_proteins[:n_train])
    val_proteins = set(shuffled_proteins[n_train:n_train+n_val])
    test_proteins = set(shuffled_proteins[n_train+n_val:])
    
    # Get sample indices for each split
    train_idx = [i for i, p in enumerate(data['protein_ids']) if p in train_proteins]
    val_idx = [i for i, p in enumerate(data['protein_ids']) if p in val_proteins]
    test_idx = [i for i, p in enumerate(data['protein_ids']) if p in test_proteins]
    
    print(f"Cold Protein Split:")
    print(f"  Train: {len(train_proteins)} proteins, {len(train_idx)} samples")
    print(f"  Val:   {len(val_proteins)} proteins, {len(val_idx)} samples")
    print(f"  Test:  {len(test_proteins)} proteins, {len(test_idx)} samples")
    
    return train_idx, val_idx, test_idx, {
        'train_proteins': list(train_proteins),
        'val_proteins': list(val_proteins),
        'test_proteins': list(test_proteins)
    }


def create_cold_ligand_split(data, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    Create cold ligand split - test on completely unseen ligands
    
    Args:
        data: Dict with keys ['protein_ids', 'ligand_ids', 'labels', ...]
        test_ratio: Fraction of ligands for testing
        val_ratio: Fraction of ligands for validation
        seed: Random seed
    
    Returns:
        train_idx, val_idx, test_idx: Lists of sample indices
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Get unique ligands
    unique_ligands = sorted(list(set(data['ligand_ids'])))
    n_ligands = len(unique_ligands)
    
    # Split ligands into train/val/test
    n_test = int(n_ligands * test_ratio)
    n_val = int(n_ligands * val_ratio)
    n_train = n_ligands - n_test - n_val
    
    # Shuffle and split
    shuffled_ligands = unique_ligands.copy()
    random.shuffle(shuffled_ligands)
    
    train_ligands = set(shuffled_ligands[:n_train])
    val_ligands = set(shuffled_ligands[n_train:n_train+n_val])
    test_ligands = set(shuffled_ligands[n_train+n_val:])
    
    # Get sample indices for each split
    train_idx = [i for i, l in enumerate(data['ligand_ids']) if l in train_ligands]
    val_idx = [i for i, l in enumerate(data['ligand_ids']) if l in val_ligands]
    test_idx = [i for i, l in enumerate(data['ligand_ids']) if l in test_ligands]
    
    print(f"Cold Ligand Split:")
    print(f"  Train: {len(train_ligands)} ligands, {len(train_idx)} samples")
    print(f"  Val:   {len(val_ligands)} ligands, {len(val_idx)} samples")
    print(f"  Test:  {len(test_ligands)} ligands, {len(test_idx)} samples")
    
    return train_idx, val_idx, test_idx, {
        'train_ligands': list(train_ligands),
        'val_ligands': list(val_ligands),
        'test_ligands': list(test_ligands)
    }


def create_cold_both_split(data, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    Create cold protein-ligand split - test on unseen protein AND ligand combinations
    Most challenging scenario
    
    Strategy:
    1. Split proteins into train/val/test sets
    2. For each protein set, split ligands independently
    3. This ensures test set has neither protein nor ligand seen in training
    
    Args:
        data: Dict with keys ['protein_ids', 'ligand_ids', 'labels', ...]
        test_ratio: Fraction for testing
        val_ratio: Fraction for validation
        seed: Random seed
    
    Returns:
        train_idx, val_idx, test_idx: Lists of sample indices
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # First split proteins
    unique_proteins = sorted(list(set(data['protein_ids'])))
    n_proteins = len(unique_proteins)
    
    n_test_p = int(n_proteins * test_ratio)
    n_val_p = int(n_proteins * val_ratio)
    
    shuffled_proteins = unique_proteins.copy()
    random.shuffle(shuffled_proteins)
    
    train_proteins = set(shuffled_proteins[:n_proteins-n_test_p-n_val_p])
    val_proteins = set(shuffled_proteins[n_proteins-n_test_p-n_val_p:n_proteins-n_test_p])
    test_proteins = set(shuffled_proteins[n_proteins-n_test_p:])
    
    # Then split ligands
    unique_ligands = sorted(list(set(data['ligand_ids'])))
    n_ligands = len(unique_ligands)
    
    n_test_l = int(n_ligands * test_ratio)
    n_val_l = int(n_ligands * val_ratio)
    
    shuffled_ligands = unique_ligands.copy()
    random.shuffle(shuffled_ligands)
    
    train_ligands = set(shuffled_ligands[:n_ligands-n_test_l-n_val_l])
    val_ligands = set(shuffled_ligands[n_ligands-n_test_l-n_val_l:n_ligands-n_test_l])
    test_ligands = set(shuffled_ligands[n_ligands-n_test_l:])
    
    # Assign samples based on both protein and ligand
    train_idx = [i for i, (p, l) in enumerate(zip(data['protein_ids'], data['ligand_ids']))
                 if p in train_proteins and l in train_ligands]
    
    val_idx = [i for i, (p, l) in enumerate(zip(data['protein_ids'], data['ligand_ids']))
               if (p in val_proteins or l in val_ligands) and 
                  not (p in test_proteins or l in test_ligands)]
    
    test_idx = [i for i, (p, l) in enumerate(zip(data['protein_ids'], data['ligand_ids']))
                if p in test_proteins or l in test_ligands]
    
    print(f"Cold Both (Protein + Ligand) Split:")
    print(f"  Train: {len(train_proteins)} proteins × {len(train_ligands)} ligands, {len(train_idx)} samples")
    print(f"  Val:   Partial overlap, {len(val_idx)} samples")
    print(f"  Test:  {len(test_proteins)} unseen proteins + {len(test_ligands)} unseen ligands, {len(test_idx)} samples")
    
    return train_idx, val_idx, test_idx, {
        'train_proteins': list(train_proteins),
        'val_proteins': list(val_proteins),
        'test_proteins': list(test_proteins),
        'train_ligands': list(train_ligands),
        'val_ligands': list(val_ligands),
        'test_ligands': list(test_ligands)
    }


def create_few_shot_episodes(data, n_episodes=100, k_shot=5, n_query=10, seed=42):
    """
    Create few-shot learning episodes for protein adaptation
    
    Each episode:
    - Sample a protein not in training set
    - Support set: k examples of that protein
    - Query set: n_query remaining examples
    
    Args:
        data: Dict with dataset
        n_episodes: Number of episodes to generate
        k_shot: Number of support examples per protein
        n_query: Number of query examples per protein
        seed: Random seed
    
    Returns:
        episodes: List of dicts with 'protein', 'support_idx', 'query_idx'
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Group samples by protein
    protein_to_samples = defaultdict(list)
    for i, protein in enumerate(data['protein_ids']):
        protein_to_samples[protein].append(i)
    
    # Filter proteins with enough samples
    valid_proteins = [p for p, samples in protein_to_samples.items() 
                     if len(samples) >= k_shot + n_query]
    
    if len(valid_proteins) < n_episodes:
        print(f"Warning: Only {len(valid_proteins)} proteins have {k_shot + n_query}+ samples")
        n_episodes = len(valid_proteins)
    
    episodes = []
    for episode_id in range(n_episodes):
        # Sample a protein
        protein = random.choice(valid_proteins)
        samples = protein_to_samples[protein].copy()
        random.shuffle(samples)
        
        # Split into support and query
        support_idx = samples[:k_shot]
        query_idx = samples[k_shot:k_shot+n_query]
        
        episodes.append({
            'episode_id': episode_id,
            'protein': protein,
            'support_idx': support_idx,
            'query_idx': query_idx,
            'n_support': len(support_idx),
            'n_query': len(query_idx)
        })
    
    print(f"\nCreated {len(episodes)} few-shot episodes:")
    print(f"  k-shot: {k_shot}")
    print(f"  n-query: {n_query}")
    print(f"  Valid proteins: {len(valid_proteins)}")
    
    return episodes


def save_split_indices(output_path, split_type, train_idx, val_idx, test_idx, metadata=None):
    """Save split indices to file"""
    split_data = {
        'split_type': split_type,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'metadata': metadata or {}
    }
    
    torch.save(split_data, output_path)
    print(f"\nSaved {split_type} split to: {output_path}")


def load_split_indices(split_path):
    """Load split indices from file"""
    split_data = torch.load(split_path)
    print(f"Loaded {split_data['split_type']} split:")
    print(f"  Train: {split_data['n_train']} samples")
    print(f"  Val:   {split_data['n_val']} samples")
    print(f"  Test:  {split_data['n_test']} samples")
    
    return split_data['train_idx'], split_data['val_idx'], split_data['test_idx'], split_data['metadata']


if __name__ == '__main__':
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create cold-start splits for drug-target dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset .pt file')
    parser.add_argument('--output-dir', type=str, default='./splits', help='Output directory for splits')
    parser.add_argument('--split-type', type=str, default='all', 
                       choices=['cold_protein', 'cold_ligand', 'cold_both', 'few_shot', 'all'],
                       help='Type of split to create')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--k-shot', type=int, default=5, help='Number of support examples for few-shot')
    parser.add_argument('--n-query', type=int, default=10, help='Number of query examples for few-shot')
    parser.add_argument('--n-episodes', type=int, default=100, help='Number of few-shot episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = torch.load(args.data)
    print(f"Loaded {len(data['ligand_ids'])} samples")
    print(f"  Unique proteins: {len(set(data['protein_ids']))}")
    print(f"  Unique ligands: {len(set(data['ligand_ids']))}")
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create splits
    if args.split_type in ['cold_protein', 'all']:
        print("\n" + "="*80)
        train_idx, val_idx, test_idx, metadata = create_cold_protein_split(
            data, args.test_ratio, args.val_ratio, args.seed
        )
        save_split_indices(
            f"{args.output_dir}/cold_protein_split.pt",
            'cold_protein', train_idx, val_idx, test_idx, metadata
        )
    
    if args.split_type in ['cold_ligand', 'all']:
        print("\n" + "="*80)
        train_idx, val_idx, test_idx, metadata = create_cold_ligand_split(
            data, args.test_ratio, args.val_ratio, args.seed
        )
        save_split_indices(
            f"{args.output_dir}/cold_ligand_split.pt",
            'cold_ligand', train_idx, val_idx, test_idx, metadata
        )
    
    if args.split_type in ['cold_both', 'all']:
        print("\n" + "="*80)
        train_idx, val_idx, test_idx, metadata = create_cold_both_split(
            data, args.test_ratio, args.val_ratio, args.seed
        )
        save_split_indices(
            f"{args.output_dir}/cold_both_split.pt",
            'cold_both', train_idx, val_idx, test_idx, metadata
        )
    
    if args.split_type in ['few_shot', 'all']:
        print("\n" + "="*80)
        episodes = create_few_shot_episodes(
            data, args.n_episodes, args.k_shot, args.n_query, args.seed
        )
        torch.save(episodes, f"{args.output_dir}/few_shot_episodes.pt")
        print(f"Saved few-shot episodes to: {args.output_dir}/few_shot_episodes.pt")
    
    print("\n" + "="*80)
    print("Done!")
