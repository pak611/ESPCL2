"""Create train/test splits based on protein sequence similarity."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import pickle
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial


def get_kmers(sequence: str, k: int = 3) -> Set[str]:
    """Extract k-mers from sequence."""
    return set(sequence[i:i+k] for i in range(len(sequence) - k + 1))


def compute_sequence_similarity_fast(seq1: str, seq2: str, k: int = 3) -> float:
    """Fast approximate sequence similarity using k-mer Jaccard similarity.
    
    Args:
        seq1: First protein sequence
        seq2: Second protein sequence
        k: k-mer size (default 3 for trigrams)
        
    Returns:
        Approximate sequence similarity as a percentage (0-100)
    """
    if len(seq1) < k or len(seq2) < k:
        # For very short sequences, use simple identity
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
        return 100.0 * matches / min_len
    
    kmers1 = get_kmers(seq1, k)
    kmers2 = get_kmers(seq2, k)
    
    if not kmers1 or not kmers2:
        return 0.0
    
    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    
    # Jaccard similarity scaled to 0-100
    jaccard = intersection / union if union > 0 else 0.0
    
    # Scale to approximate sequence identity (empirically, Jaccard * 1.2 approximates identity well)
    return min(100.0, jaccard * 120.0)


def compute_row_similarities(args):
    """Compute similarities for a single row (parallelizable).
    
    Args:
        args: Tuple of (row_index, sequence_i, all_sequences, n)
        
    Returns:
        Tuple of (row_index, similarity_values)
    """
    i, seq_i, all_sequences, n = args
    similarities = np.zeros(n, dtype=np.float32)
    similarities[i] = 100.0  # Self-similarity
    
    kmers_i = get_kmers(seq_i, k=3)
    
    for j in range(i + 1, n):
        seq_j = all_sequences[j]
        kmers_j = get_kmers(seq_j, k=3)
        
        if kmers_i and kmers_j:
            intersection = len(kmers_i & kmers_j)
            union = len(kmers_i | kmers_j)
            jaccard = intersection / union if union > 0 else 0.0
            similarity = min(100.0, jaccard * 120.0)
        else:
            similarity = 0.0
        
        similarities[j] = similarity
    
    return i, similarities


def compute_similarity_matrix(protein_sequences: List[str], cache_file: str = None, n_workers: int = None) -> np.ndarray:
    """Compute pairwise sequence similarity matrix using fast k-mer method with multiprocessing.
    
    Args:
        protein_sequences: List of protein sequences
        cache_file: Optional cache file to save/load similarity matrix
        n_workers: Number of parallel workers (default: cpu_count())
        
    Returns:
        NxN similarity matrix where entry (i,j) is approximate % identity between seq i and j
    """
    n = len(protein_sequences)
    
    # Try to load from cache
    if cache_file and Path(cache_file).exists():
        print(f"Loading cached similarity matrix from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    if n_workers is None:
        n_workers = min(cpu_count(), 32)  # Cap at 32 to avoid overhead
    
    print(f"Computing similarity matrix for {n} sequences using {n_workers} workers...")
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    
    # Compute similarities in parallel
    print("Computing pairwise similarities in parallel...")
    
    with Pool(n_workers) as pool:
        args_list = [(i, protein_sequences[i], protein_sequences, n) for i in range(n)]
        results = list(tqdm(
            pool.imap_unordered(compute_row_similarities, args_list, chunksize=10),
            total=n,
            desc="Computing rows"
        ))
    
    # Fill the matrix
    for i, similarities in results:
        similarity_matrix[i] = similarities
        # Fill symmetric part
        for j in range(i + 1, n):
            similarity_matrix[j, i] = similarity_matrix[i, j]
    
    # Save to cache
    if cache_file:
        print(f"Saving similarity matrix to {cache_file}")
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(similarity_matrix, f)
    
    return similarity_matrix


def create_similarity_based_split(
    data_file: str,
    similarity_thresholds: List[float] = [30.0, 60.0],
    train_ratio: float = 0.7,
    cache_dir: str = "data/similarity_cache"
) -> Dict[str, Dict]:
    """Create train/test splits based on sequence similarity.
    
    Args:
        data_file: Path to dataset file
        similarity_thresholds: List of max similarity thresholds for test sets
        train_ratio: Ratio of data to use for training
        cache_dir: Directory to cache similarity matrix
        
    Returns:
        Dictionary with splits: {
            'train': {'indices': [...], 'labels': [...], ...},
            'test_30': {'indices': [...], 'labels': [...], ...},  # <30% similarity to train
            'test_60': {'indices': [...], 'labels': [...], ...},  # <60% similarity to train
        }
    """
    print(f"\nLoading dataset from {data_file}")
    data = torch.load(data_file)
    
    protein_sequences = data['protein_sequences']
    labels = data['labels']
    n_samples = len(labels)
    
    print(f"Dataset: {n_samples} samples")
    print(f"Unique proteins: {len(set(protein_sequences))}")
    
    # Get unique proteins and their indices
    unique_proteins = list(set(protein_sequences))
    n_proteins = len(unique_proteins)
    protein_to_idx = {seq: i for i, seq in enumerate(unique_proteins)}
    
    # Compute similarity matrix for unique proteins
    cache_file = Path(cache_dir) / f"{Path(data_file).stem}_similarity.pkl"
    similarity_matrix = compute_similarity_matrix(unique_proteins, str(cache_file))
    
    print(f"\nSimilarity statistics:")
    # Get upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n_proteins, k=1)
    similarities = similarity_matrix[triu_indices]
    print(f"  Mean similarity: {similarities.mean():.1f}%")
    print(f"  Median similarity: {np.median(similarities):.1f}%")
    print(f"  Max similarity: {similarities.max():.1f}%")
    print(f"  Min similarity: {similarities.min():.1f}%")
    
    # Split unique proteins into train/test
    n_train_proteins = int(train_ratio * n_proteins)
    
    # Shuffle proteins
    np.random.seed(42)
    protein_indices = np.random.permutation(n_proteins)
    
    train_protein_indices = protein_indices[:n_train_proteins]
    test_protein_indices = protein_indices[n_train_proteins:]
    
    print(f"\nProtein split:")
    print(f"  Train proteins: {n_train_proteins}")
    print(f"  Test proteins: {len(test_protein_indices)}")
    
    # For each test protein, compute max similarity to any training protein
    test_protein_max_similarities = []
    for test_idx in test_protein_indices:
        max_sim = similarity_matrix[test_idx, train_protein_indices].max()
        test_protein_max_similarities.append(max_sim)
    
    test_protein_max_similarities = np.array(test_protein_max_similarities)
    
    print(f"\nTest protein similarity to train:")
    print(f"  Mean max similarity: {test_protein_max_similarities.mean():.1f}%")
    print(f"  Median max similarity: {np.median(test_protein_max_similarities):.1f}%")
    
    # Create sample-level splits
    train_sample_indices = []
    test_splits = {f'test_{int(thresh)}': [] for thresh in similarity_thresholds}
    
    for sample_idx in range(n_samples):
        protein_seq = protein_sequences[sample_idx]
        protein_idx = protein_to_idx[protein_seq]
        
        if protein_idx in train_protein_indices:
            train_sample_indices.append(sample_idx)
        else:
            # Find max similarity to training set
            max_sim = similarity_matrix[protein_idx, train_protein_indices].max()
            
            # Assign to test splits based on similarity thresholds
            for thresh in similarity_thresholds:
                if max_sim < thresh:
                    test_splits[f'test_{int(thresh)}'].append(sample_idx)
    
    # Create result dictionary
    result = {
        'train': {
            'indices': train_sample_indices,
            'n_samples': len(train_sample_indices),
            'n_proteins': n_train_proteins
        }
    }
    
    for thresh in similarity_thresholds:
        key = f'test_{int(thresh)}'
        indices = test_splits[key]
        
        # Count unique proteins in this test set
        test_proteins = set(protein_sequences[i] for i in indices)
        
        result[key] = {
            'indices': indices,
            'n_samples': len(indices),
            'n_proteins': len(test_proteins),
            'threshold': thresh,
            'description': f'<{thresh}% sequence similarity to training set'
        }
    
    print(f"\nFinal splits:")
    print(f"  Train: {result['train']['n_samples']} samples, {result['train']['n_proteins']} proteins")
    for thresh in similarity_thresholds:
        key = f'test_{int(thresh)}'
        split = result[key]
        print(f"  {key}: {split['n_samples']} samples, {split['n_proteins']} proteins ({split['description']})")
    
    return result


def save_split_indices(split_dict: Dict, output_file: str):
    """Save split indices to file."""
    print(f"\nSaving split indices to {output_file}")
    torch.save(split_dict, output_file)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create similarity-based train/test splits')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to dataset file')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[30.0, 60.0],
                        help='Similarity thresholds for test sets')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of proteins to use for training')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for split indices')
    parser.add_argument('--cache-dir', type=str, default='data/similarity_cache',
                        help='Directory to cache similarity matrix')
    
    args = parser.parse_args()
    
    # Create splits
    splits = create_similarity_based_split(
        args.data_file,
        similarity_thresholds=args.thresholds,
        train_ratio=args.train_ratio,
        cache_dir=args.cache_dir
    )
    
    # Save splits
    if args.output is None:
        data_path = Path(args.data_file)
        args.output = str(data_path.parent / f"{data_path.stem}_similarity_splits.pt")
    
    save_split_indices(splits, args.output)
    
    print(f"\n✓ Done! Split indices saved to {args.output}")
