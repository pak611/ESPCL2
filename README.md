# ESPCL2: Electrostatic Potential Contrastive Learning for Protein-Ligand Binding

Deep learning framework for protein-ligand binding affinity prediction using 3D electrostatic potential (ESP) voxel representations and contrastive learning.

## Overview

This project implements a 3D convolutional neural network that learns from dual protein-ligand electrostatic potential representations. The model uses contrastive learning with pharmacologically-informed pair selection to capture subtle structural features determining binding affinity and functional outcomes.

## Key Features

- **3D Voxel Representations**: 32×32×32 grids with 19 channels (9 protein ESP, 9 ligand ESP, 1 overlap)
- **Contrastive Learning**: Multiple strategies including dual-view masking, chemical negatives, and action-type-based pairs
- **Pharmacological Labels**: GLASS dataset integration with agonist/antagonist annotations
- **Multi-Dataset Evaluation**: Benchmarked on BindingDB, DAVIS, KIBA, and GLASS
- **Cross-Dataset Generalization**: Evaluation across different protein families

## Project Structure

```
ESPCL2/
├── models/
│   ├── esp_jointnet.py              # Main model architecture
│   └── esp_jointnet_shared_proj.py  # Variant with shared projections
├── utils/
│   ├── dataset.py                   # Dataset loading and processing
│   ├── augmentation.py              # Data augmentation strategies
│   ├── cold_split.py                # Cold-start evaluation splits
│   └── similarity_split.py          # Similarity-based splitting
├── train.py                         # Main training script
├── train_with_similarity_split.py   # Training with similarity-aware splits
├── evaluate_cross_dataset.py        # Cross-dataset evaluation
├── evaluate_oneshot.py              # One-shot learning evaluation
├── extract_embeddings.py            # Extract learned representations
└── normalize_glass_dataset.py       # GLASS data normalization
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ESPCL2.git
cd ESPCL2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
# Add other dependencies as needed
```

## Data Preparation

The model expects preprocessed voxel data in PyTorch tensor format (.pt files). Each dataset should contain:
- `unified_voxels`: 3D voxel grids (N, 19, 32, 32, 32)
- `labels`: Binding affinity values (normalized to pIC50)
- `protein_ids`: Protein identifiers
- `ligand_ids`: Ligand identifiers
- `action_types`: Pharmacological annotations (GLASS only)

Place datasets in the `data/` directory:
```
data/
├── bindingdb_2016/
├── davis/
├── kiba/
└── glass/
```

## Training

### Basic Training
```bash
python train.py --dataset glass \
                --batch_size 32 \
                --epochs 100 \
                --lr 0.001
```

### Contrastive Learning
```bash
python train.py --dataset glass \
                --use_contrastive \
                --contrastive_weight 0.1 \
                --batch_size 32
```

### With Similarity-Based Splitting
```bash
python train_with_similarity_split.py --dataset davis \
                                       --similarity_threshold 0.7
```

## Evaluation

### Cross-Dataset Evaluation
```bash
python evaluate_cross_dataset.py --train_dataset davis \
                                  --test_dataset kiba
```

### One-Shot Learning
```bash
python evaluate_oneshot.py --dataset glass --n_shot 5
```

## Datasets

- **BindingDB 2016**: ~19k diverse protein-ligand complexes
- **DAVIS**: 442 kinases × 68 inhibitors (30k pairs)
- **KIBA**: 518 kinases × 612 inhibitors (118k pairs)
- **GLASS**: 53k GPCR-ligand interactions with action type annotations

All affinity values normalized to pIC50 scale: `pIC50 = -log₁₀(M)`

## Model Architecture

The ESP_JointNet uses parallel 3D CNN encoders for protein and ligand ESP grids, with optional contrastive learning objectives:
- Protein encoder: 5-layer 3D CNN → 256-d embedding
- Ligand encoder: 5-layer 3D CNN → 256-d embedding
- Fusion: Concatenation → FC layers → Affinity prediction
- Contrastive head: Projection layers for contrastive loss

## Citation

If you use this code, please cite:
```bibtex
@article{yourpaper2026,
  title={Electrostatic Potential Contrastive Learning for Protein-Ligand Binding},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
