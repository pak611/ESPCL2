# 2.1 Datasets

We evaluated performance across four benchmark datasets: BindingDB 2016 (general proteins), KIBA and DAVIS (kinases), and GLASS (GPCRs). Each dataset offers distinct characteristics in terms of scale, protein family coverage, and annotation depth.

**BindingDB 2016** is derived from the PDBbind database and contains ~19,000 protein-ligand complexes spanning diverse protein families, including a refined core set of 195 high-quality structures. Each entry includes PDB crystal structure, ligand SMILES, protein sequence, and binding affinity (Ki or Kd). We processed 17,798 pairs through our voxelization pipeline.

**KIBA** (Kinase Inhibitor BioActivity) measures interactions between 518 kinases and 612 inhibitors, yielding ~230,000 measurements. The dataset uses unified scores integrating Ki, Kd, and IC50 from multiple sources. We used the standard split of 98,545 training and 19,710 test pairs (118,036 total). The comprehensive interaction matrix captures kinase-inhibitor selectivity patterns across the kinome.

**DAVIS** profiles 442 kinases against 68 inhibitors using consistent Kd measurements (10 nM - 10 μM range). Unlike KIBA's composite scoring, DAVIS provides direct affinity measurements from a uniform experimental protocol. We processed 30,056 pairs (25,046 train, 5,010 test), with 37% exhibiting high affinity (Kd < 30 nM).

**GLASS** (GPCR-Ligand ASsociation) contains 1.1M GPCR-ligand interaction records. Beyond binding affinities, GLASS uniquely annotates each interaction with pharmacological action types: ACTIVATION (agonist), INHIBITION (antagonist), BINDING (affinity without functional data), MODULATION (allosteric), and others. From 333,265 filtered pairs, we successfully processed 53,437 into voxel representations (16% success rate after pocket detection and ESP computation). The action type distribution is: ACTIVATION (2,950), INHIBITION (2,268), BINDING (854), MODULATION (277), others (~100).

These action type labels enable pharmacologically-informed contrastive learning. Positive pairs consist of ligands with the same action type on the same target, while hard negatives pair ligands with opposing action types (agonist vs. antagonist) on the same target. This framework allows the model to learn structural features distinguishing functional outcomes beyond simple affinity.

## Data Processing

All datasets exhibit one-to-many mappings where proteins bind multiple ligands and vice versa—particularly pronounced in DAVIS and KIBA's interaction matrices. BindingDB spans diverse protein families with heterogeneous binding mechanisms, while DAVIS, KIBA, and GLASS are family-homogeneous but cover diverse interaction profiles within kinases or GPCRs.

Our processing pipeline involves: (1) pocket extraction via fpocket, (2) ESP surface computation for protein pockets and ligands, (3) voxelization to 32×32×32 grids with 19 channels (9 protein, 9 ligand, 1 overlap). Success rates vary by structure availability and computation feasibility. GLASS's 16% rate reflects these cumulative filters but yields a substantial final dataset representative of GPCR-ligand space.

All affinity values were normalized to pIC50 scale: pIC50 = -log₁₀(M), where M is molar concentration. This compresses the wide dynamic range (picomolar to millimolar) into a tractable scale. For example, 50 nM becomes pIC50 = 7.30, while 1 μM becomes 6.00. Final distributions typically span 4.0-10.0, with GLASS showing mean = 5.88, std = 1.69.
