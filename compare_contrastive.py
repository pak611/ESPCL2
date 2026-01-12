"""
Compare ESPCL performance with and without contrastive learning on BindingDB 2016.
"""
import json
from pathlib import Path

print("="*80)
print("ESPCL Performance Comparison: With vs Without Contrastive Learning")
print("="*80)

# Results WITHOUT contrastive learning (already completed)
without_cl_dir = Path("/home/patrick/Desktop/ESPCL2/results/run_20260109_163343")
without_cl_results = json.load(open(without_cl_dir / "test_results.json"))

print("\n1. WITHOUT Contrastive Learning (Baseline)")
print("-" * 80)
print(f"Run: {without_cl_dir.name}")
print(f"Config: use_contrastive=False")
print(f"\nTest Results:")
print(f"  Pearson R:  {without_cl_results['test_metrics']['pearson_r']:.4f}")
print(f"  Spearman R: {without_cl_results['test_metrics']['spearman_r']:.4f}")
print(f"  RMSE:       {without_cl_results['test_metrics']['rmse']:.4f}")
print(f"  MAE:        {without_cl_results['test_metrics']['mae']:.4f}")
print(f"  CI:         {without_cl_results['test_metrics']['c_index']:.4f}")
print(f"  R²:         N/A")

# Find the most recent WITH contrastive learning run
results_dir = Path("/home/patrick/Desktop/ESPCL2/results")
with_cl_runs = []
for run_dir in results_dir.glob("run_*"):
    config_file = run_dir / "config.json"
    if config_file.exists():
        config = json.load(open(config_file))
        if config.get("use_contrastive", False):
            test_results_file = run_dir / "test_results.json"
            if test_results_file.exists():
                with_cl_runs.append((run_dir, test_results_file))

if with_cl_runs:
    # Use the most recent one
    with_cl_dir, with_cl_file = max(with_cl_runs, key=lambda x: x[0].name)
    with_cl_results = json.load(open(with_cl_file))
    
    print("\n2. WITH Contrastive Learning")
    print("-" * 80)
    print(f"Run: {with_cl_dir.name}")
    config = json.load(open(with_cl_dir / "config.json"))
    print(f"Config: use_contrastive=True, alpha={config['contrastive_alpha']}, temp={config['contrastive_temperature']}")
    print(f"Augmentation: {config.get('positive_augmentation', 'masking')}")
    print(f"\nTest Results:")
    print(f"  Pearson R:  {with_cl_results['test_metrics']['pearson_r']:.4f}")
    print(f"  Spearman R: {with_cl_results['test_metrics']['spearman_r']:.4f}")
    print(f"  RMSE:       {with_cl_results['test_metrics']['rmse']:.4f}")
    print(f"  MAE:        {with_cl_results['test_metrics']['mae']:.4f}")
    print(f"  CI:         {with_cl_results['test_metrics']['c_index']:.4f}")
    print(f"  R²:         N/A")
    
    # Comparison
    print("\n" + "="*80)
    print("IMPROVEMENT WITH CONTRASTIVE LEARNING")
    print("="*80)
    
    r_baseline = without_cl_results['test_metrics']['pearson_r']
    r_contrastive = with_cl_results['test_metrics']['pearson_r']
    r_improvement = ((r_contrastive - r_baseline) / r_baseline) * 100
    
    rmse_baseline = without_cl_results['test_metrics']['rmse']
    rmse_contrastive = with_cl_results['test_metrics']['rmse']
    rmse_improvement = ((rmse_baseline - rmse_contrastive) / rmse_baseline) * 100
    
    mae_baseline = without_cl_results['test_metrics']['mae']
    mae_contrastive = with_cl_results['test_metrics']['mae']
    mae_improvement = ((mae_baseline - mae_contrastive) / mae_baseline) * 100
    
    print(f"\nPearson R:  {r_baseline:.4f} → {r_contrastive:.4f} ({r_improvement:+.2f}%)")
    print(f"RMSE:       {rmse_baseline:.4f} → {rmse_contrastive:.4f} ({rmse_improvement:+.2f}%)")
    print(f"MAE:        {mae_baseline:.4f} → {mae_contrastive:.4f} ({mae_improvement:+.2f}%)")
    
    print("\n" + "="*80)
    
    # Create comparison table
    print("\n| Metric     | Without CL | With CL | Improvement |")
    print("|------------|-----------|---------|-------------|")
    print(f"| Pearson R  | {r_baseline:.4f}    | {r_contrastive:.4f}  | {r_improvement:+.2f}%      |")
    print(f"| RMSE       | {rmse_baseline:.4f}    | {rmse_contrastive:.4f}  | {rmse_improvement:+.2f}%      |")
    print(f"| MAE        | {mae_baseline:.4f}    | {mae_contrastive:.4f}  | {mae_improvement:+.2f}%      |")
    
else:
    print("\n2. WITH Contrastive Learning")
    print("-" * 80)
    print("Training in progress... Results not yet available.")
    print("\nTo check progress:")
    print("  tail -f /home/patrick/Desktop/ESPCL2/training_with_contrastive.log")
    print("\nOnce complete, run this script again to see the comparison.")

print("\n" + "="*80)
