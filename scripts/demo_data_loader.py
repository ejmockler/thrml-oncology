#!/usr/bin/env python3
"""
Demo: Data Loading Pipeline for THRML Cancer Decision Support

This script demonstrates the complete data loading workflow:
1. Generate synthetic data with known causal structure
2. Discretize continuous values for categorical models
3. Prepare data for GeneNetworkModel
4. Validate and inspect the data

Run with: python3 scripts/demo_data_loader.py
"""

import sys
sys.path.insert(0, '/Users/noot/Documents/thrml-cancer-decision-support')

from core.data_loader import (
    generate_synthetic_data,
    discretize_values,
    prepare_model_input,
    validate_data,
    load_ccle_data
)
import numpy as np
import pandas as pd


def demo_basic_workflow():
    """Demonstrate basic data loading workflow"""
    print("=" * 70)
    print("DEMO: Basic Data Loading Workflow")
    print("=" * 70)

    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic data...")
    genes = ['EGFR', 'KRAS', 'BRAF', 'PIK3CA', 'TP53']
    data = generate_synthetic_data(
        genes,
        n_sensitive=30,
        n_resistant=20,
        seed=42
    )

    print(f"  Generated {len(data['cell_lines'])} cell lines")
    print(f"  Genes: {genes}")
    print(f"  Sensitive cells: {len(data['sensitive_idx'])}")
    print(f"  Resistant cells: {len(data['resistant_idx'])}")

    # Step 2: Inspect raw data
    print("\n[Step 2] Inspecting raw data...")
    print(f"  Methylation shape: {data['methylation'].shape}")
    print(f"  Expression shape: {data['expression'].shape}")
    print(f"\n  Methylation sample (first 3 cells, first 3 genes):")
    print(data['methylation'].iloc[:3, :3])
    print(f"\n  Expression sample (first 3 cells, first 3 genes):")
    print(data['expression'].iloc[:3, :3])

    # Step 3: Validate data
    print("\n[Step 3] Validating data structure...")
    try:
        validate_data(data)
        print("  ✓ Data validation passed")
    except ValueError as e:
        print(f"  ✗ Validation failed: {e}")
        return

    # Step 4: Discretize for THRML
    print("\n[Step 4] Preparing data for THRML model...")
    model_input = prepare_model_input(data, discretize=True, n_bins=3)

    print(f"  Discretized methylation shape: {model_input['methylation_discrete'].shape}")
    print(f"  Discretized expression shape: {model_input['expression_discrete'].shape}")

    print(f"\n  Discretized methylation sample (first 5 cells, first 3 genes):")
    print(model_input['methylation_discrete'].iloc[:5, :3])
    print(f"\n  Discretized expression sample (first 5 cells, first 3 genes):")
    print(model_input['expression_discrete'].iloc[:5, :3])

    # Step 5: Inspect ground truth
    print("\n[Step 5] Ground truth causal structure...")
    gt = data['ground_truth']

    print(f"\n  Total causal edges: {len(gt['edges'])}")
    print(f"\n  Methylation → Expression correlations:")
    for gene, (sens_corr, resist_corr) in gt['methylation_expression'].items():
        print(f"    {gene:10s}: sensitive={sens_corr:+.2f}, resistant={resist_corr:+.2f}")

    print(f"\n  Pathway cascades (expression → expression):")
    for source, target, strength in gt['pathway_cascade']:
        print(f"    {source:10s} → {target:10s}  (strength={strength:.2f})")

    # Step 6: Analyze group differences
    print("\n[Step 6] Analyzing sensitive vs resistant differences...")

    sens_idx = model_input['sensitive_idx']
    resist_idx = model_input['resistant_idx']

    print(f"\n  Mean discretized expression by group:")
    print(f"    {'Gene':<10s} {'Sensitive':<12s} {'Resistant':<12s} {'Difference':<12s}")
    print(f"    {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

    for gene in genes[:3]:  # Show first 3 genes
        sens_mean = model_input['expression_discrete'].loc[
            model_input['expression_discrete'].index[sens_idx], gene
        ].mean()
        resist_mean = model_input['expression_discrete'].loc[
            model_input['expression_discrete'].index[resist_idx], gene
        ].mean()
        diff = resist_mean - sens_mean

        print(f"    {gene:<10s} {sens_mean:<12.3f} {resist_mean:<12.3f} {diff:+12.3f}")

    return model_input


def demo_ccle_fallback():
    """Demonstrate CCLE loading with synthetic fallback"""
    print("\n\n" + "=" * 70)
    print("DEMO: CCLE Data Loading (with synthetic fallback)")
    print("=" * 70)

    genes = ['EGFR', 'KRAS', 'BRAF']

    print("\n[Step 1] Attempting to load CCLE data...")
    print("  (This will fall back to synthetic data if CCLE files not found)")

    data = load_ccle_data(genes, data_dir='data/ccle', fallback_synthetic=True)

    print(f"\n  Loaded {len(data['cell_lines'])} cell lines")
    print(f"  Genes: {list(data['methylation'].columns)}")
    print(f"  Data source: {'CCLE' if data['ground_truth'] is None else 'Synthetic'}")


def demo_discretization_methods():
    """Demonstrate different discretization scenarios"""
    print("\n\n" + "=" * 70)
    print("DEMO: Discretization Edge Cases")
    print("=" * 70)

    # Case 1: Normal continuous data
    print("\n[Case 1] Normal continuous data")
    df_normal = pd.DataFrame({
        'gene_A': np.random.randn(20) * 2 + 5
    })
    disc_normal = discretize_values(df_normal)
    print(f"  Input range: [{df_normal['gene_A'].min():.2f}, {df_normal['gene_A'].max():.2f}]")
    print(f"  Discretized values: {sorted(disc_normal['gene_A'].unique())}")
    print(f"  Value counts: {disc_normal['gene_A'].value_counts().sort_index().to_dict()}")

    # Case 2: Data with NaNs
    print("\n[Case 2] Data with missing values")
    df_nan = pd.DataFrame({
        'gene_B': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0]
    })
    disc_nan = discretize_values(df_nan)
    print(f"  Input NaNs: {df_nan['gene_B'].isna().sum()}")
    print(f"  Output NaNs: {disc_nan['gene_B'].isna().sum()}")
    print(f"  Discretized values: {disc_nan['gene_B'].tolist()}")

    # Case 3: Constant values
    print("\n[Case 3] Constant values")
    df_const = pd.DataFrame({
        'gene_C': [5.0] * 10
    })
    disc_const = discretize_values(df_const)
    print(f"  Input unique values: {df_const['gene_C'].nunique()}")
    print(f"  Discretized unique values: {disc_const['gene_C'].nunique()}")
    print(f"  Discretized value: {disc_const['gene_C'].unique()}")

    # Case 4: Few unique values
    print("\n[Case 4] Few unique values")
    df_few = pd.DataFrame({
        'gene_D': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    })
    disc_few = discretize_values(df_few)
    print(f"  Input unique values: {sorted(df_few['gene_D'].unique())}")
    print(f"  Discretized values: {sorted(disc_few['gene_D'].unique())}")
    print(f"  Discretized distribution: {disc_few['gene_D'].value_counts().sort_index().to_dict()}")


def demo_integration_example():
    """Show how to use with GeneNetworkModel (conceptual)"""
    print("\n\n" + "=" * 70)
    print("DEMO: Integration with GeneNetworkModel")
    print("=" * 70)

    print("\n[Conceptual Example] Using data_loader with THRML model")
    print("""
# 1. Load and prepare data
from core.data_loader import generate_synthetic_data, prepare_model_input
from core.thrml_model import GeneNetworkModel

genes = ['EGFR', 'KRAS', 'BRAF']

# Generate synthetic data
data = generate_synthetic_data(genes, n_sensitive=25, n_resistant=25)

# Prepare for model
model_input = prepare_model_input(data, discretize=True)

# 2. Create THRML model with INDRA priors
from core.indra_client import INDRAClient

indra = INDRAClient()
prior_network = indra.get_pathway_network(genes)

model = GeneNetworkModel(genes, prior_network, n_states=3)

# 3. Test causal directions
result = model.test_causal_direction(
    'EGFR', 'KRAS',
    data=model_input,
    n_samples=1000
)

print(f"Direction: {result['direction']}")
print(f"ΔF = {result['delta_F']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")

# 4. Compare to ground truth
gt = data['ground_truth']
print(f"Ground truth: {gt}")
    """)


def main():
    """Run all demonstrations"""
    print("\n")
    print("█" * 70)
    print("   DATA LOADER PIPELINE DEMONSTRATION")
    print("   THRML Cancer Decision Support System")
    print("█" * 70)

    # Run demonstrations
    model_input = demo_basic_workflow()
    demo_ccle_fallback()
    demo_discretization_methods()
    demo_integration_example()

    # Final summary
    print("\n\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("""
Key Features:
  ✓ Synthetic data generation with known causal structure
  ✓ Quantile-based discretization to {0, 1, 2}
  ✓ Handles edge cases (NaNs, constant values, few unique values)
  ✓ CCLE data loading with synthetic fallback
  ✓ Data validation and quality checks
  ✓ Ground truth causal edges for validation
  ✓ Ready for integration with GeneNetworkModel

Next Steps:
  1. Run: python3 scripts/test_data_loader.py
  2. Use in inference pipeline
  3. Load real CCLE data (optional)
  4. Integrate with THRML causal inference
    """)


if __name__ == '__main__':
    main()
