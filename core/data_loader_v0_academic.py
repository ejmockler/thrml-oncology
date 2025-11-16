"""
Data Loading Pipeline for Thermodynamic Cancer Decision Support
Handles discretization, synthetic data generation, and CCLE data loading.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
from pathlib import Path


def discretize_values(data: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """
    Discretize continuous values to {0, 1, 2} using quantile-based binning.

    This function converts continuous gene expression or methylation values into
    discrete states (low/medium/high) suitable for categorical energy models.

    Args:
        data: DataFrame [samples × genes] with continuous values
        n_bins: Number of bins (default 3 for low/med/high)

    Returns:
        Discretized DataFrame with same shape, values in {0, 1, 2}

    Raises:
        ValueError: If n_bins is not supported or data is empty

    Examples:
        >>> df = pd.DataFrame({'EGFR': [1.2, 3.4, 5.6, 7.8, 9.0]})
        >>> discretized = discretize_values(df, n_bins=3)
        >>> set(discretized['EGFR'].unique()) <= {0, 1, 2}
        True
    """
    if data.empty:
        raise ValueError("Input DataFrame is empty")

    if n_bins != 3:
        warnings.warn(f"n_bins={n_bins} specified, but only n_bins=3 is fully tested")

    discretized = pd.DataFrame(index=data.index, columns=data.columns)

    for col in data.columns:
        values = data[col]

        # Handle edge cases
        if values.isna().all():
            warnings.warn(f"Column {col} contains only NaN values, filling with 0")
            discretized[col] = 0
            continue

        # Remove NaNs for binning
        valid_mask = ~values.isna()
        valid_values = values[valid_mask]

        # Check if all values are the same
        if valid_values.nunique() == 1:
            warnings.warn(f"Column {col} has constant value, assigning to bin 1")
            discretized.loc[valid_mask, col] = 1
            discretized.loc[~valid_mask, col] = 0  # Assign NaNs to lowest bin
            continue

        try:
            # Use quantile-based binning
            # Labels 0, 1, 2 correspond to low, medium, high
            binned = pd.qcut(
                valid_values,
                q=n_bins,
                labels=list(range(n_bins)),
                duplicates='drop'  # Handle duplicate bin edges
            )
            discretized.loc[valid_mask, col] = binned.astype(int)

            # Fill NaNs with lowest bin
            discretized.loc[~valid_mask, col] = 0

        except ValueError as e:
            # If qcut fails (e.g., too few unique values), use uniform binning
            warnings.warn(f"Column {col}: qcut failed ({e}), using value_counts binning")

            # Assign bins based on value ranking
            rank = valid_values.rank(method='first')
            n_valid = len(valid_values)
            bins = pd.cut(rank, bins=n_bins, labels=list(range(n_bins)), include_lowest=True)
            discretized.loc[valid_mask, col] = bins.astype(int)
            discretized.loc[~valid_mask, col] = 0

    return discretized.astype(int)


def generate_synthetic_data(
    genes: List[str],
    n_sensitive: int = 25,
    n_resistant: int = 25,
    seed: int = 42
) -> Dict:
    """
    Generate synthetic methylation/expression data with known causal structure.

    Creates biologically-inspired synthetic data where:
    - Sensitive cells: high EGFR methylation → low EGFR expression (responds to therapy)
    - Resistant cells: bypass pathway activation (e.g., KRAS/BRAF mutations)

    This provides ground truth for validating causal inference algorithms.

    Args:
        genes: List of gene symbols to simulate
        n_sensitive: Number of therapy-sensitive cell lines
        n_resistant: Number of therapy-resistant cell lines
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            'methylation': DataFrame [samples × genes] - continuous methylation β-values
            'expression': DataFrame [samples × genes] - continuous log2 expression
            'cell_lines': List of cell line IDs
            'sensitive_idx': List of sensitive cell line indices
            'resistant_idx': List of resistant cell line indices
            'ground_truth': Dict of true causal edges for validation

    Example:
        >>> data = generate_synthetic_data(['EGFR', 'KRAS', 'BRAF'], n_sensitive=10, n_resistant=10)
        >>> data['methylation'].shape
        (20, 3)
        >>> len(data['ground_truth'])
        3
    """
    np.random.seed(seed)

    n_total = n_sensitive + n_resistant
    n_genes = len(genes)

    # Initialize arrays
    methylation = np.zeros((n_total, n_genes))
    expression = np.zeros((n_total, n_genes))

    # Generate cell line IDs
    cell_lines = [f"CELL_{i:03d}" for i in range(n_total)]
    sensitive_idx = list(range(n_sensitive))
    resistant_idx = list(range(n_sensitive, n_total))

    # Define ground truth causal structure
    # Format: (gene_from, gene_to, effect_type, strength)
    ground_truth = {
        'edges': [],
        'methylation_expression': {},  # gene -> (correlation in sensitive, correlation in resistant)
        'pathway_cascade': []
    }

    # --- Generate data for each gene ---
    for g_idx, gene in enumerate(genes):

        if gene == 'EGFR' or g_idx == 0:  # Primary target gene
            # SENSITIVE cells: high methylation → low expression
            # Methylation: higher values
            methylation[sensitive_idx, g_idx] = np.random.beta(5, 2, n_sensitive) * 0.8 + 0.2  # β ∈ [0.2, 1.0]
            # Expression: anti-correlated with methylation + noise
            expression[sensitive_idx, g_idx] = (
                -0.8 * methylation[sensitive_idx, g_idx] +
                np.random.normal(0, 0.15, n_sensitive) + 1.0
            )

            # RESISTANT cells: low methylation → high expression (escape mechanism)
            methylation[resistant_idx, g_idx] = np.random.beta(2, 5, n_resistant) * 0.6  # β ∈ [0, 0.6]
            expression[resistant_idx, g_idx] = (
                -0.5 * methylation[resistant_idx, g_idx] +
                np.random.normal(0, 0.15, n_resistant) + 1.5
            )

            # Record ground truth
            ground_truth['methylation_expression'][gene] = (-0.8, -0.5)
            ground_truth['edges'].append((f'{gene}_meth', f'{gene}_expr', 'repression', 0.8))

        elif gene == 'KRAS' or g_idx == 1:  # Bypass pathway gene
            # SENSITIVE cells: correlated with EGFR expression (downstream)
            if g_idx > 0:
                expression[sensitive_idx, g_idx] = (
                    0.6 * expression[sensitive_idx, 0] +
                    np.random.normal(0, 0.2, n_sensitive)
                )
                ground_truth['pathway_cascade'].append((genes[0], gene, 0.6))
            else:
                expression[sensitive_idx, g_idx] = np.random.normal(0.5, 0.3, n_sensitive)

            # Methylation: moderate, less variable
            methylation[sensitive_idx, g_idx] = np.random.beta(3, 3, n_sensitive) * 0.7 + 0.15

            # RESISTANT cells: ACTIVATED bypass (high expression regardless of EGFR)
            expression[resistant_idx, g_idx] = np.random.beta(6, 2, n_resistant) * 2.0 + 1.0
            methylation[resistant_idx, g_idx] = np.random.beta(2, 4, n_resistant) * 0.5

            ground_truth['methylation_expression'][gene] = (-0.3, -0.4)
            ground_truth['edges'].append((f'{gene}_meth', f'{gene}_expr', 'repression', 0.4))

        elif gene == 'BRAF' or g_idx == 2:  # Another bypass pathway
            # SENSITIVE cells: correlated with upstream genes
            if g_idx > 1:
                expression[sensitive_idx, g_idx] = (
                    0.5 * expression[sensitive_idx, 1] +
                    0.3 * expression[sensitive_idx, 0] +
                    np.random.normal(0, 0.25, n_sensitive)
                )
                ground_truth['pathway_cascade'].append((genes[1], gene, 0.5))
            else:
                expression[sensitive_idx, g_idx] = np.random.normal(0.4, 0.3, n_sensitive)

            methylation[sensitive_idx, g_idx] = np.random.beta(3, 3, n_sensitive) * 0.6 + 0.2

            # RESISTANT cells: sometimes co-activated with KRAS
            expression[resistant_idx, g_idx] = (
                0.4 * expression[resistant_idx, min(1, n_genes-1)] +
                np.random.beta(4, 3, n_resistant) * 1.5
            )
            methylation[resistant_idx, g_idx] = np.random.beta(2, 3, n_resistant) * 0.5

            ground_truth['methylation_expression'][gene] = (-0.35, -0.3)
            ground_truth['edges'].append((f'{gene}_meth', f'{gene}_expr', 'repression', 0.35))

        else:  # Generic gene
            # SENSITIVE: moderate correlation structure
            methylation[sensitive_idx, g_idx] = np.random.beta(3, 3, n_sensitive) * 0.8 + 0.1
            expression[sensitive_idx, g_idx] = (
                -0.4 * methylation[sensitive_idx, g_idx] +
                np.random.normal(0.5, 0.3, n_sensitive)
            )

            # RESISTANT: weaker correlation
            methylation[resistant_idx, g_idx] = np.random.beta(3, 3, n_resistant) * 0.7 + 0.15
            expression[resistant_idx, g_idx] = (
                -0.2 * methylation[resistant_idx, g_idx] +
                np.random.normal(0.6, 0.35, n_resistant)
            )

            ground_truth['methylation_expression'][gene] = (-0.4, -0.2)
            ground_truth['edges'].append((f'{gene}_meth', f'{gene}_expr', 'repression', 0.4))

    # Normalize expression to reasonable log2 scale (e.g., -2 to 3)
    expression = (expression - expression.mean()) / expression.std() * 1.5 + 0.5

    # Ensure methylation is in [0, 1]
    methylation = np.clip(methylation, 0, 1)

    # Create DataFrames
    meth_df = pd.DataFrame(methylation, columns=genes, index=cell_lines)
    expr_df = pd.DataFrame(expression, columns=genes, index=cell_lines)

    return {
        'methylation': meth_df,
        'expression': expr_df,
        'cell_lines': cell_lines,
        'sensitive_idx': sensitive_idx,
        'resistant_idx': resistant_idx,
        'ground_truth': ground_truth
    }


def load_ccle_data(
    genes: List[str],
    data_dir: str = 'data/ccle',
    fallback_synthetic: bool = True
) -> Dict:
    """
    Load CCLE (Cancer Cell Line Encyclopedia) methylation and expression data.

    Attempts to load real CCLE data from files. If files don't exist and
    fallback_synthetic=True, generates synthetic data instead.

    Expected file format:
        - {data_dir}/CCLE_methylation.csv: rows=cell lines, cols=genes
        - {data_dir}/CCLE_expression.csv: rows=cell lines, cols=genes
        - {data_dir}/CCLE_annotations.csv: cell line metadata (optional)

    Args:
        genes: List of gene symbols to load
        data_dir: Directory containing CCLE data files
        fallback_synthetic: If True, generate synthetic data when files missing

    Returns:
        Dictionary with same structure as generate_synthetic_data():
            'methylation', 'expression', 'cell_lines', 'sensitive_idx',
            'resistant_idx', 'ground_truth' (None for real data)

    Raises:
        FileNotFoundError: If data files missing and fallback_synthetic=False

    Example:
        >>> data = load_ccle_data(['EGFR', 'KRAS'], data_dir='data/ccle')
        >>> 'methylation' in data
        True
    """
    data_path = Path(data_dir)
    meth_file = data_path / 'CCLE_methylation.csv'
    expr_file = data_path / 'CCLE_expression.csv'
    annot_file = data_path / 'CCLE_annotations.csv'

    # Check if files exist
    files_exist = meth_file.exists() and expr_file.exists()

    if not files_exist:
        if fallback_synthetic:
            warnings.warn(
                f"CCLE data files not found in {data_dir}, generating synthetic data instead. "
                f"To use real data, download CCLE files from https://depmap.org/portal/download/"
            )
            return generate_synthetic_data(genes, n_sensitive=25, n_resistant=25)
        else:
            raise FileNotFoundError(
                f"CCLE data files not found in {data_dir}. "
                f"Expected: {meth_file} and {expr_file}"
            )

    # Load data
    try:
        meth_df = pd.read_csv(meth_file, index_col=0)
        expr_df = pd.read_csv(expr_file, index_col=0)

        # Filter to requested genes
        available_genes = set(genes)
        meth_genes = set(meth_df.columns) & available_genes
        expr_genes = set(expr_df.columns) & available_genes
        common_genes = sorted(meth_genes & expr_genes)

        if len(common_genes) < len(genes):
            missing = set(genes) - set(common_genes)
            warnings.warn(f"Genes not found in CCLE data: {missing}")

        if not common_genes:
            raise ValueError(f"None of the requested genes found in CCLE data")

        # Align samples (use intersection of cell lines)
        common_cells = sorted(set(meth_df.index) & set(expr_df.index))

        if not common_cells:
            raise ValueError("No overlapping cell lines between methylation and expression data")

        meth_df = meth_df.loc[common_cells, common_genes]
        expr_df = expr_df.loc[common_cells, common_genes]

        # Load annotations if available
        sensitive_idx = []
        resistant_idx = []

        if annot_file.exists():
            annot_df = pd.read_csv(annot_file, index_col=0)
            # Assume there's a 'drug_response' column indicating sensitivity
            # This is dataset-specific and may need adjustment
            if 'drug_response' in annot_df.columns:
                annot_df = annot_df.loc[annot_df.index.intersection(common_cells)]
                sensitive_idx = [i for i, cell in enumerate(common_cells)
                                if cell in annot_df.index and annot_df.loc[cell, 'drug_response'] == 'sensitive']
                resistant_idx = [i for i, cell in enumerate(common_cells)
                                if cell in annot_df.index and annot_df.loc[cell, 'drug_response'] == 'resistant']

        # If no annotations or labels not found, use placeholder
        if not sensitive_idx and not resistant_idx:
            warnings.warn("No drug response annotations found, using placeholder split")
            n_cells = len(common_cells)
            sensitive_idx = list(range(n_cells // 2))
            resistant_idx = list(range(n_cells // 2, n_cells))

        return {
            'methylation': meth_df,
            'expression': expr_df,
            'cell_lines': common_cells,
            'sensitive_idx': sensitive_idx,
            'resistant_idx': resistant_idx,
            'ground_truth': None  # Real data has no ground truth
        }

    except Exception as e:
        if fallback_synthetic:
            warnings.warn(f"Error loading CCLE data: {e}. Using synthetic data instead.")
            return generate_synthetic_data(genes, n_sensitive=25, n_resistant=25)
        else:
            raise


def validate_data(data: Dict) -> bool:
    """
    Validate that data dictionary has correct structure and no critical issues.

    Checks:
        - Required keys present
        - DataFrames have matching dimensions
        - No all-NaN columns
        - Indices are consistent
        - Methylation values in reasonable range [0, 1]

    Args:
        data: Dictionary from generate_synthetic_data() or load_ccle_data()

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails with description of issue

    Example:
        >>> data = generate_synthetic_data(['EGFR'], n_sensitive=5, n_resistant=5)
        >>> validate_data(data)
        True
    """
    # Check required keys
    required_keys = {'methylation', 'expression', 'cell_lines', 'sensitive_idx', 'resistant_idx'}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    meth_df = data['methylation']
    expr_df = data['expression']
    cell_lines = data['cell_lines']
    sensitive_idx = data['sensitive_idx']
    resistant_idx = data['resistant_idx']

    # Check types
    if not isinstance(meth_df, pd.DataFrame):
        raise ValueError(f"'methylation' must be DataFrame, got {type(meth_df)}")
    if not isinstance(expr_df, pd.DataFrame):
        raise ValueError(f"'expression' must be DataFrame, got {type(expr_df)}")

    # Check shapes match
    if meth_df.shape != expr_df.shape:
        raise ValueError(
            f"Shape mismatch: methylation {meth_df.shape} vs expression {expr_df.shape}"
        )

    # Check indices match
    if not meth_df.index.equals(expr_df.index):
        raise ValueError("Methylation and expression DataFrames have different indices")

    # Check columns match
    if not meth_df.columns.equals(expr_df.columns):
        raise ValueError("Methylation and expression DataFrames have different columns")

    # Check cell lines consistency
    if len(cell_lines) != len(meth_df):
        raise ValueError(
            f"Number of cell lines ({len(cell_lines)}) doesn't match data rows ({len(meth_df)})"
        )

    # Check indices validity
    n_samples = len(cell_lines)
    all_idx = set(sensitive_idx) | set(resistant_idx)
    if max(all_idx, default=-1) >= n_samples:
        raise ValueError(f"Sample indices out of range [0, {n_samples})")

    # Check for duplicates
    if len(set(sensitive_idx) & set(resistant_idx)) > 0:
        raise ValueError("Overlap between sensitive_idx and resistant_idx")

    # Check for NaN issues
    meth_nan_cols = meth_df.columns[meth_df.isna().all()].tolist()
    expr_nan_cols = expr_df.columns[expr_df.isna().all()].tolist()

    if meth_nan_cols:
        raise ValueError(f"Methylation has all-NaN columns: {meth_nan_cols}")
    if expr_nan_cols:
        raise ValueError(f"Expression has all-NaN columns: {expr_nan_cols}")

    # Check methylation range (should be β-values in [0, 1] or close)
    meth_min = meth_df.min().min()
    meth_max = meth_df.max().max()

    if meth_min < -0.1 or meth_max > 1.1:
        warnings.warn(
            f"Methylation values outside expected [0, 1] range: [{meth_min:.3f}, {meth_max:.3f}]. "
            f"Consider normalizing."
        )

    # Validation passed
    return True


def prepare_model_input(
    data: Dict,
    genes: Optional[List[str]] = None,
    discretize: bool = True,
    n_bins: int = 3
) -> Dict:
    """
    Prepare data for input to GeneNetworkModel.

    Converts continuous data to discretized format and structures it for THRML.

    Args:
        data: Dictionary from generate_synthetic_data() or load_ccle_data()
        genes: Optional subset of genes to include (default: all)
        discretize: Whether to discretize values (default: True)
        n_bins: Number of bins for discretization (default: 3)

    Returns:
        Dictionary with:
            'methylation_discrete': Discretized methylation DataFrame
            'expression_discrete': Discretized expression DataFrame
            'methylation_continuous': Original continuous methylation
            'expression_continuous': Original continuous expression
            'cell_lines': List of cell line IDs
            'sensitive_idx': Sensitive cell line indices
            'resistant_idx': Resistant cell line indices

    Example:
        >>> data = generate_synthetic_data(['EGFR', 'KRAS'])
        >>> model_input = prepare_model_input(data)
        >>> model_input['methylation_discrete'].max().max() <= 2
        True
    """
    validate_data(data)

    # Filter genes if specified
    if genes is not None:
        available = set(data['methylation'].columns)
        genes = [g for g in genes if g in available]
        if not genes:
            raise ValueError("None of the specified genes found in data")
        meth_df = data['methylation'][genes]
        expr_df = data['expression'][genes]
    else:
        meth_df = data['methylation']
        expr_df = data['expression']

    # Discretize if requested
    if discretize:
        meth_discrete = discretize_values(meth_df, n_bins=n_bins)
        expr_discrete = discretize_values(expr_df, n_bins=n_bins)
    else:
        meth_discrete = meth_df
        expr_discrete = expr_df

    return {
        'methylation_discrete': meth_discrete,
        'expression_discrete': expr_discrete,
        'methylation_continuous': meth_df,
        'expression_continuous': expr_df,
        'cell_lines': data['cell_lines'],
        'sensitive_idx': data['sensitive_idx'],
        'resistant_idx': data['resistant_idx']
    }


# ========== Unit Tests ==========

if __name__ == '__main__':
    print("=" * 70)
    print("Data Loader Unit Tests")
    print("=" * 70)

    # Test 1: Discretization basic
    print("\n[Test 1] Basic discretization")
    df = pd.DataFrame({
        'EGFR': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        'KRAS': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    })
    disc = discretize_values(df)
    print(f"  Original shape: {df.shape}")
    print(f"  Discretized shape: {disc.shape}")
    print(f"  Unique values: {sorted(disc.values.flatten().unique())}")
    assert disc.shape == df.shape
    assert set(disc.values.flatten()) <= {0, 1, 2}
    print("  ✓ PASSED")

    # Test 2: Discretization with NaNs
    print("\n[Test 2] Discretization with NaNs")
    df_nan = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    disc_nan = discretize_values(df_nan)
    print(f"  NaN handling: {disc_nan['A'].tolist()}")
    assert disc_nan['A'].isna().sum() == 0  # NaNs should be filled
    assert disc_nan['B'].tolist() == [0, 0, 0, 0, 0]  # All-NaN column
    print("  ✓ PASSED")

    # Test 3: Discretization with constant values
    print("\n[Test 3] Discretization with constant values")
    df_const = pd.DataFrame({
        'C': [5, 5, 5, 5, 5]
    })
    disc_const = discretize_values(df_const)
    print(f"  Constant column result: {disc_const['C'].tolist()}")
    assert len(set(disc_const['C'])) == 1  # All same value
    print("  ✓ PASSED")

    # Test 4: Synthetic data generation
    print("\n[Test 4] Synthetic data generation")
    genes = ['EGFR', 'KRAS', 'BRAF']
    data = generate_synthetic_data(genes, n_sensitive=10, n_resistant=10, seed=42)
    print(f"  Genes: {genes}")
    print(f"  Methylation shape: {data['methylation'].shape}")
    print(f"  Expression shape: {data['expression'].shape}")
    print(f"  Cell lines: {len(data['cell_lines'])}")
    print(f"  Sensitive indices: {len(data['sensitive_idx'])}")
    print(f"  Resistant indices: {len(data['resistant_idx'])}")
    print(f"  Ground truth edges: {len(data['ground_truth']['edges'])}")
    assert data['methylation'].shape == (20, 3)
    assert data['expression'].shape == (20, 3)
    assert len(data['cell_lines']) == 20
    assert len(data['sensitive_idx']) == 10
    assert len(data['resistant_idx']) == 10
    print("  ✓ PASSED")

    # Test 5: Data validation
    print("\n[Test 5] Data validation")
    try:
        validate_data(data)
        print("  ✓ PASSED - validation successful")
    except ValueError as e:
        print(f"  ✗ FAILED - {e}")
        raise

    # Test 6: Methylation range check
    print("\n[Test 6] Methylation value ranges")
    meth_min = data['methylation'].min().min()
    meth_max = data['methylation'].max().max()
    print(f"  Methylation range: [{meth_min:.3f}, {meth_max:.3f}]")
    assert 0 <= meth_min <= 1
    assert 0 <= meth_max <= 1
    print("  ✓ PASSED")

    # Test 7: Prepare model input
    print("\n[Test 7] Prepare model input")
    model_input = prepare_model_input(data, discretize=True)
    print(f"  Discretized methylation shape: {model_input['methylation_discrete'].shape}")
    print(f"  Discretized expression shape: {model_input['expression_discrete'].shape}")
    print(f"  Methylation discrete values: {sorted(model_input['methylation_discrete'].values.flatten().unique())}")
    print(f"  Expression discrete values: {sorted(model_input['expression_discrete'].values.flatten().unique())}")
    assert model_input['methylation_discrete'].max().max() <= 2
    assert model_input['expression_discrete'].max().max() <= 2
    print("  ✓ PASSED")

    # Test 8: CCLE data loading (fallback to synthetic)
    print("\n[Test 8] CCLE data loading with synthetic fallback")
    ccle_data = load_ccle_data(['EGFR', 'KRAS'], data_dir='data/ccle', fallback_synthetic=True)
    print(f"  Loaded data shape: {ccle_data['methylation'].shape}")
    assert 'methylation' in ccle_data
    assert 'expression' in ccle_data
    print("  ✓ PASSED (using synthetic fallback)")

    # Test 9: Causal structure verification
    print("\n[Test 9] Ground truth causal structure")
    gt = data['ground_truth']
    print(f"  Total causal edges: {len(gt['edges'])}")
    print(f"  Methylation-expression correlations:")
    for gene, (sens_corr, resist_corr) in gt['methylation_expression'].items():
        print(f"    {gene}: sensitive={sens_corr:.2f}, resistant={resist_corr:.2f}")
    print(f"  Pathway cascades: {len(gt['pathway_cascade'])}")
    for source, target, strength in gt['pathway_cascade']:
        print(f"    {source} -> {target} (strength={strength:.2f})")
    assert len(gt['edges']) > 0
    print("  ✓ PASSED")

    # Test 10: Smoke test for integration
    print("\n[Test 10] Integration smoke test")
    test_genes = ['EGFR', 'KRAS', 'BRAF', 'PIK3CA']
    test_data = generate_synthetic_data(test_genes, n_sensitive=25, n_resistant=25, seed=123)
    test_input = prepare_model_input(test_data)

    # Verify discretization
    disc_meth = test_input['methylation_discrete']
    disc_expr = test_input['expression_discrete']

    print(f"  Test with {len(test_genes)} genes")
    print(f"  Discretized methylation unique values: {sorted(disc_meth.values.flatten().unique())}")
    print(f"  Discretized expression unique values: {sorted(disc_expr.values.flatten().unique())}")

    # Check sensitive vs resistant differences
    sens_mean_expr = disc_expr.iloc[test_input['sensitive_idx']].mean()
    resist_mean_expr = disc_expr.iloc[test_input['resistant_idx']].mean()
    print(f"  Mean expression (sensitive): {sens_mean_expr.mean():.3f}")
    print(f"  Mean expression (resistant): {resist_mean_expr.mean():.3f}")

    assert set(disc_meth.values.flatten()) <= {0, 1, 2}
    assert set(disc_expr.values.flatten()) <= {0, 1, 2}
    print("  ✓ PASSED")

    # Summary
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nModule ready for use in inference pipeline!")
    print("\nExample usage:")
    print("  from core.data_loader import generate_synthetic_data, prepare_model_input")
    print("  data = generate_synthetic_data(['EGFR', 'KRAS', 'BRAF'])")
    print("  model_input = prepare_model_input(data)")
    print("  # Use model_input['methylation_discrete'] in GeneNetworkModel")
