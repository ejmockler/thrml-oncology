# Data Loader Module

## Overview

The `data_loader.py` module provides a complete data loading pipeline for the THRML cancer decision support system. It handles discretization of continuous gene expression and methylation data, generates synthetic test data with known causal structure, and can load real CCLE data.

## Features

- **Quantile-based discretization**: Converts continuous values to {0, 1, 2} states (low/medium/high)
- **Synthetic data generation**: Creates biologically-inspired test data with ground truth causal relationships
- **CCLE data loading**: Loads real Cancer Cell Line Encyclopedia data (with synthetic fallback)
- **Data validation**: Comprehensive checks for data quality and structure
- **Edge case handling**: Robust handling of NaNs, constant values, and sparse data

## Quick Start

```python
from core.data_loader import generate_synthetic_data, prepare_model_input

# Generate synthetic data
genes = ['EGFR', 'KRAS', 'BRAF']
data = generate_synthetic_data(genes, n_sensitive=25, n_resistant=25)

# Prepare for THRML model
model_input = prepare_model_input(data, discretize=True)

# Use discretized data
methylation = model_input['methylation_discrete']  # Shape: [50, 3], values in {0, 1, 2}
expression = model_input['expression_discrete']    # Shape: [50, 3], values in {0, 1, 2}
```

## API Reference

### Core Functions

#### `discretize_values(data, n_bins=3)`

Discretize continuous values using quantile-based binning.

**Parameters:**
- `data` (pd.DataFrame): Continuous values [samples × genes]
- `n_bins` (int): Number of bins (default: 3 for low/med/high)

**Returns:**
- pd.DataFrame with same shape, values in {0, 1, 2}

**Example:**
```python
import pandas as pd
df = pd.DataFrame({'EGFR': [1.2, 3.4, 5.6, 7.8, 9.0]})
discretized = discretize_values(df)
# Result: values in {0, 1, 2} representing low/medium/high
```

**Edge Cases:**
- **NaNs**: Filled with 0 (lowest bin)
- **Constant values**: Assigned to bin 1 (medium)
- **Few unique values**: Falls back to rank-based binning

---

#### `generate_synthetic_data(genes, n_sensitive=25, n_resistant=25, seed=42)`

Generate synthetic methylation/expression data with known causal structure.

**Parameters:**
- `genes` (List[str]): Gene symbols to simulate
- `n_sensitive` (int): Number of therapy-sensitive cell lines
- `n_resistant` (int): Number of therapy-resistant cell lines
- `seed` (int): Random seed for reproducibility

**Returns:**
Dictionary with:
- `methylation` (pd.DataFrame): Continuous β-values in [0, 1]
- `expression` (pd.DataFrame): Continuous log2 expression
- `cell_lines` (List[str]): Cell line IDs
- `sensitive_idx` (List[int]): Indices of sensitive cells
- `resistant_idx` (List[int]): Indices of resistant cells
- `ground_truth` (Dict): True causal edges for validation

**Example:**
```python
data = generate_synthetic_data(['EGFR', 'KRAS', 'BRAF'], n_sensitive=10, n_resistant=10)
print(data['methylation'].shape)  # (20, 3)
print(data['ground_truth']['edges'])  # List of causal edges
```

**Biological Model:**
- **Sensitive cells**: High EGFR methylation → Low EGFR expression
- **Resistant cells**: Bypass pathway activation (KRAS/BRAF)
- **Causal structure**: Methylation → Expression (within gene), Expression → Expression (between genes)

---

#### `load_ccle_data(genes, data_dir='data/ccle', fallback_synthetic=True)`

Load CCLE (Cancer Cell Line Encyclopedia) methylation and expression data.

**Parameters:**
- `genes` (List[str]): Genes to load
- `data_dir` (str): Directory containing CCLE files
- `fallback_synthetic` (bool): Generate synthetic data if files missing

**Expected Files:**
- `{data_dir}/CCLE_methylation.csv`: Rows=cell lines, columns=genes
- `{data_dir}/CCLE_expression.csv`: Rows=cell lines, columns=genes
- `{data_dir}/CCLE_annotations.csv`: Optional metadata

**Returns:**
Same structure as `generate_synthetic_data()` (but `ground_truth` is None for real data)

**Example:**
```python
# Will use synthetic fallback if CCLE files not found
data = load_ccle_data(['EGFR', 'KRAS'], data_dir='data/ccle')
```

---

#### `validate_data(data)`

Validate data dictionary structure and quality.

**Checks:**
- Required keys present
- DataFrame shapes match
- Indices consistent
- No all-NaN columns
- Methylation values in [0, 1]

**Returns:**
- `True` if validation passes

**Raises:**
- `ValueError` with detailed error message if validation fails

---

#### `prepare_model_input(data, genes=None, discretize=True, n_bins=3)`

Prepare data for input to GeneNetworkModel.

**Parameters:**
- `data` (Dict): From `generate_synthetic_data()` or `load_ccle_data()`
- `genes` (List[str], optional): Subset of genes to include
- `discretize` (bool): Whether to discretize (default: True)
- `n_bins` (int): Bins for discretization

**Returns:**
Dictionary with:
- `methylation_discrete`: Discretized methylation
- `expression_discrete`: Discretized expression
- `methylation_continuous`: Original continuous values
- `expression_continuous`: Original continuous values
- `cell_lines`, `sensitive_idx`, `resistant_idx`: Preserved from input

**Example:**
```python
data = generate_synthetic_data(['EGFR', 'KRAS'])
model_input = prepare_model_input(data, discretize=True)

# Use in THRML model
meth = model_input['methylation_discrete']  # Values in {0, 1, 2}
expr = model_input['expression_discrete']   # Values in {0, 1, 2}
```

## Ground Truth Structure

The `ground_truth` dictionary from `generate_synthetic_data()` contains:

```python
{
    'edges': [
        ('EGFR_meth', 'EGFR_expr', 'repression', 0.8),
        ('KRAS_meth', 'KRAS_expr', 'repression', 0.4),
        ...
    ],
    'methylation_expression': {
        'EGFR': (-0.8, -0.5),  # (sensitive_corr, resistant_corr)
        'KRAS': (-0.3, -0.4),
        ...
    },
    'pathway_cascade': [
        ('EGFR', 'KRAS', 0.6),  # (source, target, strength)
        ('KRAS', 'BRAF', 0.5),
        ...
    ]
}
```

**Use Cases:**
1. Validate causal inference algorithms
2. Compare predicted vs. true causal edges
3. Benchmark model performance
4. Debug inference pipelines

## Integration with THRML Model

```python
from core.data_loader import generate_synthetic_data, prepare_model_input
from core.thrml_model import GeneNetworkModel
from core.indra_client import INDRAClient

# 1. Load data
genes = ['EGFR', 'KRAS', 'BRAF']
data = generate_synthetic_data(genes, n_sensitive=25, n_resistant=25)
model_input = prepare_model_input(data)

# 2. Get INDRA priors
indra = INDRAClient()
prior_network = indra.get_pathway_network(genes)

# 3. Create THRML model
model = GeneNetworkModel(genes, prior_network, n_states=3)

# 4. Test causal direction
result = model.test_causal_direction(
    'EGFR', 'KRAS',
    data=model_input,
    n_samples=1000
)

# 5. Compare to ground truth
print(f"Inferred: {result['direction']}")
print(f"Ground truth: {data['ground_truth']['pathway_cascade']}")
```

## Testing

Run unit tests:
```bash
python3 core/data_loader.py
```

Run comprehensive test suite:
```bash
python3 scripts/test_data_loader.py
```

Run demonstration:
```bash
python3 scripts/demo_data_loader.py
```

## Performance

- **Discretization**: O(n log n) per gene (due to sorting for quantiles)
- **Synthetic generation**: O(n × g) where n=samples, g=genes
- **Memory**: ~100 MB for 1000 samples × 100 genes

## Edge Cases and Warnings

### NaN Handling
```python
df = pd.DataFrame({'gene': [1.0, 2.0, np.nan, 4.0]})
disc = discretize_values(df)
# NaNs are filled with 0 (lowest bin)
```

### Constant Values
```python
df = pd.DataFrame({'gene': [5.0, 5.0, 5.0]})
disc = discretize_values(df)
# All values assigned to bin 1 (medium)
```

### Few Unique Values
```python
df = pd.DataFrame({'gene': [1.0, 1.0, 2.0, 2.0, 3.0]})
disc = discretize_values(df)
# Falls back to rank-based binning
```

## Citation

If using this module in research:

```bibtex
@software{thrml_cancer_data_loader,
  title={Data Loading Pipeline for THRML Cancer Decision Support},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/thrml-cancer-decision-support}
}
```

## License

See main project LICENSE file.

## Contact

For questions or issues, please open a GitHub issue or contact [maintainer email].
