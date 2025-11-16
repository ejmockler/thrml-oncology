# Data Loader Quick Reference

## Import
```python
from core.data_loader import (
    discretize_values,
    generate_synthetic_data,
    load_ccle_data,
    validate_data,
    prepare_model_input
)
```

## Common Workflows

### 1. Generate Synthetic Data for Testing
```python
# Basic usage
genes = ['EGFR', 'KRAS', 'BRAF']
data = generate_synthetic_data(genes, n_sensitive=25, n_resistant=25)

# Returns:
# - data['methylation']: pd.DataFrame [50 × 3]
# - data['expression']: pd.DataFrame [50 × 3]
# - data['ground_truth']: Dict with true causal edges
```

### 2. Discretize Data for THRML
```python
# Prepare for categorical energy models
model_input = prepare_model_input(data, discretize=True, n_bins=3)

# Use discretized data
meth = model_input['methylation_discrete']  # Values in {0, 1, 2}
expr = model_input['expression_discrete']   # Values in {0, 1, 2}
```

### 3. Load Real CCLE Data
```python
# Loads CCLE files, falls back to synthetic if missing
data = load_ccle_data(['EGFR', 'KRAS'], data_dir='data/ccle')
```

### 4. Validate Data Quality
```python
# Check data structure and quality
validate_data(data)  # Raises ValueError if issues found
```

## Function Signatures

### discretize_values
```python
discretize_values(data: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame
```
**Input**: Continuous values
**Output**: Discrete values in {0, 1, 2}
**Method**: Quantile-based binning

### generate_synthetic_data
```python
generate_synthetic_data(
    genes: List[str],
    n_sensitive: int = 25,
    n_resistant: int = 25,
    seed: int = 42
) -> Dict
```
**Returns**: Dict with methylation, expression, ground_truth

### load_ccle_data
```python
load_ccle_data(
    genes: List[str],
    data_dir: str = 'data/ccle',
    fallback_synthetic: bool = True
) -> Dict
```
**Returns**: Same structure as generate_synthetic_data()

### validate_data
```python
validate_data(data: Dict) -> bool
```
**Returns**: True if valid, raises ValueError otherwise

### prepare_model_input
```python
prepare_model_input(
    data: Dict,
    genes: Optional[List[str]] = None,
    discretize: bool = True,
    n_bins: int = 3
) -> Dict
```
**Returns**: Dict with discrete + continuous data

## Integration with THRML

```python
# Complete workflow
from core.data_loader import generate_synthetic_data, prepare_model_input
from core.thrml_model import GeneNetworkModel
from core.indra_client import INDRAClient

# 1. Data
genes = ['EGFR', 'KRAS', 'BRAF']
data = generate_synthetic_data(genes)
model_input = prepare_model_input(data)

# 2. INDRA priors
indra = INDRAClient()
priors = indra.get_pathway_network(genes)

# 3. THRML model
model = GeneNetworkModel(genes, priors, n_states=3)

# 4. Causal inference
result = model.test_causal_direction(
    'EGFR', 'KRAS',
    data=model_input,
    n_samples=1000
)

# 5. Validation
print(f"Predicted: {result['direction']}")
print(f"ΔF = {result['delta_F']:.3f}")
print(f"Ground truth: {data['ground_truth']['pathway_cascade']}")
```

## Ground Truth Access

```python
data = generate_synthetic_data(['EGFR', 'KRAS', 'BRAF'])
gt = data['ground_truth']

# Causal edges
edges = gt['edges']
# [('EGFR_meth', 'EGFR_expr', 'repression', 0.8), ...]

# Methylation-expression correlations
corr = gt['methylation_expression']['EGFR']
# (-0.8, -0.5) = (sensitive, resistant)

# Pathway cascades
cascade = gt['pathway_cascade']
# [('EGFR', 'KRAS', 0.6), ...]
```

## Testing

```bash
# Run unit tests
python3 core/data_loader.py

# Run test suite
python3 scripts/test_data_loader.py

# Run demo
python3 scripts/demo_data_loader.py
```

## Edge Cases

| Situation | Behavior |
|-----------|----------|
| NaN values | Filled with 0 (lowest bin) |
| Constant values | Assigned to bin 1 (medium) |
| Few unique values | Rank-based binning |
| All-NaN column | Filled with 0, warning issued |
| Missing CCLE files | Falls back to synthetic (if enabled) |

## Performance

| Operation | Time | Memory |
|-----------|------|--------|
| discretize_values (50×10) | ~2ms | <1 MB |
| generate_synthetic (50×10) | ~5ms | ~1 MB |
| validate_data | ~1ms | <1 MB |
| prepare_model_input | ~3ms | ~2 MB |

## Common Patterns

### Pattern 1: Quick Test
```python
# Minimal code for testing
data = generate_synthetic_data(['EGFR', 'KRAS'])
model_input = prepare_model_input(data)
# Ready to use
```

### Pattern 2: Validation Loop
```python
# Test multiple gene pairs
results = []
for gene1, gene2 in pairs:
    result = model.test_causal_direction(gene1, gene2, data=model_input)
    gt_edge = find_ground_truth(data['ground_truth'], gene1, gene2)
    results.append({
        'pair': (gene1, gene2),
        'predicted': result['direction'],
        'ground_truth': gt_edge,
        'correct': check_match(result, gt_edge)
    })
```

### Pattern 3: Batch Processing
```python
# Process large gene sets in batches
all_results = []
for gene_batch in np.array_split(all_genes, 10):
    data = generate_synthetic_data(gene_batch)
    model_input = prepare_model_input(data)
    # Process batch
    all_results.extend(process_batch(model_input))
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| ModuleNotFoundError: pandas | `pip install -r requirements.txt` |
| CCLE files not found | Use `fallback_synthetic=True` (default) |
| Discretization warning | Expected for constant/few unique values |
| Memory error | Process genes in smaller batches |

## Files

- **Core**: `/core/data_loader.py`
- **Docs**: `/core/DATA_LOADER_README.md`
- **Tests**: `/scripts/test_data_loader.py`
- **Demo**: `/scripts/demo_data_loader.py`
- **Delivery**: `/DATA_LOADER_DELIVERY.md`

## Status: ✅ Production Ready
