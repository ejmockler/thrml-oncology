# Inference Pipeline API Documentation

Complete reference for the causal network inference orchestration pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Functions](#core-functions)
3. [Network Comparison](#network-comparison)
4. [Batch Processing](#batch-processing)
5. [Progress Tracking](#progress-tracking)
6. [Export & Utilities](#export--utilities)
7. [Usage Examples](#usage-examples)
8. [Performance Optimization](#performance-optimization)

---

## Overview

The `core/inference.py` module provides high-level orchestration for causal network inference using THRML energy-based models. It coordinates:

- **Batch inference** across all gene pairs
- **Parallel execution** using ThreadPoolExecutor (JAX manages GPU)
- **Network comparison** between biological conditions (e.g., sensitive vs resistant)
- **Checkpointing** for fault-tolerant long-running jobs
- **Export** to standard formats (JSON, CSV for Cytoscape)

### Architecture

```
┌─────────────────────────────────────────────────┐
│         User Application / Pipeline             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│       core/inference.py (Orchestration)         │
│  ┌──────────────────────────────────────────┐   │
│  │  infer_network_structure()               │   │
│  │  compare_networks()                      │   │
│  │  run_inference_with_progress()           │   │
│  └──────────────┬───────────────────────────┘   │
└─────────────────┼───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│        core/thrml_model.py (Energy Model)       │
│  ┌──────────────────────────────────────────┐   │
│  │  GeneNetworkModel.test_causal_direction()│   │
│  │  - build_model_forward()                 │   │
│  │  - build_model_backward()                │   │
│  │  - sample_from_model()                   │   │
│  │  - compute_free_energy()                 │   │
│  └──────────────┬───────────────────────────┘   │
└─────────────────┼───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│              THRML Library (JAX)                │
│    Block Gibbs Sampling, CategoricalEBM         │
└─────────────────────────────────────────────────┘
```

---

## Core Functions

### `infer_network_structure()`

**Primary function for network-wide causal inference.**

```python
def infer_network_structure(
    genes: List[str],
    methylation_data: pd.DataFrame,
    expression_data: pd.DataFrame,
    prior_network: Dict[Tuple[str, str], float],
    n_samples: int = 1000,
    n_warmup: int = 100,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    threshold: float = 1.0
) -> Dict[Tuple[str, str], Dict]:
```

#### Parameters

- **`genes`**: List of gene symbols (e.g., `["EGFR", "KRAS", "BRAF"]`)
- **`methylation_data`**: DataFrame with shape `[n_samples, n_genes]`, discretized values `{0, 1, 2}`
- **`expression_data`**: DataFrame with shape `[n_samples, n_genes]`, discretized values `{0, 1, 2}`
- **`prior_network`**: INDRA priors mapping `(gene1, gene2) -> belief_score [0, 1]`
- **`n_samples`**: Number of MCMC samples for free energy estimation
- **`n_warmup`**: Burn-in iterations to discard
- **`parallel`**: Enable parallel execution (uses `ThreadPoolExecutor`)
- **`max_workers`**: Number of parallel workers (`None` = use all cores)
- **`threshold`**: Free energy threshold for direction decision (default: 1.0)

#### Returns

Dictionary mapping `(gene1, gene2)` to result dict:

```python
{
    (gene1, gene2): {
        'direction': str,        # 'gene1 -> gene2' | 'gene2 -> gene1' | 'undecided'
        'delta_F': float,        # F_backward - F_forward (positive favors forward)
        'F_forward': float,      # Free energy of forward model
        'F_backward': float,     # Free energy of backward model
        'confidence': float,     # |ΔF| / (1 + |ΔF|) ∈ [0, 1]
        'prior_belief': float,   # INDRA belief score
        'n_samples': int         # Number of samples used
    }
}
```

#### Example

```python
from core.inference import infer_network_structure
import pandas as pd

genes = ["EGFR", "KRAS", "BRAF"]

# Discretized data (0=low, 1=med, 2=high)
methylation_df = pd.read_csv("data/methylation_discretized.csv")
expression_df = pd.read_csv("data/expression_discretized.csv")

# INDRA priors
priors = {
    ("EGFR", "KRAS"): 0.9,
    ("KRAS", "BRAF"): 0.85
}

# Run inference
results = infer_network_structure(
    genes=genes,
    methylation_data=methylation_df,
    expression_data=expression_df,
    prior_network=priors,
    n_samples=2000,
    n_warmup=500,
    parallel=True
)

# Inspect results
for (g1, g2), res in results.items():
    print(f"{res['direction']}: ΔF={res['delta_F']:.2f}, conf={res['confidence']:.3f}")
```

---

## Network Comparison

### `compare_networks()`

**Identify structural changes between two network inference results (e.g., sensitive vs resistant cells).**

```python
def compare_networks(
    network_sensitive: Dict[Tuple[str, str], Dict],
    network_resistant: Dict[Tuple[str, str], Dict],
    threshold: float = 2.0,
    min_delta_F: float = 1.0
) -> Dict[str, List[Tuple[str, str]]]:
```

#### Parameters

- **`network_sensitive`**: Inference results for condition 1 (e.g., drug-sensitive cells)
- **`network_resistant`**: Inference results for condition 2 (e.g., drug-resistant cells)
- **`threshold`**: Minimum change in `|ΔF|` to flag edge strengthening/weakening
- **`min_delta_F`**: Minimum `|ΔF|` to consider an edge significant

#### Returns

```python
{
    'edge_flips': [(g1, g2), ...],          # Direction reversed between conditions
    'edge_weakening': [(g1, g2), ...],      # |ΔF| decreased > threshold
    'edge_strengthening': [(g1, g2), ...],  # |ΔF| increased > threshold
    'new_edges': [(g1, g2), ...],           # Present in resistant only
    'lost_edges': [(g1, g2), ...],          # Present in sensitive only
    'stable_edges': [(g1, g2), ...]         # No significant change
}
```

#### Example

```python
from core.inference import infer_network_structure, compare_networks

# Infer networks for both conditions
network_sensitive = infer_network_structure(
    genes=genes,
    methylation_data=meth_sensitive,
    expression_data=expr_sensitive,
    prior_network=priors,
    n_samples=2000
)

network_resistant = infer_network_structure(
    genes=genes,
    methylation_data=meth_resistant,
    expression_data=expr_resistant,
    prior_network=priors,
    n_samples=2000
)

# Compare
changes = compare_networks(
    network_sensitive,
    network_resistant,
    threshold=2.0,
    min_delta_F=1.0
)

print(f"Edge flips: {changes['edge_flips']}")
print(f"New edges in resistant: {changes['new_edges']}")
```

#### Interpretation

- **Edge flips**: Suggest fundamental rewiring (e.g., feedback loops reversed)
- **Edge strengthening**: Indicates pathway upregulation (resistance mechanism?)
- **Edge weakening**: Indicates pathway downregulation
- **New edges**: Gained interactions (novel dependencies in resistant state)
- **Lost edges**: Broken dependencies (desensitization)

---

## Batch Processing

### `process_gene_pairs_batch()`

**Process a batch of gene pairs (useful for custom parallelization or distributed computing).**

```python
def process_gene_pairs_batch(
    gene_pairs: List[Tuple[str, str]],
    model: GeneNetworkModel,
    data: Dict[str, jnp.ndarray],
    n_samples: int = 1000,
    n_warmup: int = 100
) -> List[Dict]:
```

#### Parameters

- **`gene_pairs`**: List of `(gene1, gene2)` tuples to test
- **`model`**: Initialized `GeneNetworkModel` instance
- **`data`**: Data dict with `{gene}_meth` and `{gene}_expr` keys
- **`n_samples`**, **`n_warmup`**: Sampling parameters

#### Returns

List of result dicts (one per pair), same structure as `infer_network_structure()`

#### Use Case: Custom Parallelization

```python
from core.inference import process_gene_pairs_batch
from core.thrml_model import GeneNetworkModel
from concurrent.futures import ProcessPoolExecutor

# Initialize model
model = GeneNetworkModel(genes, priors)

# Prepare data
data = {
    f'{g}_meth': jnp.array(meth_df[g].values) for g in genes
} | {
    f'{g}_expr': jnp.array(expr_df[g].values) for g in genes
}

# Divide gene pairs into batches
all_pairs = [(g1, g2) for g1 in genes for g2 in genes if g1 != g2]
batch_size = 10
batches = [all_pairs[i:i+batch_size] for i in range(0, len(all_pairs), batch_size)]

# Process batches across multiple GPUs
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(process_gene_pairs_batch, batch, model, data, 2000)
        for batch in batches
    ]

    all_results = []
    for future in futures:
        all_results.extend(future.result())
```

---

## Progress Tracking

### `run_inference_with_progress()`

**Run inference with automatic checkpointing and resume capability.**

```python
def run_inference_with_progress(
    genes: List[str],
    data: Dict[str, Any],
    prior_network: Dict[Tuple[str, str], float],
    output_file: str,
    checkpoint_every: int = 10,
    n_samples: int = 1000,
    n_warmup: int = 100,
    parallel: bool = True
) -> Dict[Tuple[str, str], Dict]:
```

#### Parameters

- **`genes`**: List of gene symbols
- **`data`**: Dict with `{'methylation': DataFrame, 'expression': DataFrame}`
- **`prior_network`**: INDRA priors
- **`output_file`**: Path to save final JSON results
- **`checkpoint_every`**: Save checkpoint every N gene pairs
- **`n_samples`**, **`n_warmup`**: Sampling parameters
- **`parallel`**: Enable parallel execution

#### Behavior

1. Creates checkpoint file: `{output_file}_checkpoint.pkl`
2. Saves intermediate results every `checkpoint_every` pairs
3. Automatically resumes from checkpoint if file exists
4. Removes checkpoint file upon successful completion
5. Saves final results as JSON to `output_file`

#### Example

```python
from core.inference import run_inference_with_progress

data = {
    'methylation': meth_df,
    'expression': expr_df
}

results = run_inference_with_progress(
    genes=large_gene_list,
    data=data,
    prior_network=priors,
    output_file="results/network_inference.json",
    checkpoint_every=50,  # Checkpoint every 50 pairs
    n_samples=2000,
    parallel=True
)
```

#### Recovery from Failure

If the job crashes:

```bash
# Checkpoint file exists: results/network_inference_checkpoint.pkl
# Simply re-run the same command:
python my_pipeline.py
# It will automatically resume from the checkpoint
```

---

## Export & Utilities

### `export_network_to_cytoscape()`

**Export network to Cytoscape-compatible CSV format.**

```python
def export_network_to_cytoscape(
    network_results: Dict[Tuple[str, str], Dict],
    output_file: str,
    min_confidence: float = 0.5,
    min_delta_F: float = 1.0
):
```

#### Parameters

- **`network_results`**: Output from `infer_network_structure()`
- **`output_file`**: Path to save CSV
- **`min_confidence`**: Only export edges with `confidence >= threshold`
- **`min_delta_F`**: Only export edges with `|ΔF| >= threshold`

#### Output CSV Format

```csv
source,target,delta_F,confidence,F_forward,F_backward,prior_belief
EGFR,KRAS,3.45,0.776,12.3,15.75,0.9
KRAS,BRAF,2.18,0.685,10.1,12.28,0.85
```

#### Cytoscape Import

1. Open Cytoscape
2. **File → Import → Network from File**
3. Select CSV file
4. Map columns: `source` → Source, `target` → Target
5. Import edge attributes: `delta_F`, `confidence`, `prior_belief`

#### Example

```python
from core.inference import export_network_to_cytoscape

export_network_to_cytoscape(
    results,
    "outputs/network.csv",
    min_confidence=0.7,  # High-confidence edges only
    min_delta_F=2.0      # Strong directional signal
)
```

---

### `summarize_network()`

**Generate summary statistics for network inference results.**

```python
def summarize_network(
    network_results: Dict[Tuple[str, str], Dict]
) -> Dict[str, Any]:
```

#### Returns

```python
{
    'total_pairs_tested': int,
    'decided_directions': int,
    'undecided_directions': int,
    'strong_edges': int,             # |ΔF| > 2.0
    'edges_with_prior_support': int,
    'mean_delta_F': float,
    'median_delta_F': float,
    'std_delta_F': float,
    'mean_confidence': float,
    'median_confidence': float
}
```

#### Example

```python
from core.inference import summarize_network

summary = summarize_network(results)
print(f"Total pairs: {summary['total_pairs_tested']}")
print(f"Decided: {summary['decided_directions']}")
print(f"Mean ΔF: {summary['mean_delta_F']:.2f}")
```

---

## Usage Examples

### Example 1: Basic Workflow

```python
import pandas as pd
from core.inference import infer_network_structure, summarize_network
from core.indra_client import IndraClient

# Step 1: Load discretized data
genes = ["EGFR", "KRAS", "BRAF", "PIK3CA", "PTEN"]

meth_df = pd.read_csv("data/methylation_discretized.csv")
expr_df = pd.read_csv("data/expression_discretized.csv")

# Step 2: Get INDRA priors
client = IndraClient()
priors = client.build_prior_network(genes)

# Step 3: Run inference
results = infer_network_structure(
    genes=genes,
    methylation_data=meth_df,
    expression_data=expr_df,
    prior_network=priors,
    n_samples=2000,
    n_warmup=500,
    parallel=True
)

# Step 4: Summarize
summary = summarize_network(results)
print(summary)

# Step 5: Export
export_network_to_cytoscape(results, "outputs/network.csv")
```

---

### Example 2: Sensitive vs Resistant Comparison

```python
from core.inference import infer_network_structure, compare_networks

# Load data for both conditions
meth_sens = pd.read_csv("data/sensitive_methylation.csv")
expr_sens = pd.read_csv("data/sensitive_expression.csv")

meth_res = pd.read_csv("data/resistant_methylation.csv")
expr_res = pd.read_csv("data/resistant_expression.csv")

# Infer networks
net_sens = infer_network_structure(
    genes, meth_sens, expr_sens, priors, n_samples=2000
)

net_res = infer_network_structure(
    genes, meth_res, expr_res, priors, n_samples=2000
)

# Compare
changes = compare_networks(net_sens, net_res, threshold=2.0)

# Analyze resistance mechanisms
print("Resistance-associated changes:")
for flip in changes['edge_flips']:
    print(f"  Edge flip: {flip}")
for gain in changes['new_edges']:
    print(f"  Gained edge: {gain}")
```

---

### Example 3: Large-Scale Inference with Checkpointing

```python
from core.inference import run_inference_with_progress

# 100-gene network (4950 pairs)
genes = load_cancer_hallmark_genes()  # e.g., 100 genes

data = {
    'methylation': pd.read_csv("data/tcga_methylation.csv"),
    'expression': pd.read_csv("data/tcga_expression.csv")
}

# Run with checkpointing
results = run_inference_with_progress(
    genes=genes,
    data=data,
    prior_network=priors,
    output_file="results/large_network.json",
    checkpoint_every=100,  # Checkpoint every 100 pairs
    n_samples=5000,        # High-quality samples
    n_warmup=1000,
    parallel=True
)
```

---

## Performance Optimization

### Memory Management

JAX uses XLA compilation and GPU memory pooling. For large networks:

```python
# Set JAX memory preallocation
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'

# Run inference
results = infer_network_structure(...)
```

### Parallel Execution Tips

- **ThreadPoolExecutor** (default): JAX manages GPU internally
- **ProcessPoolExecutor**: Use only if you have multiple GPUs and manually assign devices

```python
# Multi-GPU setup (advanced)
import jax

# Manually assign workers to GPUs
jax.config.update('jax_platform_name', 'gpu')
devices = jax.devices('gpu')

# Run batches on different devices
# (requires custom batch processing logic)
```

### Sampling Quality vs Speed

| Use Case | `n_samples` | `n_warmup` | Runtime (2 genes) |
|----------|-------------|------------|-------------------|
| Quick test | 500 | 100 | ~10 sec |
| Standard | 2000 | 500 | ~30 sec |
| High quality | 5000 | 1000 | ~60 sec |
| Publication | 10000 | 2000 | ~120 sec |

---

## Error Handling

The pipeline handles errors gracefully:

```python
# Failed gene pairs return None or error dict
results = infer_network_structure(...)

for (g1, g2), result in results.items():
    if result is None:
        print(f"Failed: {g1} <-> {g2}")
    elif result.get('direction') == 'error':
        print(f"Error: {g1} <-> {g2}: {result['error']}")
```

Common issues:

1. **Out of memory**: Reduce `n_samples` or use checkpointing
2. **Sampling divergence**: Increase `n_warmup` or check data quality
3. **Slow inference**: Enable `parallel=True` or reduce gene set

---

## References

- **THRML Documentation**: See `THRML_COMPREHENSIVE_DOCUMENTATION.md`
- **Model Details**: See `core/thrml_model.py`
- **INDRA Priors**: See `core/indra_client.py`
- **Test Examples**: See `scripts/test_inference.py`

---

**Next Steps**: Integrate into main pipeline (`main.py`) for end-to-end workflow.
