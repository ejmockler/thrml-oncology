# Inference Pipeline - Quick Start Guide

**5-minute guide to causal network inference**

---

## Installation Check

```bash
cd thrml-cancer-decision-support
python3 -c "import jax, thrml, pandas, numpy; print('✓ All dependencies installed')"
```

---

## Basic Workflow

### 1. Prepare Data

```python
import pandas as pd
from core.inference import infer_network_structure

# Load discretized data (values: 0, 1, 2)
genes = ["EGFR", "KRAS", "BRAF"]

methylation_df = pd.read_csv("data/methylation_discretized.csv")
expression_df = pd.read_csv("data/expression_discretized.csv")

# Optional: Get INDRA priors
from core.indra_client import IndraClient
client = IndraClient()
priors = client.build_prior_network(genes)
# Or use empty priors:
# priors = {}
```

### 2. Run Inference

```python
results = infer_network_structure(
    genes=genes,
    methylation_data=methylation_df,
    expression_data=expression_df,
    prior_network=priors,
    n_samples=2000,    # More samples = higher quality
    n_warmup=500,      # Burn-in
    parallel=True      # Use all CPU cores
)
```

### 3. Inspect Results

```python
# Print all edges
for (g1, g2), res in results.items():
    print(f"{res['direction']:20s} | ΔF={res['delta_F']:6.2f} | conf={res['confidence']:.3f}")

# Get summary statistics
from core.inference import summarize_network
summary = summarize_network(results)
print(f"Decided: {summary['decided_directions']}/{summary['total_pairs_tested']}")
```

---

## Network Comparison (Sensitive vs Resistant)

```python
from core.inference import compare_networks

# Infer both networks
net_sensitive = infer_network_structure(genes, meth_sens, expr_sens, priors)
net_resistant = infer_network_structure(genes, meth_res, expr_res, priors)

# Compare
changes = compare_networks(net_sensitive, net_resistant, threshold=2.0)

# Resistance mechanisms
print("Edge flips (direction reversed):", changes['edge_flips'])
print("New edges in resistant:", changes['new_edges'])
print("Strengthened edges:", changes['edge_strengthening'])
```

---

## Export for Visualization

### Cytoscape

```python
from core.inference import export_network_to_cytoscape

export_network_to_cytoscape(
    results,
    "outputs/network.csv",
    min_confidence=0.5,  # Filter weak edges
    min_delta_F=1.0
)
```

**Import to Cytoscape**:
1. Open Cytoscape
2. **File → Import → Network from File**
3. Select `network.csv`
4. Map: `source` → Source, `target` → Target

---

## Long-Running Jobs (Checkpointing)

```python
from core.inference import run_inference_with_progress

data = {
    'methylation': methylation_df,
    'expression': expression_df
}

results = run_inference_with_progress(
    genes=large_gene_list,
    data=data,
    prior_network=priors,
    output_file="results/network.json",
    checkpoint_every=50,  # Save every 50 pairs
    n_samples=2000,
    parallel=True
)
```

**If job crashes**: Just re-run the same command → auto-resumes from checkpoint

---

## Common Parameters

| Parameter | Default | Recommended | Effect |
|-----------|---------|-------------|--------|
| `n_samples` | 1000 | 2000-5000 | Higher = more accurate ΔF |
| `n_warmup` | 100 | 500-1000 | Higher = better convergence |
| `parallel` | True | True | Use multi-threading |
| `threshold` | 1.0 | 1.0-2.0 | Decision threshold for direction |

---

## Interpreting Results

### Direction Field

- `"EGFR -> KRAS"`: Forward direction (EGFR regulates KRAS)
- `"KRAS -> EGFR"`: Backward direction (KRAS regulates EGFR)
- `"undecided"`: No clear direction (|ΔF| < threshold)

### ΔF (Free Energy Difference)

- **ΔF > 0**: Forward model preferred (G1 → G2)
- **ΔF < 0**: Backward model preferred (G2 → G1)
- **|ΔF| > 2**: Strong directional signal
- **|ΔF| < 1**: Weak/unclear direction

### Confidence

- Range: [0, 1]
- Formula: `|ΔF| / (1 + |ΔF|)`
- **> 0.7**: High confidence
- **0.5-0.7**: Moderate confidence
- **< 0.5**: Low confidence

---

## Example Output

```
EGFR -> KRAS         | ΔF=  3.45 | conf=0.776 | prior=0.90
KRAS -> BRAF         | ΔF=  2.18 | conf=0.685 | prior=0.85
EGFR -> BRAF         | ΔF=  1.52 | conf=0.603 | prior=0.50
PIK3CA -> KRAS       | ΔF= -0.84 | conf=0.457 | prior=0.60 (undecided)
```

**Interpretation**:
- Strong EGFR → KRAS edge (ΔF=3.45, high confidence, supported by INDRA)
- Moderate KRAS → BRAF edge (ΔF=2.18, moderate confidence)
- Weak EGFR → BRAF (ΔF=1.52, borderline)
- Undecided PIK3CA ↔ KRAS (ΔF close to 0)

---

## Troubleshooting

### Out of Memory

```python
# Reduce samples
results = infer_network_structure(..., n_samples=500)

# Or use checkpointing
results = run_inference_with_progress(..., checkpoint_every=10)
```

### Slow Inference

```python
# Enable parallelization
results = infer_network_structure(..., parallel=True)

# Filter to high-prior edges only
filtered_priors = {k: v for k, v in priors.items() if v > 0.5}
results = infer_network_structure(..., prior_network=filtered_priors)
```

### Failed Gene Pairs

```python
# Check for None results
failed = [(g1, g2) for (g1, g2), res in results.items() if res is None]
print(f"Failed pairs: {failed}")

# Check logs for errors
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Testing

```bash
# Run test suite
python scripts/test_inference.py

# Expected: 4 tests, all passing
# - Test 1: Basic inference
# - Test 2: Network comparison
# - Test 3: Checkpointing
# - Test 4: Parallel execution
```

---

## Next Steps

1. **Full API**: See `docs/INFERENCE_API.md`
2. **Model Details**: See `core/thrml_model.py`
3. **INDRA Priors**: See `core/indra_client.py`
4. **Integration**: See `INFERENCE_MODULE_SUMMARY.md`

---

**Ready to run? Start with `scripts/test_inference.py` to verify installation!**
