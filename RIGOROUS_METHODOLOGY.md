# Rigorous Methodology: Data to Demo Pipeline

**XTR-0 Hackathon: Thermodynamic Causal Inference for Drug Response Prediction**

---

## 📚 Navigation
- **[← Back to Index](DOCUMENTATION_INDEX.md)** - All documentation
- **[QUICK_START.md](QUICK_START.md)** - Immediate action items
- **[DATA_AND_METHODOLOGY_SUMMARY.md](DATA_AND_METHODOLOGY_SUMMARY.md)** - Executive summary

**Implementation guides by section**:
- § 1 → `core/data_loader.py`
- § 2 → `core/thrml_model.py` + `core/indra_client.py`
- § 3 → `core/inference.py`
- § 4 → `core/validation.py`
- § 5 → `scripts/02_run_inference.py`

---

## Table of Contents
1. [Data Preprocessing Pipeline](#1-data-preprocessing-pipeline)
2. [THRML Model Construction](#2-thrml-model-construction)
3. [Causal Inference Procedure](#3-causal-inference-procedure)
4. [Validation Framework](#4-validation-framework)
5. [Demo Execution Workflow](#5-demo-execution-workflow)

---

## 1. Data Preprocessing Pipeline

### 1.1 Input Data Specifications

#### CCLE Expression Data
**File**: `data/raw/ccle/CCLE_expression_TPM.csv` (~200 MB)
- **Format**: CSV, rows = cell lines (n≈1300), columns = genes (n≈19000)
- **Values**: log₂(TPM + 1) where TPM = Transcripts Per Million
- **Range**: [0, ~15] (log-transformed, already normalized by DepMap)

#### CCLE Methylation Data
**File**: `data/raw/ccle/CCLE_RRBS_methylation.txt` (~variable size)
- **Format**: Tab-separated, rows = cell lines (n≈800), columns = gene TSS regions
- **Values**: β-values ∈ [0, 1] (proportion methylated CpGs)
- **Method**: RRBS (Reduced Representation Bisulfite Sequencing)
- **Regions**: TSS ±1kb (transcription start site flanking regions)

#### GDSC Drug Response Data
**File**: `data/raw/gdsc/GDSC_drug_data_fitted.xlsx` (3.3 MB)
- **Format**: Excel with columns:
  - `CELL_LINE_NAME`: Cell line identifier
  - `DRUG_NAME`: Compound name (e.g., "Erlotinib")
  - `LN_IC50`: Natural log of IC₅₀ (μM)
  - `AUC`: Area under dose-response curve
  - `RMSE`: Fit quality metric
  - `Z_SCORE`: Normalized sensitivity

#### CCLE Metadata
**File**: `data/raw/ccle/Model.csv`
- **Purpose**: Map DepMap IDs ↔ cell line names, tissue annotations
- **Key Columns**: `ModelID`, `CellLineName`, `OncotreeLineage`, `OncotreePrimaryDisease`

---

### 1.2 Gene Selection Strategy

**Biological Rationale**: Focus on EGFR pathway for erlotinib resistance

#### Target Gene Set (15-20 genes)
```python
EGFR_PATHWAY_GENES = [
    # Core signaling
    'EGFR',      # Primary target
    'KRAS',      # Downstream effector, common bypass
    'BRAF',      # RAF/MEK/ERK pathway
    'PIK3CA',    # PI3K/AKT pathway
    'PTEN',      # PI3K negative regulator
    'AKT1',      # Survival signaling
    'MTOR',      # Growth regulation
    'MEK1',      # MAPK pathway (MAP2K1)
    'ERK2',      # MAPK endpoint (MAPK1)

    # Feedback and regulation
    'MET',       # Common bypass mechanism
    'HER2',      # ERBB2, receptor crosstalk
    'HER3',      # ERBB3, receptor crosstalk
    'SOS1',      # GEF for RAS activation
    'GRB2',      # Adaptor protein

    # Apoptosis/cell cycle
    'TP53',      # Apoptosis gatekeeper
    'CDKN2A',    # p16, cell cycle regulator

    # Additional candidates if needed
    'NF1',       # RAS negative regulator
    'IGF1R',     # Bypass pathway
]
```

**Selection Criteria**:
1. Present in both CCLE expression AND methylation datasets
2. Known involvement in EGFR pathway (KEGG, Reactome databases)
3. Sufficient variance across cell lines (σ² > threshold)
4. Literature support for erlotinib resistance mechanisms

---

### 1.3 Cell Line Stratification

**Objective**: Define "sensitive" vs "resistant" cohorts for erlotinib

#### Step 1: Extract Erlotinib Response
```python
# From GDSC data
erlotinib_data = gdsc_df[gdsc_df['DRUG_NAME'].str.contains('Erlotinib', case=False)]

# Merge with CCLE cell lines (use CellLineName mapping)
cell_lines_with_response = merge_ccle_gdsc_ids(erlotinib_data, metadata)
```

#### Step 2: Define Sensitivity Thresholds
**Method**: IC₅₀-based stratification with quality filters

```python
# Quality filters
valid_responses = erlotinib_data[
    (erlotinib_data['RMSE'] < 0.3) &  # Good curve fit
    (~erlotinib_data['LN_IC50'].isna())
]

# Define groups by IC50 percentiles
ic50_values = np.exp(valid_responses['LN_IC50'])  # Convert back to μM

# Stratification
SENSITIVE_THRESHOLD = np.percentile(ic50_values, 25)   # Lower quartile (e.g., < 0.1 μM)
RESISTANT_THRESHOLD = np.percentile(ic50_values, 75)   # Upper quartile (e.g., > 10 μM)

sensitive_lines = valid_responses[ic50_values < SENSITIVE_THRESHOLD]['CELL_LINE_NAME']
resistant_lines = valid_responses[ic50_values > RESISTANT_THRESHOLD]['CELL_LINE_NAME']

# Exclude intermediate responders for clearer signal
# Result: ~50-100 cell lines per group
```

**Alternative Method** (if Z-scores available):
```python
# Z-score based (normalized across all drugs)
SENSITIVE = Z_SCORE < -1.0  # More sensitive than average
RESISTANT = Z_SCORE > +1.0  # More resistant than average
```

---

### 1.4 Data Alignment and Filtering

#### Step 1: Identify Common Cell Lines
```python
# Find cell lines present in ALL datasets
expression_lines = set(expression_df.index)
methylation_lines = set(methylation_df.index)
erlotinib_lines = set(sensitive_lines) | set(resistant_lines)

common_lines = expression_lines & methylation_lines & erlotinib_lines

# Separate by phenotype
sensitive_final = [cl for cl in sensitive_lines if cl in common_lines]
resistant_final = [cl for cl in resistant_lines if cl in common_lines]

print(f"Sensitive: {len(sensitive_final)} cell lines")
print(f"Resistant: {len(resistant_final)} cell lines")
# Expected: 30-80 per group
```

#### Step 2: Extract Gene-Matched Data
```python
# For each gene in EGFR_PATHWAY_GENES:
for gene in EGFR_PATHWAY_GENES:
    # Expression: Direct lookup (log2 TPM values)
    expr_data = expression_df.loc[common_lines, gene]

    # Methylation: TSS region lookup
    # Find column matching gene TSS (may have chromosome coordinates)
    meth_cols = [c for c in methylation_df.columns if gene in c and 'TSS' in c]

    if meth_cols:
        # Average multiple CpG sites if present
        meth_data = methylation_df.loc[common_lines, meth_cols].mean(axis=1)
    else:
        # Gene not in methylation dataset - exclude from analysis
        print(f"Warning: {gene} not in methylation data")
        continue

    # Store aligned data
    gene_data[gene] = {
        'expression': expr_data,
        'methylation': meth_data,
        'phenotype': [phenotype_map[cl] for cl in common_lines]
    }
```

**Expected Data Structure**:
```python
gene_data = {
    'EGFR': {
        'expression': Series(80 cell lines, float, range [0-15]),
        'methylation': Series(80 cell lines, float, range [0-1]),
        'phenotype': List(80 labels, {'sensitive', 'resistant'})
    },
    'KRAS': {...},
    # ... 15-20 genes total
}
```

---

### 1.5 Discretization to Categorical States

**Rationale**: THRML currently supports SpinNode and CategoricalNode (discrete), not continuous

#### Discretization Strategy: Tertile-Based Binning

**For Each Gene and Data Type**:
```python
def discretize_tertiles(continuous_values, labels=['low', 'med', 'high']):
    """
    Convert continuous data to 3-state categorical.

    Args:
        continuous_values: np.array or pd.Series
        labels: State labels (maps to {0, 1, 2} for CategoricalNode)

    Returns:
        categorical_values: np.array of {0, 1, 2}
        thresholds: (t1, t2) tertile boundaries
    """
    t1 = np.percentile(continuous_values, 33.33)
    t2 = np.percentile(continuous_values, 66.67)

    categorical = np.zeros(len(continuous_values), dtype=int)
    categorical[continuous_values >= t1] = 1  # medium
    categorical[continuous_values >= t2] = 2  # high

    return categorical, (t1, t2)

# Apply to each gene
for gene in gene_data:
    # Expression discretization
    expr_discrete, expr_thresh = discretize_tertiles(
        gene_data[gene]['expression']
    )

    # Methylation discretization
    meth_discrete, meth_thresh = discretize_tertiles(
        gene_data[gene]['methylation']
    )

    # Store
    gene_data[gene]['expression_discrete'] = expr_discrete
    gene_data[gene]['methylation_discrete'] = meth_discrete
    gene_data[gene]['thresholds'] = {
        'expression': expr_thresh,
        'methylation': meth_thresh
    }
```

**Alternative: Biologically-Informed Thresholds**
```python
# Use known expression ranges for cancer genes
EXPRESSION_THRESHOLDS = {
    'EGFR': (8.0, 10.5),  # log2(TPM+1) thresholds from literature
    'KRAS': (7.5, 9.5),
    # ... gene-specific if available
}

# For methylation: 0-0.3 (low), 0.3-0.7 (med), 0.7-1.0 (high)
METHYLATION_THRESHOLDS = (0.3, 0.7)  # Standard β-value interpretation
```

**Validation of Discretization**:
```python
# Check distribution balance
for gene in gene_data:
    expr_counts = np.bincount(gene_data[gene]['expression_discrete'])
    meth_counts = np.bincount(gene_data[gene]['methylation_discrete'])

    print(f"{gene} Expression: Low={expr_counts[0]}, Med={expr_counts[1]}, High={expr_counts[2]}")
    print(f"{gene} Methylation: Low={meth_counts[0]}, Med={meth_counts[1]}, High={meth_counts[2]}")

    # Warn if severely imbalanced (e.g., >80% in one category)
    if max(expr_counts) / sum(expr_counts) > 0.8:
        print(f"WARNING: {gene} expression is severely skewed")
```

---

### 1.6 Data Quality Checks

#### QC Metrics
```python
def quality_check(gene_data):
    """Perform comprehensive QC on preprocessed data"""

    qc_report = {}

    for gene in gene_data:
        # 1. Missing data check
        expr_missing = np.isnan(gene_data[gene]['expression']).sum()
        meth_missing = np.isnan(gene_data[gene]['methylation']).sum()

        # 2. Variance check (continuous data)
        expr_var = np.var(gene_data[gene]['expression'])
        meth_var = np.var(gene_data[gene]['methylation'])

        # 3. Correlation (sanity check: methylation should anti-correlate with expression)
        corr = np.corrcoef(
            gene_data[gene]['expression'],
            gene_data[gene]['methylation']
        )[0, 1]

        # 4. Phenotype separation (t-test)
        sensitive_expr = gene_data[gene]['expression'][phenotype == 'sensitive']
        resistant_expr = gene_data[gene]['expression'][phenotype == 'resistant']
        t_stat, p_val = scipy.stats.ttest_ind(sensitive_expr, resistant_expr)

        qc_report[gene] = {
            'expr_missing': expr_missing,
            'meth_missing': meth_missing,
            'expr_variance': expr_var,
            'meth_variance': meth_var,
            'expr_meth_corr': corr,
            'phenotype_diff_pval': p_val
        }

    return pd.DataFrame(qc_report).T
```

**Exclusion Criteria**:
- Missing data >20%
- Variance <0.1 (effectively constant)
- Unexpected positive correlation between methylation and expression (should be negative)

---

### 1.7 Final Preprocessed Data Format

**Output**: Structured arrays for THRML input

```python
# Per-phenotype data structures
sensitive_data = {
    'n_samples': 50,  # number of cell lines
    'n_genes': 15,    # number of genes after QC
    'genes': ['EGFR', 'KRAS', ...],  # gene names

    # CategoricalNode states (n_samples × n_genes)
    'expression': np.array([[0, 2, 1, ...],   # Cell line 1
                           [1, 1, 2, ...],   # Cell line 2
                           ...], dtype=int),  # Values ∈ {0, 1, 2}

    'methylation': np.array([[2, 0, 1, ...],  # Cell line 1
                            [1, 1, 0, ...],  # Cell line 2
                            ...], dtype=int),

    # Metadata
    'cell_lines': ['ACH-000001', 'ACH-000002', ...],
    'thresholds': {...},  # For reverse-mapping discrete → continuous
}

resistant_data = {...}  # Same structure
```

**File Output**:
```python
# Save preprocessed data
import pickle

with open('data/processed/sensitive_discretized.pkl', 'wb') as f:
    pickle.dump(sensitive_data, f)

with open('data/processed/resistant_discretized.pkl', 'wb') as f:
    pickle.dump(resistant_data, f)
```

---

## 2. THRML Model Construction

### 2.1 Probabilistic Graphical Model Design

#### Variable Definitions

For each gene *g* in {EGFR, KRAS, ..., N genes}:

**Methylation Variable** M_g:
- **Type**: `CategoricalNode(num_categories=3)`
- **States**: {0="low", 1="medium", 2="high"}
- **Role**: Causal "anchor" (methylation is upstream of expression)

**Expression Variable** E_g:
- **Type**: `CategoricalNode(num_categories=3)`
- **States**: {0="low", 1="medium", 2="high"}
- **Role**: Downstream outcome (regulated by methylation)

**Total Variables per Model**: 2N (N methylation + N expression nodes)

#### Factor Graph Structure

**Energy Function**:
```
E_total(M, E | θ) = Σ_g E_local(M_g, E_g) + Σ_{g,h} E_pairwise(E_g, E_h)
```

**Components**:

1. **Local Methylation → Expression Factors**:
   - One factor per gene: Ψ(M_g, E_g)
   - Encodes: "High methylation → low expression" (typical pattern)
   - Learned from data or INDRA priors

2. **Pairwise Expression Factors**:
   - Between genes in pathway: Ψ(E_g, E_h)
   - Encodes: Pathway interactions (e.g., EGFR → KRAS → BRAF)
   - Derived from INDRA statements

---

### 2.2 INDRA Prior Integration

**INDRA** = Integrated Network and Dynamical Reasoning Assembler
- **Purpose**: Extract pathway knowledge from literature
- **API**: REST endpoint at `http://api.indra.bio:8000`
- **Output**: Probabilistic statements about gene interactions

#### Query INDRA for Pathway Structure

```python
import requests

def query_indra_interactions(gene_list):
    """
    Retrieve pairwise interactions from INDRA API.

    Returns:
        interactions: dict {(gene1, gene2): {'type': str, 'belief': float}}
    """
    interactions = {}

    for i, gene1 in enumerate(gene_list):
        for gene2 in gene_list[i+1:]:
            # Query for directed statements
            url = f"http://api.indra.bio:8000/statements/from_agents"
            params = {
                'subject': gene1,
                'object': gene2,
                'type': 'Phosphorylation,Activation,Inhibition'
            }

            response = requests.get(url, params=params)
            statements = response.json()

            if statements['statements']:
                # Extract highest-belief statement
                best_stmt = max(statements['statements'],
                               key=lambda s: s.get('belief', 0))

                interactions[(gene1, gene2)] = {
                    'type': best_stmt['type'],
                    'direction': 'forward',  # gene1 → gene2
                    'belief': best_stmt['belief']  # ∈ [0, 1]
                }

            # Also check reverse direction
            params_rev = {'subject': gene2, 'object': gene1, ...}
            # ... (similar logic)

    return interactions

# Execute
indra_priors = query_indra_interactions(EGFR_PATHWAY_GENES)
```

**Example Output**:
```python
indra_priors = {
    ('EGFR', 'KRAS'): {'type': 'Activation', 'direction': 'forward', 'belief': 0.92},
    ('KRAS', 'BRAF'): {'type': 'Activation', 'direction': 'forward', 'belief': 0.88},
    ('PIK3CA', 'AKT1'): {'type': 'Phosphorylation', 'direction': 'forward', 'belief': 0.95},
    ('PTEN', 'AKT1'): {'type': 'Inhibition', 'direction': 'forward', 'belief': 0.87},
    # ... ~30-50 interactions for 15 genes
}
```

---

### 2.3 Factor Weight Initialization

#### Convert INDRA Beliefs to Factor Weights

**Principle**: Higher belief → stronger coupling in energy function

```python
def indra_to_weights(indra_priors, activation_strength=2.0, inhibition_strength=-2.0):
    """
    Convert INDRA statements to factor weights.

    Args:
        activation_strength: Base weight for activating interactions
        inhibition_strength: Base weight for inhibiting interactions (negative)

    Returns:
        weight_matrix: (n_pairs, 3, 3) array for CategoricalEBMFactor
    """
    n_genes = len(EGFR_PATHWAY_GENES)
    gene_to_idx = {g: i for i, g in enumerate(EGFR_PATHWAY_GENES)}

    # Initialize all pairwise weights
    pairwise_weights = []
    gene_pairs = []

    for (gene1, gene2), info in indra_priors.items():
        idx1, idx2 = gene_to_idx[gene1], gene_to_idx[gene2]

        # Create 3×3 weight matrix (for 3 states each)
        W = np.zeros((3, 3))

        # Belief-scaled strength
        belief = info['belief']

        if info['type'] in ['Activation', 'Phosphorylation']:
            # Positive interaction: High E1 → High E2
            base_strength = activation_strength * belief
            W[2, 2] = base_strength    # High-High
            W[1, 1] = base_strength/2  # Med-Med
            W[0, 0] = -base_strength/2 # Low-Low (penalize)

        elif info['type'] == 'Inhibition':
            # Negative interaction: High E1 → Low E2
            base_strength = inhibition_strength * belief
            W[2, 0] = -base_strength   # High E1, Low E2
            W[0, 2] = base_strength    # Low E1, High E2

        pairwise_weights.append(W)
        gene_pairs.append((idx1, idx2))

    return np.array(pairwise_weights), gene_pairs
```

#### Methylation → Expression Weights

**Prior**: Negative correlation expected (high methylation silences genes)

```python
def methylation_expression_weights(n_genes, default_strength=-1.5):
    """
    Create weights for M_g → E_g factors.

    Returns:
        weights: (n_genes, 3, 3) array [meth_state, expr_state]
    """
    W_meth_expr = []

    for g in range(n_genes):
        W = np.zeros((3, 3))  # [methylation state, expression state]

        # Negative correlation pattern
        W[0, 2] = default_strength    # Low meth → High expr
        W[0, 1] = default_strength/2
        W[1, 1] = default_strength/3  # Med meth → Med expr
        W[2, 0] = default_strength    # High meth → Low expr
        W[2, 1] = default_strength/2

        W_meth_expr.append(W)

    return np.array(W_meth_expr)
```

---

### 2.4 THRML Factor Construction

```python
import thrml
import jax.numpy as jnp

# Create nodes
methylation_nodes = [
    thrml.CategoricalNode(shape=(), num_categories=3, name=f"M_{gene}")
    for gene in EGFR_PATHWAY_GENES
]

expression_nodes = [
    thrml.CategoricalNode(shape=(), num_categories=3, name=f"E_{gene}")
    for gene in EGFR_PATHWAY_GENES
]

# Create blocks
meth_block = thrml.Block(methylation_nodes)
expr_block = thrml.Block(expression_nodes)

# Factor 1: Methylation → Expression (per gene)
meth_expr_weights = methylation_expression_weights(n_genes=len(EGFR_PATHWAY_GENES))

meth_expr_factor = thrml.CategoricalEBMFactor(
    categorical_node_groups=[meth_block, expr_block],
    weights=jnp.array(meth_expr_weights),  # (n_genes, 3, 3)
    is_spin={}  # All categorical, no spins
)

# Factor 2: Expression ↔ Expression (pathway interactions)
expr_expr_weights, gene_pairs = indra_to_weights(indra_priors)

# Need to create batch-compatible structure for all pairs
# This requires indexing into expr_block for each pair
# (Implementation detail: use InteractionGroups or multiple factors)

expr_expr_factor = thrml.CategoricalEBMFactor(
    categorical_node_groups=[expr_block, expr_block],  # Self-interaction
    weights=jnp.array(expr_expr_weights),  # (n_pairs, 3, 3)
    is_spin={}
)

# Combine into FactorizedEBM
factors = [meth_expr_factor, expr_expr_factor]

model_ebm = thrml.FactorizedEBM(
    factors=factors,
    categorical_node_config={
        'num_categories': 3,
        'shape': ()
    }
)
```

---

### 2.5 Block Gibbs Sampling Setup

```python
# Define sampling specification
gibbs_spec = thrml.BlockGibbsSpec(
    free_blocks=[meth_block, expr_block],
    sampling_order=[0, 1],  # Alternate: sample methylation, then expression
    clamped_blocks=[]  # No observations during free sampling
)

# Conditional samplers for each block
samplers = {
    meth_block: thrml.CategoricalGibbsConditional(),
    expr_block: thrml.CategoricalGibbsConditional()
}

# Create sampling program
sampling_program = thrml.FactorSamplingProgram(
    gibbs_spec=gibbs_spec,
    samplers=samplers,
    factors=factors
)

# Sampling schedule
schedule = thrml.SamplingSchedule(
    n_warmup=500,         # Burn-in iterations
    n_samples=1000,       # Posterior samples to collect
    steps_per_sample=10   # Thinning (take every 10th sample)
)
```

---

## 3. Causal Inference Procedure

### 3.1 Model Discrimination Framework

**Goal**: For each gene pair (g, h), determine causal direction: E_g → E_h vs. E_h → E_g

**Method**: Free energy comparison via thermodynamic discrimination

#### Hypothesis Models

**Model A**: E_g → E_h (g causes h)
```
E_A(M_g, M_h, E_g, E_h) = -α_g · ψ(M_g, E_g)      # M_g influences E_g
                          -α_h · ψ(M_h, E_h)      # M_h influences E_h
                          -β · ψ(E_g, E_h)        # E_g causes E_h
                          +λ · ||data - model||²  # Data fidelity
```

**Model B**: E_h → E_g (h causes g)
```
E_B(M_g, M_h, E_g, E_h) = -α_g · ψ(M_g, E_g)
                          -α_h · ψ(M_h, E_h)
                          -β · ψ(E_h, E_g)        # E_h causes E_g (reversed!)
                          +λ · ||data - model||²
```

**Key Difference**: Direction of E_g ↔ E_h coupling

---

### 3.2 Free Energy Computation

**Free Energy**: F = -log Z = -log ∫ exp(-E(x)) dx

**Estimation via Sampling**:

```python
def estimate_free_energy(sampling_program, data, model_params, n_samples=1000):
    """
    Estimate free energy F = -log(E[exp(-E(samples))]).

    Args:
        sampling_program: THRML BlockSamplingProgram
        data: Observed data to clamp
        model_params: Factor weights for this model
        n_samples: Number of MC samples

    Returns:
        F: Free energy estimate
        F_std: Standard error
    """
    import jax
    import jax.numpy as jnp

    # Sample from model
    key = jax.random.PRNGKey(0)

    # Option 1: Clamped sampling (data as constraints)
    gibbs_spec_clamped = thrml.BlockGibbsSpec(
        free_blocks=[expr_block],  # Sample expression given methylation
        sampling_order=[0],
        clamped_blocks=[meth_block]  # Fix methylation to observed data
    )

    # Set methylation to observed values
    initial_state = {...}  # Initialize from data

    # Run Gibbs sampling
    samples = thrml.sample_with_observation(
        key=key,
        program=sampling_program,
        schedule=schedule,
        initial_state=initial_state,
        observer=thrml.StateObserver([expr_block])
    )

    # Compute energies for all samples
    energies = []
    for sample in samples:
        E_val = model_ebm.energy(sample, block_spec)
        energies.append(E_val)

    energies = np.array(energies)

    # Free energy: F = -log(mean(exp(-E)))
    # Numerically stable computation:
    E_min = np.min(energies)
    F = E_min - np.log(np.mean(np.exp(-(energies - E_min))))

    # Bootstrap for error estimate
    F_bootstrap = []
    for _ in range(100):
        idx = np.random.choice(len(energies), size=len(energies), replace=True)
        F_boot = E_min - np.log(np.mean(np.exp(-(energies[idx] - E_min))))
        F_bootstrap.append(F_boot)

    F_std = np.std(F_bootstrap)

    return F, F_std
```

---

### 3.3 Pairwise Causal Direction Testing

**For each gene pair** (g, h):

```python
def test_causal_direction(gene_g, gene_h, data_dict, n_samples=1000):
    """
    Test E_g → E_h vs. E_h → E_g using free energy.

    Returns:
        direction: 'g->h', 'h->g', or 'undecided'
        delta_F: F_B - F_A (discriminator)
        confidence: Statistical confidence (|ΔF| / σ)
    """
    # Build Model A: E_g → E_h
    model_A = build_model_with_direction(
        gene_g, gene_h, direction='forward'
    )

    # Build Model B: E_h → E_g
    model_B = build_model_with_direction(
        gene_g, gene_h, direction='reverse'
    )

    # Estimate free energies
    F_A, sigma_A = estimate_free_energy(model_A, data_dict, n_samples)
    F_B, sigma_B = estimate_free_energy(model_B, data_dict, n_samples)

    # Compute difference
    delta_F = F_B - F_A
    sigma_delta = np.sqrt(sigma_A**2 + sigma_B**2)

    # Decision threshold (typically 1 k_B T = 1.0 in natural units)
    THRESHOLD = 1.0

    if delta_F > THRESHOLD:
        direction = 'g->h'  # Model A preferred
    elif delta_F < -THRESHOLD:
        direction = 'h->g'  # Model B preferred
    else:
        direction = 'undecided'

    # Confidence (z-score)
    confidence = abs(delta_F) / sigma_delta if sigma_delta > 0 else 0.0

    return {
        'direction': direction,
        'delta_F': delta_F,
        'F_A': F_A,
        'F_B': F_B,
        'confidence': confidence,
        'p_value': 2 * (1 - scipy.stats.norm.cdf(abs(delta_F) / sigma_delta))
    }
```

**Interpretation**:
- ΔF > +1.0: Forward direction (g → h) is more likely
- ΔF < -1.0: Reverse direction (h → g) is more likely
- |ΔF| < 1.0: Insufficient evidence to discriminate

---

### 3.4 Network Construction

**Infer network for each phenotype**:

```python
def infer_causal_network(gene_list, data, n_samples=1000):
    """
    Infer full causal network structure.

    Returns:
        edges: List of (gene_i, gene_j, weight, confidence)
    """
    edges = []
    n_genes = len(gene_list)

    # Test all pairs (n choose 2)
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            gene_i, gene_j = gene_list[i], gene_list[j]

            result = test_causal_direction(gene_i, gene_j, data, n_samples)

            # Add edge if confident (p < 0.05)
            if result['p_value'] < 0.05:
                if result['direction'] == 'g->h':
                    edges.append((gene_i, gene_j, result['delta_F'], result['confidence']))
                elif result['direction'] == 'h->g':
                    edges.append((gene_j, gene_i, -result['delta_F'], result['confidence']))

    return edges

# Execute for both phenotypes
sensitive_network = infer_causal_network(
    EGFR_PATHWAY_GENES,
    sensitive_data,
    n_samples=1000
)

resistant_network = infer_causal_network(
    EGFR_PATHWAY_GENES,
    resistant_data,
    n_samples=1000
)
```

**Expected Output**:
```python
sensitive_network = [
    ('EGFR', 'KRAS', 2.3, 4.5),   # Edge, ΔF, confidence
    ('KRAS', 'BRAF', 1.8, 3.2),
    ('PIK3CA', 'AKT1', 2.1, 4.0),
    # ... 20-40 edges total
]

resistant_network = [
    ('MET', 'KRAS', 2.5, 4.8),    # Bypass edge (new in resistance)
    ('KRAS', 'BRAF', 1.7, 3.1),   # Preserved edge
    # ... different network structure
]
```

---

### 3.5 Network Comparison (Sensitive vs. Resistant)

**Identify changed edges**:

```python
def compare_networks(network_A, network_B, threshold=0.5):
    """
    Find edges that differ between two networks.

    Returns:
        new_edges: Present in B, absent in A (bypass mechanisms)
        lost_edges: Present in A, absent in B
        flipped_edges: Reversed direction
        weakened_edges: Significantly weaker in B
    """
    edges_A = {(e[0], e[1]): e[2] for e in network_A}  # {(src, dst): weight}
    edges_B = {(e[0], e[1]): e[2] for e in network_B}

    new_edges = []
    lost_edges = []
    flipped_edges = []
    weakened_edges = []

    # Check edges in B
    for (src, dst), weight_B in edges_B.items():
        if (src, dst) not in edges_A:
            # Check for flip
            if (dst, src) in edges_A:
                flipped_edges.append((src, dst, weight_B))
            else:
                new_edges.append((src, dst, weight_B))
        else:
            weight_A = edges_A[(src, dst)]
            # Check for significant weakening
            if abs(weight_B) < abs(weight_A) * 0.5:  # >50% reduction
                weakened_edges.append((src, dst, weight_A, weight_B))

    # Check edges lost in B
    for (src, dst), weight_A in edges_A.items():
        if (src, dst) not in edges_B and (dst, src) not in edges_B:
            lost_edges.append((src, dst, weight_A))

    return {
        'new_edges': new_edges,
        'lost_edges': lost_edges,
        'flipped_edges': flipped_edges,
        'weakened_edges': weakened_edges
    }

# Execute comparison
network_changes = compare_networks(sensitive_network, resistant_network)

print(f"New bypass edges: {len(network_changes['new_edges'])}")
print(f"Lost edges: {len(network_changes['lost_edges'])}")
print(f"Flipped directions: {len(network_changes['flipped_edges'])}")
```

**Expected Output**:
```python
network_changes = {
    'new_edges': [
        ('MET', 'KRAS', 2.5),     # MET bypass activation
        ('IGF1R', 'PIK3CA', 1.9), # Alternative PI3K activation
    ],
    'lost_edges': [
        ('EGFR', 'KRAS', 2.3),    # Original EGFR signaling lost
    ],
    'flipped_edges': [
        ('PTEN', 'AKT1', -1.2),   # Feedback reversal
    ],
    'weakened_edges': [
        ('EGFR', 'PIK3CA', 2.1, 0.8),  # Weakened by 62%
    ]
}
```

---

## 4. Validation Framework

### 4.1 Drug Prediction from Network Changes

**Principle**: Target bypass mechanisms identified in resistant network

#### Step 1: Map Changed Edges to Drug Targets

```python
def predict_drugs_from_network_changes(network_changes, indra_priors):
    """
    Map network changes to targetable proteins and drugs.

    Returns:
        drug_predictions: List of (drug_name, target, mechanism, score)
    """
    drug_predictions = []

    # Focus on NEW edges (bypass mechanisms)
    for (src_gene, dst_gene, weight) in network_changes['new_edges']:
        # Query INDRA for drugs targeting src_gene (upstream of bypass)
        drugs = query_indra_drugs(target_gene=src_gene)

        for drug in drugs:
            # Score based on:
            # 1. Strength of bypass edge (weight)
            # 2. Drug-target binding affinity (from INDRA)
            # 3. INDRA belief score
            score = weight * drug['binding_affinity'] * drug['belief']

            drug_predictions.append({
                'drug': drug['name'],
                'target': src_gene,
                'mechanism': f"Inhibit {src_gene}→{dst_gene} bypass",
                'score': score,
                'edge_weight': weight
            })

    # Rank by score
    drug_predictions.sort(key=lambda x: x['score'], reverse=True)

    return drug_predictions
```

**Example Output**:
```python
drug_predictions = [
    {'drug': 'Crizotinib', 'target': 'MET', 'mechanism': 'Inhibit MET→KRAS bypass',
     'score': 5.2, 'edge_weight': 2.5},
    {'drug': 'Cabozantinib', 'target': 'MET', 'mechanism': 'Inhibit MET→KRAS bypass',
     'score': 4.8, 'edge_weight': 2.5},
    {'drug': 'Linsitinib', 'target': 'IGF1R', 'mechanism': 'Inhibit IGF1R→PIK3CA bypass',
     'score': 3.9, 'edge_weight': 1.9},
    # ... 10-20 predictions
]
```

---

### 4.2 Validation Against GDSC IC50 Data

**Ground Truth**: Measure if predicted drugs actually work in resistant cell lines

```python
def validate_drug_predictions(drug_predictions, gdsc_data, resistant_lines):
    """
    Check if predicted drugs show efficacy in resistant cell lines.

    Returns:
        validation_results: Dict with precision, recall, enrichment
    """
    # Extract IC50 data for resistant cell lines
    resistant_ic50 = gdsc_data[gdsc_data['CELL_LINE_NAME'].isin(resistant_lines)]

    # Define "effective drug" as IC50 < threshold (e.g., median across all drugs)
    ic50_threshold = np.median(resistant_ic50['IC50'])

    validated_predictions = []

    for pred in drug_predictions[:10]:  # Top 10 predictions
        drug_name = pred['drug']

        # Find IC50 data for this drug in resistant lines
        drug_data = resistant_ic50[
            resistant_ic50['DRUG_NAME'].str.contains(drug_name, case=False)
        ]

        if len(drug_data) > 0:
            mean_ic50 = np.exp(drug_data['LN_IC50'].mean())  # Convert back to μM

            # Check if effective
            is_effective = mean_ic50 < ic50_threshold

            validated_predictions.append({
                'drug': drug_name,
                'target': pred['target'],
                'predicted_score': pred['score'],
                'measured_IC50': mean_ic50,
                'is_effective': is_effective,
                'n_tested_lines': len(drug_data)
            })

    # Compute metrics
    n_effective = sum(p['is_effective'] for p in validated_predictions)
    precision = n_effective / len(validated_predictions) if validated_predictions else 0.0

    # Baseline: random drug selection
    random_precision = 0.15  # ~15% of drugs work by chance
    enrichment = precision / random_precision if random_precision > 0 else 0.0

    return {
        'validated_predictions': validated_predictions,
        'precision': precision,
        'baseline_precision': random_precision,
        'enrichment': enrichment,
        'n_predicted': len(validated_predictions),
        'n_effective': n_effective
    }
```

**Success Metrics**:
- **Minimum Viable**: Precision > 30% (2× baseline)
- **Competitive**: Precision > 40%, Enrichment > 2.5×
- **Winning**: Precision > 50%, Novel mechanistic insights

---

### 4.3 Statistical Significance Testing

```python
def compute_validation_significance(validation_results, n_permutations=1000):
    """
    Test if drug prediction performance is statistically significant.

    Uses permutation test: shuffle drug-target assignments and recompute precision.
    """
    observed_precision = validation_results['precision']

    # Permutation test
    all_drugs = list(validation_results['validated_predictions'])
    null_precisions = []

    for _ in range(n_permutations):
        # Shuffle is_effective labels randomly
        shuffled_effective = np.random.permutation(
            [p['is_effective'] for p in all_drugs]
        )
        null_precision = np.mean(shuffled_effective)
        null_precisions.append(null_precision)

    # Compute p-value
    p_value = np.mean(np.array(null_precisions) >= observed_precision)

    # Effect size (Cohen's h for proportions)
    h = 2 * (np.arcsin(np.sqrt(observed_precision)) -
             np.arcsin(np.sqrt(validation_results['baseline_precision'])))

    return {
        'p_value': p_value,
        'effect_size_h': h,
        'null_mean': np.mean(null_precisions),
        'null_std': np.std(null_precisions)
    }
```

---

## 5. Demo Execution Workflow

### 5.1 Complete Pipeline Script

**File**: `scripts/02_run_inference.py`

```python
#!/usr/bin/env python3
"""
XTR-0 Hackathon: Full Inference Pipeline
Runs thermodynamic causal inference end-to-end.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Import local modules
from core.data_loader import load_and_preprocess_data
from core.thrml_model import build_thrml_model, infer_causal_network
from core.validation import validate_predictions, compute_metrics

def main(args):
    print("=== XTR-0 Thermodynamic Causal Inference ===\n")

    # -----------------
    # 1. LOAD DATA
    # -----------------
    print("[1/6] Loading and preprocessing data...")

    if args.synthetic_data:
        # Generate synthetic data with known ground truth
        from core.data_loader import generate_synthetic_data
        sensitive_data, resistant_data, ground_truth = generate_synthetic_data(
            n_genes=args.genes,
            n_samples=args.samples
        )
    else:
        # Load real CCLE/GDSC data
        sensitive_data, resistant_data = load_and_preprocess_data(
            ccle_expression='data/raw/ccle/CCLE_expression_TPM.csv',
            ccle_methylation='data/raw/ccle/CCLE_RRBS_methylation.txt',
            gdsc_ic50='data/raw/gdsc/GDSC_drug_data_fitted.xlsx',
            target_genes=args.target_genes or EGFR_PATHWAY_GENES[:args.genes],
            discretize_method='tertiles'
        )

    print(f"  Sensitive: {sensitive_data['n_samples']} cell lines, "
          f"{sensitive_data['n_genes']} genes")
    print(f"  Resistant: {resistant_data['n_samples']} cell lines, "
          f"{resistant_data['n_genes']} genes")

    # -----------------
    # 2. BUILD MODELS
    # -----------------
    print("\n[2/6] Building THRML models...")

    # Build for sensitive
    sensitive_model = build_thrml_model(
        gene_list=sensitive_data['genes'],
        use_indra_priors=not args.no_indra
    )

    # Build for resistant (same structure, different data)
    resistant_model = build_thrml_model(
        gene_list=resistant_data['genes'],
        use_indra_priors=not args.no_indra
    )

    print(f"  Model nodes: {len(sensitive_model['nodes'])} "
          f"(2 × {sensitive_data['n_genes']} genes)")
    print(f"  Model factors: {len(sensitive_model['factors'])}")

    # -----------------
    # 3. INFER NETWORKS
    # -----------------
    print("\n[3/6] Inferring causal networks via thermodynamic sampling...")

    sensitive_network = infer_causal_network(
        model=sensitive_model,
        data=sensitive_data,
        n_samples=args.samples,
        n_warmup=args.warmup,
        verbose=args.verbose
    )

    resistant_network = infer_causal_network(
        model=resistant_model,
        data=resistant_data,
        n_samples=args.samples,
        n_warmup=args.warmup,
        verbose=args.verbose
    )

    print(f"  Sensitive network: {len(sensitive_network)} edges")
    print(f"  Resistant network: {len(resistant_network)} edges")

    # -----------------
    # 4. COMPARE NETWORKS
    # -----------------
    print("\n[4/6] Comparing networks (sensitive vs resistant)...")

    from core.inference import compare_networks
    network_changes = compare_networks(sensitive_network, resistant_network)

    print(f"  New bypass edges: {len(network_changes['new_edges'])}")
    print(f"  Lost edges: {len(network_changes['lost_edges'])}")
    print(f"  Flipped edges: {len(network_changes['flipped_edges'])}")

    # Display key changes
    if network_changes['new_edges']:
        print("\n  Top bypass mechanisms:")
        for src, dst, weight in network_changes['new_edges'][:5]:
            print(f"    {src} → {dst} (ΔF = {weight:.2f})")

    # -----------------
    # 5. PREDICT DRUGS
    # -----------------
    print("\n[5/6] Predicting drugs from network changes...")

    from core.inference import predict_drugs_from_network_changes
    drug_predictions = predict_drugs_from_network_changes(
        network_changes=network_changes,
        use_indra=not args.no_indra
    )

    print(f"  Generated {len(drug_predictions)} drug predictions")
    print("\n  Top 5 predictions:")
    for i, pred in enumerate(drug_predictions[:5], 1):
        print(f"    {i}. {pred['drug']} (target: {pred['target']}, score: {pred['score']:.2f})")

    # -----------------
    # 6. VALIDATE
    # -----------------
    print("\n[6/6] Validating predictions against GDSC IC50 data...")

    validation_results = validate_predictions(
        drug_predictions=drug_predictions,
        gdsc_data='data/raw/gdsc/GDSC_drug_data_fitted.xlsx',
        resistant_cell_lines=resistant_data['cell_lines']
    )

    print(f"\n  Precision: {validation_results['precision']:.1%} "
          f"({validation_results['n_effective']}/{validation_results['n_predicted']})")
    print(f"  Baseline: {validation_results['baseline_precision']:.1%}")
    print(f"  Enrichment: {validation_results['enrichment']:.2f}×")

    # Statistical significance
    from core.validation import compute_validation_significance
    sig_test = compute_validation_significance(validation_results)
    print(f"  P-value: {sig_test['p_value']:.4f}")
    print(f"  Effect size (h): {sig_test['effect_size_h']:.3f}")

    # -----------------
    # SAVE RESULTS
    # -----------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save networks
    with open(output_dir / 'sensitive_network.pkl', 'wb') as f:
        pickle.dump(sensitive_network, f)

    with open(output_dir / 'resistant_network.pkl', 'wb') as f:
        pickle.dump(resistant_network, f)

    # Save predictions and validation
    results = {
        'network_changes': network_changes,
        'drug_predictions': drug_predictions,
        'validation': validation_results,
        'significance': sig_test
    }

    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Save summary JSON
    import json
    summary = {
        'n_genes': sensitive_data['n_genes'],
        'n_samples_sensitive': sensitive_data['n_samples'],
        'n_samples_resistant': resistant_data['n_samples'],
        'n_edges_sensitive': len(sensitive_network),
        'n_edges_resistant': len(resistant_network),
        'n_bypass_edges': len(network_changes['new_edges']),
        'n_drug_predictions': len(drug_predictions),
        'precision': float(validation_results['precision']),
        'enrichment': float(validation_results['enrichment']),
        'p_value': float(sig_test['p_value'])
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}/")

    # -----------------
    # SUCCESS CRITERIA
    # -----------------
    print("\n" + "="*60)
    print("DEMO SUCCESS CRITERIA:")
    print("="*60)

    if validation_results['precision'] >= 0.30:
        print("✓ Minimum Viable: Precision > 30%")
    else:
        print("✗ Minimum Viable: Precision > 30% (FAILED)")

    if validation_results['precision'] >= 0.40 and sig_test['p_value'] < 0.05:
        print("✓ Competitive: Precision > 40%, p < 0.05")
    else:
        print("✗ Competitive: Precision > 40%, p < 0.05")

    if validation_results['precision'] >= 0.50 and len(network_changes['new_edges']) >= 3:
        print("✓ Winning: Precision > 50%, Novel bypass mechanisms discovered")
    else:
        print("✗ Winning: Precision > 50%, Novel mechanisms")

    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XTR-0 Thermodynamic Causal Inference')

    # Data args
    parser.add_argument('--genes', type=int, default=15,
                       help='Number of genes to analyze')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of MCMC samples per model')
    parser.add_argument('--warmup', type=int, default=500,
                       help='Number of burn-in iterations')
    parser.add_argument('--target-genes', nargs='+',
                       help='Specific genes to analyze (overrides --genes)')

    # Model args
    parser.add_argument('--no-indra', action='store_true',
                       help='Disable INDRA prior integration')

    # Execution args
    parser.add_argument('--synthetic-data', action='store_true',
                       help='Use synthetic data with known ground truth')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (5 genes, 100 samples)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Quick test mode overrides
    if args.quick_test:
        args.genes = 5
        args.samples = 100
        args.warmup = 50
        args.synthetic_data = True
        print("QUICK TEST MODE: Using 5 genes, 100 samples, synthetic data\n")

    main(args)
```

---

### 5.2 Demo Execution Timeline

**Hour 0-1: Environment & Data**
```bash
# Install dependencies
pip install thrml jax jaxlib numpy pandas scipy matplotlib networkx requests openpyxl

# Test environment
python scripts/00_test_environment.py

# Download data
bash scripts/01_download_data.sh

# Quick test (5 min)
python scripts/02_run_inference.py --quick-test
```

**Hour 1-4: Full Inference (15 genes, 1000 samples)**
```bash
# Run on real data
python scripts/02_run_inference.py \
  --genes 15 \
  --samples 1000 \
  --warmup 500 \
  --output-dir results/full_run \
  --verbose

# Monitor progress
tail -f inference.log
```

**Hour 4-5: Analysis & Visualization**
```bash
python scripts/03_analyze_results.py \
  --results results/full_run \
  --make-figures \
  --make-report
```

**Hour 5-6: Validation Deep Dive**
```bash
python scripts/03_analyze_results.py \
  --results results/full_run \
  --validate-detailed \
  --bootstrap-ci 1000
```

**Hour 6-8: Presentation Prep**
- Generate final figures
- Write 2-page report
- Prepare 5-minute pitch

---

### 5.3 Expected Outputs

#### Network Comparison Figure
**File**: `results/figures/network_comparison.png`
- Side-by-side networks: sensitive (left) vs resistant (right)
- Nodes = genes, edges = causal influences
- Color coding: red = new bypass edges, blue = preserved, gray = lost
- Edge thickness ∝ |ΔF| (causal strength)

#### Validation Results Figure
**File**: `results/figures/precision_vs_baseline.png`
- Bar chart: Predicted drugs (precision %) vs Random baseline
- Error bars from bootstrap CI
- P-value annotation

#### Drug Ranking Table
**File**: `results/drug_predictions.csv`

| Rank | Drug | Target | Mechanism | Score | IC50 (μM) | Validated |
|------|------|--------|-----------|-------|-----------|-----------|
| 1 | Crizotinib | MET | Inhibit MET→KRAS | 5.2 | 0.8 | ✓ |
| 2 | Cabozantinib | MET | Inhibit MET→KRAS | 4.8 | 1.2 | ✓ |
| 3 | Linsitinib | IGF1R | Inhibit IGF1R→PI3K | 3.9 | 2.5 | ✓ |

---

### 5.4 Contingency Plans

**If inference is too slow**:
```bash
# Reduce scope
python scripts/02_run_inference.py --genes 10 --samples 500
```

**If THRML has bugs**:
```python
# Fallback: Use NumPy Gibbs sampler
from core.fallback_sampler import NumpyGibbsSampler
```

**If validation precision is low (<30%)**:
- Still show methodology works
- Emphasize: "validatable in principle"
- Discuss: "needs more genes / samples for higher power"

---

## Summary: Data → Demo Checklist

- [x] CCLE expression data downloaded and discretized
- [x] CCLE methylation data downloaded and discretized
- [x] GDSC IC50 data for validation
- [x] Gene selection (EGFR pathway, 15-20 genes)
- [x] Cell line stratification (sensitive vs resistant)
- [x] Data alignment and QC
- [x] THRML model construction with INDRA priors
- [x] Causal inference via free energy comparison
- [x] Network comparison (bypass mechanism identification)
- [x] Drug prediction from network changes
- [x] Validation against GDSC ground truth
- [x] Statistical significance testing
- [x] Visualization and reporting

**Total Pipeline**: Raw data → Validated drug predictions in 8 hours
