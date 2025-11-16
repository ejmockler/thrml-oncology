# CRITICAL TECHNICAL ASSESSMENT: XTR-0 Hackathon Demo

## Executive Summary

**Verdict**: Your proposal is **fundamentally sound but has critical implementation gaps** that will prevent execution without immediate fixes. The conceptual approach maximizes THRML's actual capabilities, but the code architecture contains significant deviations from the authoritative THRML API.

**Recommendation**: **Fix the implementation issues below before attempting execution on H100s**. You're 60% there - don't waste GPU time debugging basic API misuse.

---

## Part I: Conceptual Alignment Assessment

### ✅ WHAT YOU GOT RIGHT

#### 1. Discrete Variable Strategy
**Your Approach**: Discretize methylation/expression to 3 states, use `CategoricalNode`

**Assessment**: **PERFECT** ✓
- THRML does NOT have continuous variables ready (pmode/pmog are roadmap items)
- Discretization to biological regimes (low/med/high) is scientifically defensible
- `CategoricalNode` is fully implemented and tested
- This matches hardware reality (pdit = categorical sampler)

**Evidence from THRML docs**:
```
CategoricalNode: K-state discrete variables: {0, 1, ..., K-1}
Software simulation of pdit behavior
Supports arbitrary cardinality
```

#### 2. Block Gibbs Sampling Core
**Your Approach**: Use THRML's `BlockGibbsSpec` + `FactorSamplingProgram`

**Assessment**: **CORRECT** ✓
- Block Gibbs is THRML's native operation
- Direct hardware mapping: sampling cells → blocks
- This is exactly what TSUs will execute

**Evidence**:
```
BlockSamplingProgram: "The beating heart of THRML"
Handles reindexing, slicing, padding for vectorized execution
```

#### 3. Causal Discrimination via Free Energy
**Your Approach**: Compare F_A vs F_B to determine direction

**Assessment**: **THEORETICALLY SOUND** ✓
- Energy-based causal inference is valid
- ΔF threshold = 1.0 k_B T is standard in statistical physics
- Free energy differences discriminate model quality

**Mathematical foundation**:
```
P(x) ∝ exp(-E(x))
F = -log(Z)
ΔF > 0 → Model A favored
```

#### 4. Validation Strategy
**Your Approach**: Check predictions against GDSC IC50 data

**Assessment**: **EXCELLENT** ✓
- Direct ground truth validation
- Precision metric (% predictions that work) is appropriate
- Baseline comparison (random = 15%) is honest
- This will convince judges

#### 5. TSU Advantage Argument
**Your Approach**: 600× cost reduction via native block Gibbs

**Assessment**: **DEFENSIBLE** ✓
- Block Gibbs IS native to TSU (pbit/pdit circuits)
- Energy calculation via resistor networks (analog)
- GPU simulation overhead is real
- Projection is honest (labeled as "projected")

---

## Part II: CRITICAL IMPLEMENTATION PROBLEMS

### ❌ PROBLEM 1: Incorrect Factor API Usage

**Your Code** (thrml_model.py:84-92):
```python
# Factor 1: M1 -> E1
# TODO: Create CategoricalEBMFactor
# Hint: weights should be [1, 3, 3] for pairwise interaction
```

**Issue**: This violates THRML's factor construction API.

**Correct API** (from THRML_COMPREHENSIVE_DOCUMENTATION.md):
```python
from thrml.models.discrete_ebm import CategoricalEBMFactor

factor = CategoricalEBMFactor(
    categorical_node_groups=[block1, block2],  # List of Blocks
    weights=W  # Shape [batch, n_states_1, n_states_2]
)
```

**What You Need to Fix**:
1. Create `Block` objects FIRST: `Block([m1])`, `Block([e1])`
2. Pass Blocks to `categorical_node_groups`, NOT individual nodes
3. Weight shape must be `[1, 3, 3]` for batch=1, 3x3 pairwise
4. Weights encode energy contribution: negative = favorable

**Correct Implementation**:
```python
# Factor 1: M1 -> E1
m1_block = Block([m1])
e1_block = Block([e1])

# Weight matrix: encourage concordance
# W[i,j] = -1.0 if i==j (same state favored), else 0.0
W_m1_e1 = -jnp.eye(self.n_states)[None, :, :]  # Shape [1, 3, 3]

factor_m1_e1 = CategoricalEBMFactor(
    categorical_node_groups=[m1_block, e1_block],
    weights=W_m1_e1
)
```

**Severity**: **CRITICAL** - Code will crash immediately without this fix.

---

### ❌ PROBLEM 2: Missing Sampler Configuration

**Your Code** (thrml_model.py:180):
```python
# TODO: Create samplers for each block
samplers = {}
```

**Issue**: You MUST specify conditional samplers for each free block.

**Correct API**:
```python
from thrml.models.discrete_ebm import CategoricalGibbsConditional

samplers = {
    blocks[0]: CategoricalGibbsConditional(),  # Methylation block
    blocks[1]: CategoricalGibbsConditional(),  # Expression block
}
```

**Why This Matters**:
- `BlockSamplingProgram` expects `dict[Block -> ConditionalSampler]`
- `CategoricalGibbsConditional` computes softmax parameters from factors
- Without this, sampling cannot execute

**Severity**: **CRITICAL** - Sampling will fail without samplers.

---

### ❌ PROBLEM 3: Incorrect FactorSamplingProgram Construction

**Your Code** (thrml_model.py:183):
```python
# TODO: Build FactorSamplingProgram
program = None
```

**Correct Implementation**:
```python
from thrml.factor import FactorSamplingProgram

program = FactorSamplingProgram(
    gibbs_spec=spec,
    samplers=samplers,
    factors=factors,  # List of CategoricalEBMFactor
    other_interaction_groups=[]  # Optional additional interactions
)
```

**Key Insight from Docs**:
```
FactorSamplingProgram: Thin wrapper converting factors to interaction groups
Decomposes factors into directed InteractionGroups for block sampling
```

**Severity**: **CRITICAL** - No program = no sampling.

---

### ❌ PROBLEM 4: Sample Execution Missing

**Your Code** (thrml_model.py:193):
```python
# TODO: Run sampling and return samples
samples = None
```

**Correct Implementation**:
```python
from thrml.block_sampling import sample_with_observation
from thrml.observers import StateObserver

# Create observer to capture states
observer = StateObserver(blocks=spec.free_blocks)

# Initialize state
key = jax.random.PRNGKey(0)
block_spec = program.block_spec
initial_state = block_spec.make_empty_block_state()

# Run sampling
result = sample_with_observation(
    key=key,
    program=program,
    schedule=schedule,
    observer=observer,
    initial_free_state=initial_state,
    clamped_state={}  # No clamped variables in this case
)

# Extract samples from observer state
samples = result  # Observer returns list of states
```

**Severity**: **CRITICAL** - This is the actual sampling step.

---

### ❌ PROBLEM 5: Energy Computation Not Implemented

**Your Code** (thrml_model.py:139):
```python
# TODO: Implement energy computation
# Sum over all factors
total_energy = 0.0

for factor in factors:
    # Extract relevant states from factor.node_groups
    # Compute factor energy
    # Add to total
    pass
```

**Issue**: Factors have an `energy()` method but you need to pass global state.

**Correct Pattern**:
```python
# Convert state dict to global representation
global_state = block_spec.block_state_to_global(state)

# Sum factor energies
total_energy = 0.0
for factor in factors:
    total_energy += factor.energy(global_state, block_spec)

return total_energy
```

**Key API Detail**:
- Factors expect **global state** (stacked arrays), not block state
- Must use `BlockSpec.block_state_to_global()` conversion
- Each factor computes its contribution via `.energy()` method

**Severity**: **HIGH** - Free energy estimation depends on this.

---

### ❌ PROBLEM 6: State Conversion Missing

**Your Code** (thrml_model.py:296-298):
```python
def _sample_to_state(self, sample: jnp.ndarray) -> Dict[str, int]:
    """Convert flat sample array to state dict"""
    # TODO: Implement conversion
    state = {}
    return state
```

**Issue**: You need to map sample arrays back to gene states for energy computation.

**Design Decision Needed**:
- How are samples structured? (depends on observer)
- What's the mapping from array indices to genes?
- Need consistent ordering

**Recommended Approach**:
```python
def _sample_to_state(self, sample: jnp.ndarray) -> Dict[str, int]:
    """
    Assume sample structure: [m1, m2, e1, e2]
    where genes are ordered as in self.genes
    """
    state = {}
    n = len(self.genes)

    for i, gene in enumerate(self.genes):
        state[f"{gene}_meth"] = int(sample[i])
        state[f"{gene}_expr"] = int(sample[i + n])

    return state
```

**Severity**: **MEDIUM** - Needed for free energy calculation loop.

---

## Part III: Architectural Issues

### ⚠️ ISSUE 1: Block Organization Confusion

**Your Current Design**:
```python
blocks = [
    Block([m1, m2]),  # All methylation together
    Block([e1, e2])   # All expression together
]
```

**Potential Problem**: Different genes in same block

**THRML Constraint**:
```
Block: Collection of same-type nodes sampled simultaneously (SIMD)
All nodes must share identical type
```

**Question**: Are `CategoricalNode(num_categories=3)` for different genes the same "type"?

**Answer from Docs**: **YES** - same `num_categories` = same type ✓

**Verdict**: Your block structure is **valid** for 2-gene case. But be careful when scaling to 15 genes - you might need separate blocks per gene for proper conditioning.

---

### ⚠️ ISSUE 2: Interaction Group Structure

**Your Forward Model**: M1 → E1 → E2

**Factor Design**:
1. Factor(M1, E1) - methylation affects expression
2. Factor(E1, E2) - gene 1 affects gene 2
3. Factor(M2, E2) - baseline methylation effect

**Question**: Does this correctly encode the causal structure?

**Analysis**:
- ✓ M1 → E1: Local regulation (correct)
- ✓ M2 → E2: Local regulation (correct)
- ⚠️ E1 → E2: Inter-gene regulation

**Potential Issue**: When you update E1, does it see E2's current state?

**THRML Interaction Semantics**:
```
InteractionGroup:
    head_nodes: Nodes being updated
    tail_nodes: Neighbor information sources
```

**What This Means**:
- When updating E1, it receives M1 as input (from Factor 1)
- When updating E2, it receives M2 AND E1 as input (from Factors 2 & 3)
- This is **asymmetric** (correct for causal direction!)

**Verdict**: Your factor structure **correctly encodes asymmetric causality** ✓

---

### ⚠️ ISSUE 3: Free Energy Estimation Numerical Stability

**Your Code** (thrml_model.py:224):
```python
# Numerical stability: subtract min before exp
energies_shifted = energies - jnp.min(energies)
F = -jnp.log(jnp.mean(jnp.exp(-energies_shifted))) + jnp.min(energies)
```

**Assessment**: **GOOD** - You're using log-sum-exp trick ✓

**Potential Issue**: What if all energies are very large?

**Better Approach** (from statistical mechanics):
```python
# Use scipy.special.logsumexp for numerical stability
from scipy.special import logsumexp

# F = -log(mean(exp(-E)))
#   = -log(sum(exp(-E))/N)
#   = -log(sum(exp(-E))) + log(N)
#   = -logsumexp(-E) + log(N)

N = len(energies)
F = -logsumexp(-energies) + jnp.log(N)
```

**Severity**: **LOW** - Current approach works, but upgrade for robustness.

---

## Part IV: Missing Components

### 🔴 MISSING: Data Loading

**Status**: Not implemented at all

**Required**:
```python
# core/data_loader.py

def load_ccle_data(genes: List[str]) -> Dict:
    """
    Load CCLE methylation + expression for specified genes.

    Returns:
        {
            'methylation': DataFrame [samples × genes],
            'expression': DataFrame [samples × genes],
            'cell_lines': List of cell line IDs,
            'sensitive': List of sensitive cell line indices,
            'resistant': List of resistant cell line indices
        }
    """
    pass

def discretize_values(data: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """
    Discretize continuous values to {0, 1, 2} using quantiles.

    Args:
        data: Continuous values
        n_bins: Number of discrete bins

    Returns:
        Discretized data with same shape
    """
    # Use pd.qcut or np.percentile
    pass
```

**Severity**: **CRITICAL** - No data = no demo.

**Workaround**: Use synthetic data for smoke test (acceptable).

---

### 🔴 MISSING: Inference Pipeline

**Status**: Stubbed in thrml_model.py but needs orchestration

**Required**:
```python
# scripts/02_run_inference.py

def run_inference_pipeline(
    genes: List[str],
    data: Dict,
    n_samples: int = 1000,
    output_file: str = "results/inference.pkl"
):
    """
    Run causal inference on all gene pairs.

    Process:
    1. Load data and discretize
    2. Build INDRA prior network
    3. For each gene pair:
        a. Build forward/backward models
        b. Sample from both
        c. Compute ΔF
        d. Determine direction
    4. Build networks for sensitive vs resistant
    5. Save results
    """
    pass
```

**Severity**: **CRITICAL** - This is the main execution script.

---

### 🔴 MISSING: Network Comparison

**Status**: Not implemented

**Required**:
```python
# core/validation.py

def compare_networks(
    network_sensitive: Dict,
    network_resistant: Dict,
    threshold: float = 2.0
) -> Dict:
    """
    Find significantly changed edges between networks.

    Returns:
        {
            'edge_flips': List of (g1, g2) that reversed direction,
            'edge_weakening': List of (g1, g2) with |ΔF| decreased >50%,
            'edge_strengthening': List of (g1, g2) with |ΔF| increased >50%
        }
    """
    pass
```

**Severity**: **HIGH** - Core analysis step.

---

### 🔴 MISSING: Drug Prediction Mapping

**Status**: INDRA client works but mapping logic not implemented

**Required**:
```python
def predict_drugs_from_changes(
    changed_edges: List[Tuple[str, str]],
    indra_client: IndraClient
) -> List[Dict]:
    """
    Map changed edges to drug targets.

    Strategy:
    1. For each bypassed edge (e.g., E1 -X-> E2 becomes weak):
        - Identify which gene is upregulated in resistance
        - Query INDRA for drugs that inhibit that gene
    2. Rank by mechanistic match + belief score
    """
    pass
```

**Severity**: **HIGH** - Core prediction step.

---

## Part V: Scaling Concerns for H100s

### GPU Utilization Strategy

**Your Plan**: Split 105 pairs across 2 H100s

**Assessment**: **Needs refinement**

**Issues**:
1. THRML is built on JAX - it will use ALL visible GPUs by default
2. `CUDA_VISIBLE_DEVICES` works but you need to spawn separate processes
3. Each process should handle a subset of pairs independently

**Correct Approach**:
```bash
# Option 1: Sequential (simpler, slower)
python scripts/02_run_inference.py --all-pairs

# Option 2: Parallel (faster, more complex)
# Process 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/02_run_inference.py \
    --pairs-start 0 --pairs-end 52 \
    --output results/gpu0.pkl &

# Process 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/02_run_inference.py \
    --pairs-start 53 --pairs-end 105 \
    --output results/gpu1.pkl &

# Wait for both
wait

# Merge results
python scripts/03_analyze_results.py --merge results/gpu*.pkl
```

**JAX Compilation Overhead**:
- First run will be SLOW (30-60 seconds per pair)
- JIT compilation time
- After first pair, subsequent pairs faster
- Budget extra time for this

**Memory Concerns**:
- H100 has 80GB - you're fine
- But JAX can be memory-hungry with large pytrees
- Monitor with `nvidia-smi`

---

### Realistic Timing Estimate

**Your Plan**: 3 hours for 105 pairs

**Analysis**:
- 105 pairs × 2 models = 210 sampling runs
- Each run: 1000 samples, 100 warmup, thinning=10
- Effective iterations: 100 + 1000×10 = 10,100 iterations
- JAX overhead: first-run compilation ~60 seconds
- Subsequent runs: ~10-30 seconds per model (estimated)

**Conservative Estimate**:
- First pair: 2 minutes (compilation + sampling)
- Subsequent pairs: 1 minute each
- Total: 2 + 104 = **106 minutes = 1.8 hours** (single GPU)
- With 2 GPUs: **~1 hour** (parallelized)

**Your 3-hour budget**: **Realistic** ✓ (includes debugging time)

---

## Part VI: Recommendations

### IMMEDIATE (Before Hackathon)

1. **Fix Factor Construction** (thrml_model.py:84-100)
   - Use correct `CategoricalEBMFactor` API
   - Create Blocks properly
   - Set weight matrices correctly

2. **Fix Sampler Setup** (thrml_model.py:180-195)
   - Add `CategoricalGibbsConditional` for each block
   - Build `FactorSamplingProgram` correctly
   - Implement sampling execution

3. **Fix Energy Computation** (thrml_model.py:131-141)
   - Use `factor.energy(global_state, block_spec)`
   - Implement state conversion

4. **Test Environment**
   - Run `python scripts/00_test_environment.py`
   - Fix any import errors
   - Verify INDRA API works

5. **Create Synthetic Data Fallback**
   - Generate fake methylation/expression arrays
   - Known ground truth for testing
   - Use if CCLE download fails

### HACKATHON START (Hour 0-1)

1. **Complete data_loader.py**
   - Download CCLE or use synthetic
   - Implement discretization
   - Test on 5 genes

2. **Smoke Test** (Hour 1)
   - Run inference on 2 genes, 1 pair
   - Verify ΔF values reasonable (-5 to +5)
   - Check for NaNs, crashes
   - **FIX BUGS NOW** before scaling

3. **Scale Gradually** (Hour 2-5)
   - 5 genes → 10 pairs: 30 minutes
   - 10 genes → 45 pairs: 1.5 hours
   - 15 genes → 105 pairs: 3 hours
   - Monitor memory, check for crashes

### FALLBACK STRATEGIES

If sampling is too slow:
- ✓ Reduce to 10 genes (45 pairs, ~45 min)
- ✓ Reduce samples to 500 (faster, less accurate)
- ✓ Use only sensitive OR resistant (not both)
- ✓ Show synthetic data with known truth

If validation doesn't work:
- ✓ Focus on methodology
- ✓ Show ΔF distributions make sense
- ✓ Emphasize "validatable in principle"

If no novel insights:
- ✓ Emphasize technical execution
- ✓ Show TSU advantage clearly
- ✓ Demonstrate thermodynamic computing is real

---

## Part VII: Final Verdict

### Conceptual Soundness: 9/10 ✓
Your approach maximizes THRML's actual capabilities and is scientifically defensible.

### Implementation Completeness: 3/10 ❌
Critical API misuse and missing components will prevent execution.

### Feasibility: 6/10 ⚠️
- ✓ Doable in 8 hours IF you fix implementation
- ✓ H100s are appropriate
- ⚠️ Tight timeline requires no major debugging
- ❌ Current code will crash immediately

### Winning Potential: 7/10 ✓
- ✓ Novel application (thermodynamic causal inference)
- ✓ Direct validation (IC50 ground truth)
- ✓ Honest execution (no vaporware)
- ⚠️ Depends on clean execution
- ❌ Competition unknown

---

## THE CRITICAL PATH

**DO THIS NOW** (Before hackathon):
1. Fix thrml_model.py (3 hours of focused work)
2. Write data_loader.py (1 hour)
3. Test on synthetic data (1 hour)

**DO THIS FIRST** (Hour 0):
1. Smoke test on H100s (catch JAX/GPU issues)
2. Debug any crashes
3. Establish baseline timing

**DO THIS NEXT** (Hours 1-5):
1. Run full inference (monitor closely)
2. Save intermediate results (checkpointing)
3. Have fallback ready

**DO THIS LAST** (Hours 6-8):
1. Analysis + validation
2. Figures (use templates)
3. Presentation (practice pitch)

---

## CONCLUSION

Your hackathon proposal is **conceptually excellent but technically broken in its current state**. You understand THRML's capabilities and limitations correctly - the discretization strategy is sound, the causal inference approach is valid, and the TSU advantage argument is defensible.

**However**, you have critical implementation gaps that will cause immediate crashes:
- Factor construction uses wrong API
- Samplers not configured
- Sampling execution not implemented
- Energy computation incomplete

**The good news**: All of these are fixable in ~5 hours of focused development. The bad news: trying to debug these during the hackathon will waste your GPU time and create stress.

**My recommendation**: Spend the time NOW to fix thrml_model.py properly, write data_loader.py, and test on synthetic data locally. Then you can execute confidently on the H100s without wasting precious hackathon hours on basic API bugs.

**You have a winning idea**. Don't let poor implementation lose it for you.

Fix the code, test locally, then execute flawlessly on H100s.

You got this.
