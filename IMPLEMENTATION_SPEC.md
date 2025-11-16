# THRML Implementation Specification: Fixes Grounded in Actual API

**Status**: Detailed specification for correcting all implementation errors
**Source**: THRML_COMPREHENSIVE_DOCUMENTATION.md (authoritative reference)
**Target**: thrml_model.py fixes + missing components

---

## Part I: State Representation Design

### Problem: How do we organize gene states?

**Decision**: Use separate blocks for each variable type, organized by gene

```python
# For 2 genes (G1, G2), we have 4 nodes:
# - M1: methylation of gene 1 (CategoricalNode, 3 states)
# - M2: methylation of gene 2 (CategoricalNode, 3 states
# - E1: expression of gene 1 (CategoricalNode, 3 states)
# - E2: expression of gene 2 (CategoricalNode, 3 states)

# Block organization (CRITICAL CHOICE):
# Option A: Group by type
meth_block = Block([m1, m2])  # All methylation together
expr_block = Block([e1, e2])  # All expression together

# Option B: Group by gene
gene1_block = Block([m1, e1])  # Gene 1 variables together
gene2_block = Block([m2, e2])  # Gene 2 variables together
```

**THRML Constraint from docs**:
> Block: All nodes must share identical type
> same num_categories = same type ✓

**Chosen Approach**: **Option A** (group by type)

**Rationale**:
1. CategoricalNode(num_categories=3) are all the same type ✓
2. Easier to create factors that connect M → E
3. Matches bipartite structure (methylation layer → expression layer)
4. Allows parallel Gibbs updates

**State Array Structure**:
```python
# After sampling, state will be organized as:
# meth_block state: [m1_value, m2_value]  # Each in {0, 1, 2}
# expr_block state: [e1_value, e2_value]  # Each in {0, 1, 2}

# Total state: 4 discrete values
```

---

## Part II: CategoricalEBMFactor Construction

### Reference from THRML docs (Pattern 2):

```python
# Define categorical nodes (e.g., 10 states each)
nodes = [thrml.CategoricalNode(shape=(), num_categories=10) for _ in range(N)]

# Create blocks
block = thrml.Block(nodes)

# Define pairwise factor
factor = thrml.CategoricalEBMFactor(
    categorical_node_groups=[block, block],
    weights=W  # Shape [num_pairs, 10, 10]
)
```

**Key API Details**:
1. `categorical_node_groups` is a `list[Block]`, NOT list of nodes
2. Weights shape: `[batch_size, n_states_1, n_states_2, ...]`
3. For pairwise: `[batch, n_states, n_states]`
4. Energy contribution: `E_factor = W[batch_idx, state_1, state_2]`

### Applying to Gene Networks:

**Goal**: Create factors for M1 → E1, E1 → E2, M2 → E2

#### Factor 1: M1 → E1 (local methylation regulation)

```python
# Create single-element blocks for this factor
m1_block = Block([m1])  # Single methylation node
e1_block = Block([e1])  # Single expression node

# Weight matrix encoding M1 → E1 relationship
# Shape: [1, 3, 3] = [batch=1, m1_states=3, e1_states=3]
#
# Interpretation:
# W[0, i, j] = energy when M1=i and E1=j
#
# Biological prior: high methylation → low expression
# W[0, 0, 2] should be LOW (favorable) - low meth, high expr
# W[0, 2, 0] should be LOW (favorable) - high meth, low expr
# W[0, 0, 0] should be HIGH (unfavorable) - low meth, low expr
# W[0, 2, 2] should be HIGH (unfavorable) - high meth, high expr

import jax.numpy as jnp

# Concordance matrix: diagonal favorable (i == j means concordance)
# But we want ANTI-concordance for methylation→expression
# Solution: Use negative of anti-diagonal

W_m1_e1 = jnp.array([
    [
        [ 1.0, -0.5, -1.0],  # M1=0 (low): favor E1=2 (high)
        [-0.5,  0.0, -0.5],  # M1=1 (med): neutral
        [-1.0, -0.5,  1.0],  # M1=2 (high): favor E1=0 (low)
    ]
])  # Shape [1, 3, 3]

# Alternative: Scale by INDRA prior strength
prior_strength = prior_network.get(('gene1', 'gene1_expr'), 0.5)
W_m1_e1 = prior_strength * W_m1_e1

factor_m1_e1 = CategoricalEBMFactor(
    categorical_node_groups=[m1_block, e1_block],
    weights=W_m1_e1
)
```

#### Factor 2: E1 → E2 (inter-gene expression regulation)

```python
e1_block = Block([e1])
e2_block = Block([e2])

# Biological prior from INDRA: does E1 activate or inhibit E2?
reg_type = indra_client.get_regulation_type('GENE1', 'GENE2')

if reg_type['activates'] > reg_type['inhibits']:
    # Activation: concordance matrix (same states favored)
    W_e1_e2 = jnp.array([
        [
            [-1.0, -0.5,  0.0],  # E1=0: favor E2=0
            [-0.5, -1.0, -0.5],  # E1=1: favor E2=1
            [ 0.0, -0.5, -1.0],  # E1=2: favor E2=2
        ]
    ])
else:
    # Inhibition: anti-concordance
    W_e1_e2 = jnp.array([
        [
            [ 0.0, -0.5, -1.0],  # E1=0: favor E2=2
            [-0.5,  0.0, -0.5],  # E1=1: neutral
            [-1.0, -0.5,  0.0],  # E1=2: favor E2=0
        ]
    ])

# Scale by belief score
prior_strength = reg_type.get('activates', 0.0) + reg_type.get('inhibits', 0.0)
W_e1_e2 = prior_strength * W_e1_e2

factor_e1_e2 = CategoricalEBMFactor(
    categorical_node_groups=[e1_block, e2_block],
    weights=W_e1_e2
)
```

#### Factor 3: M2 → E2 (local methylation regulation for gene 2)

```python
# Identical structure to Factor 1
m2_block = Block([m2])
e2_block = Block([e2])

W_m2_e2 = jnp.array([
    [
        [ 1.0, -0.5, -1.0],
        [-0.5,  0.0, -0.5],
        [-1.0, -0.5,  1.0],
    ]
])

factor_m2_e2 = CategoricalEBMFactor(
    categorical_node_groups=[m2_block, e2_block],
    weights=W_m2_e2
)
```

### Complete Forward Model: M1 → E1 → E2 (with M2 → E2)

```python
def build_model_forward(self, gene1: str, gene2: str) -> Tuple[List, List[Block]]:
    """
    Build energy model for: M1 -> E1 -> E2 (with M2 -> E2)

    Returns:
        factors: List of CategoricalEBMFactor objects
        blocks: List of Block objects for sampling
    """
    # Get nodes
    m1 = self.meth_nodes[gene1]
    m2 = self.meth_nodes[gene2]
    e1 = self.expr_nodes[gene1]
    e2 = self.expr_nodes[gene2]

    # Create single-node blocks for factors
    m1_block = Block([m1])
    m2_block = Block([m2])
    e1_block = Block([e1])
    e2_block = Block([e2])

    # Factor 1: M1 -> E1
    W_m1_e1 = jnp.array([[
        [ 1.0, -0.5, -1.0],
        [-0.5,  0.0, -0.5],
        [-1.0, -0.5,  1.0],
    ]])

    factor_m1_e1 = CategoricalEBMFactor(
        categorical_node_groups=[m1_block, e1_block],
        weights=W_m1_e1
    )

    # Factor 2: E1 -> E2
    # Get INDRA prior
    prior_key = (gene1, gene2)
    prior_strength = self.prior_network.get(prior_key, 0.5)

    # Assume activation for now (could query INDRA)
    W_e1_e2 = prior_strength * jnp.array([[
        [-1.0, -0.5,  0.0],
        [-0.5, -1.0, -0.5],
        [ 0.0, -0.5, -1.0],
    ]])

    factor_e1_e2 = CategoricalEBMFactor(
        categorical_node_groups=[e1_block, e2_block],
        weights=W_e1_e2
    )

    # Factor 3: M2 -> E2
    W_m2_e2 = jnp.array([[
        [ 1.0, -0.5, -1.0],
        [-0.5,  0.0, -0.5],
        [-1.0, -0.5,  1.0],
    ]])

    factor_m2_e2 = CategoricalEBMFactor(
        categorical_node_groups=[m2_block, e2_block],
        weights=W_m2_e2
    )

    factors = [factor_m1_e1, factor_e1_e2, factor_m2_e2]

    # For sampling: group by variable type
    meth_sampling_block = Block([m1, m2])
    expr_sampling_block = Block([e1, e2])

    blocks = [meth_sampling_block, expr_sampling_block]

    return factors, blocks
```

### Backward Model: M2 → E2 → E1 (with M1 → E1)

```python
def build_model_backward(self, gene1: str, gene2: str) -> Tuple[List, List[Block]]:
    """Mirror of forward model with reversed E1 ← E2 direction"""

    m1 = self.meth_nodes[gene1]
    m2 = self.meth_nodes[gene2]
    e1 = self.expr_nodes[gene1]
    e2 = self.expr_nodes[gene2]

    m1_block = Block([m1])
    m2_block = Block([m2])
    e1_block = Block([e1])
    e2_block = Block([e2])

    # Factor 1: M2 -> E2 (same as forward)
    W_m2_e2 = jnp.array([[
        [ 1.0, -0.5, -1.0],
        [-0.5,  0.0, -0.5],
        [-1.0, -0.5,  1.0],
    ]])
    factor_m2_e2 = CategoricalEBMFactor(
        categorical_node_groups=[m2_block, e2_block],
        weights=W_m2_e2
    )

    # Factor 2: E2 -> E1 (REVERSED direction)
    prior_key = (gene2, gene1)  # NOTE: reversed
    prior_strength = self.prior_network.get(prior_key, 0.5)

    W_e2_e1 = prior_strength * jnp.array([[
        [-1.0, -0.5,  0.0],
        [-0.5, -1.0, -0.5],
        [ 0.0, -0.5, -1.0],
    ]])
    factor_e2_e1 = CategoricalEBMFactor(
        categorical_node_groups=[e2_block, e1_block],  # REVERSED
        weights=W_e2_e1
    )

    # Factor 3: M1 -> E1 (same as forward)
    W_m1_e1 = jnp.array([[
        [ 1.0, -0.5, -1.0],
        [-0.5,  0.0, -0.5],
        [-1.0, -0.5,  1.0],
    ]])
    factor_m1_e1 = CategoricalEBMFactor(
        categorical_node_groups=[m1_block, e1_block],
        weights=W_m1_e1
    )

    factors = [factor_m2_e2, factor_e2_e1, factor_m1_e1]

    # Sampling blocks (same organization)
    meth_sampling_block = Block([m1, m2])
    expr_sampling_block = Block([e1, e2])
    blocks = [meth_sampling_block, expr_sampling_block]

    return factors, blocks
```

---

## Part III: Sampler Configuration

### Reference from THRML docs:

```python
sampler = {block: thrml.CategoricalGibbsConditional()}

program = thrml.FactorSamplingProgram(
    gibbs_spec=spec,
    samplers=sampler,  # dict[Block -> ConditionalSampler]
    factors=[factor]
)
```

**API Requirements**:
- Samplers is a `dict[Block, ConditionalSampler]`
- Key is the ACTUAL Block object (not string, not index)
- Value is a ConditionalSampler instance
- For categorical variables: use `CategoricalGibbsConditional()`

### Correct Implementation:

```python
from thrml.models.discrete_ebm import CategoricalGibbsConditional

def sample_from_model(self,
                     factors: List,
                     blocks: List[Block],
                     n_samples: int = 1000,
                     n_warmup: int = 100) -> jnp.ndarray:
    """
    Sample states from model using THRML block Gibbs.
    """
    from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_with_observation
    from thrml.observers import StateObserver
    from thrml.factor import FactorSamplingProgram

    # Create BlockGibbsSpec
    spec = BlockGibbsSpec(
        free_blocks=blocks,  # [meth_block, expr_block]
        sampling_order=[0, 1],  # Sample methylation first, then expression
        clamped_blocks=[]  # No clamped variables
    )

    # Create samplers for each block
    # CRITICAL: Keys must be the ACTUAL Block objects from `blocks` list
    samplers = {
        blocks[0]: CategoricalGibbsConditional(),  # Methylation block
        blocks[1]: CategoricalGibbsConditional(),  # Expression block
    }

    # Build FactorSamplingProgram
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=samplers,
        factors=factors,
        other_interaction_groups=[]
    )

    # Create sampling schedule
    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=10  # Thinning to reduce autocorrelation
    )

    # Create observer to record states
    observer = StateObserver(blocks=spec.free_blocks)

    # Initialize state (random or zeros)
    key = jax.random.PRNGKey(42)
    block_spec = program.block_spec
    initial_state = block_spec.make_empty_block_state()

    # Run sampling
    result = sample_with_observation(
        key=key,
        program=program,
        schedule=schedule,
        observer=observer,
        initial_free_state=initial_state,
        clamped_state={}  # Empty since no clamped blocks
    )

    # Extract samples
    # result is observer state containing recorded samples
    # StateObserver returns dict[Block -> Array of samples]
    samples = result

    return samples
```

**Critical Details**:
1. `samplers` keys are Block objects (same instances as in `blocks` list)
2. `CategoricalGibbsConditional()` handles softmax computation automatically
3. `sample_with_observation()` requires initial_free_state (use `make_empty_block_state()`)
4. Return value from `StateObserver` is dict mapping blocks to sample arrays

---

## Part IV: Energy Computation

### Reference from THRML docs (Part III, Section 7):

```python
class EBMFactor:
    def energy(
        self,
        global_state: Array,
        block_spec: BlockSpec
    ) -> float:
        """Evaluate factor contribution to total energy"""
```

**Key Insight**: Factors expect **global_state**, not block_state

### State Conversion Pattern:

```python
def compute_energy(self,
                  factors: List,
                  state_dict: Dict[str, int]) -> float:
    """
    Compute total energy for a given state.

    Args:
        factors: List of CategoricalEBMFactor objects
        state_dict: Dict mapping variable names to states
                   e.g., {'EGFR_meth': 1, 'EGFR_expr': 2, ...}

    Returns:
        Total energy value
    """
    # Convert state_dict to block_state format
    # block_state = dict[Block -> Array]

    # We need to know which blocks were used during model construction
    # For 2-gene model with blocks = [meth_block, expr_block]:

    # Extract values in correct order
    gene1, gene2 = self.genes[0], self.genes[1]  # Assuming 2 genes

    # Methylation block state
    meth_values = jnp.array([
        state_dict.get(f'{gene1}_meth', 0),
        state_dict.get(f'{gene2}_meth', 0)
    ])

    # Expression block state
    expr_values = jnp.array([
        state_dict.get(f'{gene1}_expr', 0),
        state_dict.get(f'{gene2}_expr', 0)
    ])

    # Create block_state dict
    # PROBLEM: We don't have Block object references here!
    # Solution: Store blocks during model construction

    # Better approach: Use program's block_spec
    # But we're computing energy BEFORE creating program...

    # SOLUTION: Create temporary BlockSpec
    from thrml.block_management import BlockSpec

    # Reconstruct blocks
    m1 = self.meth_nodes[gene1]
    m2 = self.meth_nodes[gene2]
    e1 = self.expr_nodes[gene1]
    e2 = self.expr_nodes[gene2]

    meth_block = Block([m1, m2])
    expr_block = Block([e1, e2])

    block_state = {
        meth_block: meth_values,
        expr_block: expr_values
    }

    # Create BlockSpec
    block_spec = BlockSpec([meth_block, expr_block])

    # Convert to global state
    global_state = block_spec.block_state_to_global(block_state)

    # Sum energy contributions from all factors
    total_energy = 0.0
    for factor in factors:
        total_energy += factor.energy(global_state, block_spec)

    return float(total_energy)
```

**Problem Identified**: Energy computation needs Block objects, but we're passing state_dict

**Better Design**: Store blocks with factors during model construction

### Improved Approach:

```python
class GeneNetworkModel:
    def __init__(self, genes, prior_network, n_states=3):
        # ... existing init ...

        # Cache for model components
        self._model_cache = {}

    def _get_cached_model(self, gene1, gene2, direction):
        """Get or create cached model components"""
        cache_key = (gene1, gene2, direction)

        if cache_key not in self._model_cache:
            if direction == 'forward':
                factors, blocks = self.build_model_forward(gene1, gene2)
            else:
                factors, blocks = self.build_model_backward(gene1, gene2)

            # Create BlockSpec
            block_spec = BlockSpec(blocks)

            self._model_cache[cache_key] = {
                'factors': factors,
                'blocks': blocks,
                'block_spec': block_spec
            }

        return self._model_cache[cache_key]

    def compute_energy(self,
                      gene1: str,
                      gene2: str,
                      direction: str,
                      state_dict: Dict[str, int]) -> float:
        """
        Compute energy with cached model components.
        """
        model = self._get_cached_model(gene1, gene2, direction)
        factors = model['factors']
        blocks = model['blocks']
        block_spec = model['block_spec']

        # Convert state_dict to block_state
        meth_values = jnp.array([
            state_dict.get(f'{gene1}_meth', 0),
            state_dict.get(f'{gene2}_meth', 0)
        ])

        expr_values = jnp.array([
            state_dict.get(f'{gene1}_expr', 0),
            state_dict.get(f'{gene2}_expr', 0)
        ])

        block_state = {
            blocks[0]: meth_values,  # Methylation block
            blocks[1]: expr_values   # Expression block
        }

        # Convert to global state
        global_state = block_spec.block_state_to_global(block_state)

        # Sum energies
        total_energy = 0.0
        for factor in factors:
            total_energy += factor.energy(global_state, block_spec)

        return float(total_energy)
```

---

## Part V: Sample Execution and State Extraction

### Problem: What does `sample_with_observation()` return?

**From THRML docs**:
> StateObserver returns dict mapping blocks to sample arrays

### State Observer Return Format:

```python
# After sampling:
result = sample_with_observation(..., observer=StateObserver(blocks))

# result structure (for StateObserver):
result = {
    blocks[0]: Array([samples, nodes_in_block_0]),  # Shape: [n_samples, n_nodes]
    blocks[1]: Array([samples, nodes_in_block_1])
}

# For our case:
# blocks[0] = meth_block with 2 nodes (m1, m2)
# blocks[1] = expr_block with 2 nodes (e1, e2)

# result = {
#     meth_block: Array([n_samples, 2]),  # Methylation samples
#     expr_block: Array([n_samples, 2])   # Expression samples
# }
```

### Extracting Samples:

```python
def sample_from_model(self, factors, blocks, n_samples=1000, n_warmup=100):
    """
    Sample and return in usable format.
    """
    # ... sampling code from Part III ...

    # result is dict[Block -> Array]
    result = sample_with_observation(...)

    # Extract arrays
    meth_samples = result[blocks[0]]  # Shape: [n_samples, 2]
    expr_samples = result[blocks[1]]  # Shape: [n_samples, 2]

    # Concatenate to single array
    # samples = [m1, m2, e1, e2] for each sample
    all_samples = jnp.concatenate([meth_samples, expr_samples], axis=1)
    # Shape: [n_samples, 4]

    return all_samples
```

### Converting Samples to State Dicts:

```python
def _sample_to_state(self, sample: jnp.ndarray, gene1: str, gene2: str) -> Dict[str, int]:
    """
    Convert flat sample array to state dict.

    Args:
        sample: Array of length 4: [m1, m2, e1, e2]
        gene1, gene2: Gene symbols

    Returns:
        State dict with variable names
    """
    assert len(sample) == 4, "Sample must have 4 values"

    state = {
        f'{gene1}_meth': int(sample[0]),
        f'{gene2}_meth': int(sample[1]),
        f'{gene1}_expr': int(sample[2]),
        f'{gene2}_expr': int(sample[3])
    }

    return state
```

---

## Part VI: Free Energy Estimation

### Numerical Stability Implementation:

```python
from scipy.special import logsumexp

def compute_free_energy(self,
                       gene1: str,
                       gene2: str,
                       direction: str,
                       samples: jnp.ndarray) -> float:
    """
    Estimate free energy from samples using log-sum-exp trick.

    F = -log(Z) ≈ -log(mean(exp(-E)))
      = -log(sum(exp(-E))/N)
      = -logsumexp(-E) + log(N)

    Args:
        gene1, gene2: Gene symbols
        direction: 'forward' or 'backward'
        samples: Array of samples, shape [n_samples, 4]

    Returns:
        Free energy estimate
    """
    # Compute energy for each sample
    energies = []
    for sample in samples:
        state_dict = self._sample_to_state(sample, gene1, gene2)
        energy = self.compute_energy(gene1, gene2, direction, state_dict)
        energies.append(energy)

    energies = jnp.array(energies)

    # Free energy via log-sum-exp
    N = len(energies)
    F = -logsumexp(-energies) + jnp.log(N)

    return float(F)
```

---

## Part VII: Complete Implementation Pattern

### Minimal Working Example:

```python
import jax
import jax.numpy as jnp
from thrml.pgm import CategoricalNode
from thrml.block_management import Block, BlockSpec
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_with_observation
from thrml.observers import StateObserver
from thrml.factor import FactorSamplingProgram

# 1. Create nodes
m1 = CategoricalNode(shape=(), num_categories=3)
m2 = CategoricalNode(shape=(), num_categories=3)
e1 = CategoricalNode(shape=(), num_categories=3)
e2 = CategoricalNode(shape=(), num_categories=3)

# 2. Create factors
m1_block_factor = Block([m1])
e1_block_factor = Block([e1])
W_m1_e1 = jnp.array([[[1.0, -0.5, -1.0], [-0.5, 0.0, -0.5], [-1.0, -0.5, 1.0]]])

factor = CategoricalEBMFactor(
    categorical_node_groups=[m1_block_factor, e1_block_factor],
    weights=W_m1_e1
)

# 3. Create sampling blocks (can be different from factor blocks!)
meth_block = Block([m1, m2])
expr_block = Block([e1, e2])
blocks = [meth_block, expr_block]

# 4. Create sampling specification
spec = BlockGibbsSpec(
    free_blocks=blocks,
    sampling_order=[0, 1],
    clamped_blocks=[]
)

# 5. Create samplers
samplers = {
    blocks[0]: CategoricalGibbsConditional(),
    blocks[1]: CategoricalGibbsConditional()
}

# 6. Build program
program = FactorSamplingProgram(
    gibbs_spec=spec,
    samplers=samplers,
    factors=[factor],
    other_interaction_groups=[]
)

# 7. Create schedule
schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=10)

# 8. Create observer
observer = StateObserver(blocks=spec.free_blocks)

# 9. Initialize state
key = jax.random.PRNGKey(0)
initial_state = program.block_spec.make_empty_block_state()

# 10. Sample
result = sample_with_observation(
    key=key,
    program=program,
    schedule=schedule,
    observer=observer,
    initial_free_state=initial_state,
    clamped_state={}
)

# 11. Extract samples
meth_samples = result[blocks[0]]  # Shape: [1000, 2]
expr_samples = result[blocks[1]]  # Shape: [1000, 2]
```

---

## Part VIII: Implementation Checklist

### thrml_model.py Fixes:

- [ ] Import `CategoricalGibbsConditional` from `thrml.models.discrete_ebm`
- [ ] Import `FactorSamplingProgram` from `thrml.factor`
- [ ] Import `sample_with_observation` from `thrml.block_sampling`
- [ ] Import `StateObserver` from `thrml.observers`
- [ ] Import `BlockSpec` from `thrml.block_management`

- [ ] Fix `build_model_forward()`: Create single-node blocks for factors
- [ ] Fix `build_model_forward()`: Create correct weight matrices [1, 3, 3]
- [ ] Fix `build_model_forward()`: Return multi-node blocks for sampling

- [ ] Fix `build_model_backward()`: Mirror of forward with reversed direction

- [ ] Fix `sample_from_model()`: Create BlockGibbsSpec correctly
- [ ] Fix `sample_from_model()`: Create samplers dict with Block keys
- [ ] Fix `sample_from_model()`: Build FactorSamplingProgram
- [ ] Fix `sample_from_model()`: Call sample_with_observation()
- [ ] Fix `sample_from_model()`: Extract and concatenate samples

- [ ] Fix `compute_energy()`: Use cached model components
- [ ] Fix `compute_energy()`: Convert state_dict to block_state
- [ ] Fix `compute_energy()`: Use block_spec.block_state_to_global()
- [ ] Fix `compute_energy()`: Call factor.energy() for each factor

- [ ] Fix `compute_free_energy()`: Use logsumexp for stability
- [ ] Fix `_sample_to_state()`: Implement conversion logic

### Testing Steps:

1. Test node creation:
   ```python
   nodes = [CategoricalNode(shape=(), num_categories=3) for _ in range(4)]
   assert len(nodes) == 4
   ```

2. Test block creation:
   ```python
   block = Block(nodes[:2])
   assert len(block.nodes) == 2
   ```

3. Test factor creation:
   ```python
   b1 = Block([nodes[0]])
   b2 = Block([nodes[1]])
   W = jnp.ones((1, 3, 3))
   factor = CategoricalEBMFactor(categorical_node_groups=[b1, b2], weights=W)
   assert factor is not None
   ```

4. Test sampling (minimal):
   ```python
   model = GeneNetworkModel(['G1', 'G2'], {})
   factors, blocks = model.build_model_forward('G1', 'G2')
   samples = model.sample_from_model(factors, blocks, n_samples=10, n_warmup=5)
   assert samples.shape == (10, 4)
   ```

5. Test energy computation:
   ```python
   state = {'G1_meth': 0, 'G2_meth': 1, 'G1_expr': 2, 'G2_expr': 1}
   energy = model.compute_energy('G1', 'G2', 'forward', state)
   assert not jnp.isnan(energy)
   ```

---

## Part IX: Common Pitfalls to Avoid

### ❌ WRONG: Passing nodes to CategoricalEBMFactor

```python
# WRONG!
factor = CategoricalEBMFactor(
    categorical_node_groups=[m1, e1],  # Nodes, not Blocks
    weights=W
)
```

### ✅ CORRECT: Passing Blocks

```python
# CORRECT!
m1_block = Block([m1])
e1_block = Block([e1])
factor = CategoricalEBMFactor(
    categorical_node_groups=[m1_block, e1_block],
    weights=W
)
```

### ❌ WRONG: Sampler keys as strings or indices

```python
# WRONG!
samplers = {
    0: CategoricalGibbsConditional(),  # Index
    "meth_block": CategoricalGibbsConditional()  # String
}
```

### ✅ CORRECT: Sampler keys as Block objects

```python
# CORRECT!
samplers = {
    blocks[0]: CategoricalGibbsConditional(),  # Actual Block object
    blocks[1]: CategoricalGibbsConditional()
}
```

### ❌ WRONG: Passing block_state to factor.energy()

```python
# WRONG!
block_state = {meth_block: ..., expr_block: ...}
energy = factor.energy(block_state, block_spec)  # Type error!
```

### ✅ CORRECT: Converting to global_state first

```python
# CORRECT!
block_state = {meth_block: ..., expr_block: ...}
global_state = block_spec.block_state_to_global(block_state)
energy = factor.energy(global_state, block_spec)
```

---

## Part X: Next Steps

1. **Implement fixes in thrml_model.py** (3-4 hours)
2. **Create test_thrml_model.py** with unit tests (1 hour)
3. **Create data_loader.py** with discretization (1 hour)
4. **Create inference.py** orchestration (1 hour)
5. **Smoke test on synthetic data** (30 min)

Total estimated time: **6-7 hours before hackathon**

This leaves buffer for debugging and ensures H100 time is productive.
