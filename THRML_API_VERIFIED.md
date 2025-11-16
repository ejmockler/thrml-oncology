# THRML 0.1.3 API - VERIFIED FROM INSTALLED PACKAGE

**Source**: Extracted from installed package signatures and docstrings
**Date**: 2025-11-16
**Status**: ✅ VERIFIED

---

## Core API Signatures

### 1. Nodes

```python
CategoricalNode(*args, **kwargs)
```
- **Factory function** - no required arguments
- Creates a node representing discrete states (0, K]
- Number of categories determined by weight matrix shape

### 2. Blocks

```python
Block(nodes: list)
```
- Groups nodes for parallel sampling
- Example: `Block([node1, node2, node3])`

###3. BlockGibbsSpec

```python
BlockGibbsSpec(
    free_super_blocks: Sequence[SuperBlock],  # SuperBlock = Block | tuple[Block, ...]
    clamped_blocks: list[Block],
    node_shape_dtypes: Mapping = DEFAULT_NODE_SHAPE_DTYPES
)
```
- Defines sampling strategy (which blocks to sample, which to clamp)
- `free_super_blocks`: Blocks to sample (can be single or tuple for same-time sampling)
- `clamped_blocks`: Blocks held fixed during sampling

### 4. Factors

```python
CategoricalEBMFactor(
    node_groups: list[Block],  # NOT categorical_node_groups!
    weights: jax.Array
)
```
- **Key**: Parameter is `node_groups` NOT `categorical_node_groups`
- `weights`: Shape `[batch, n_cats1, n_cats2, ...]`
- Implements energy: `E = -sum(W[c1, c2, ...] * interactions)`

### 5. Samplers

```python
CategoricalGibbsConditional(n_categories: int)
```
- `n_categories`: Number of discrete states (e.g., 3 for low/med/high)
- Performs Gibbs update for categorical variables

### 6. Sampling Schedule

```python
SamplingSchedule(
    n_warmup: int,
    n_samples: int,
    steps_per_sample: int
)
```
- `n_warmup`: Burn-in iterations
- `n_samples`: Number of samples to collect
- `steps_per_sample`: Thinning factor

### 7. Sample States

```python
sample_states(
    key: PRNGKey,
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state_free: list[PyTree],
    state_clamp: list[PyTree],
    nodes_to_sample: list[Block]
) -> list[PyTree]
```
- **Returns**: List of PyTrees with shape `[n_samples, n_nodes, ...]`
- **Does NOT return energies** - only samples
- Energy must be computed separately using `factor.energy(state, spec)`

---

## Complete Working Pattern

```python
import jax
import jax.numpy as jnp
from thrml import (
    CategoricalNode, Block, BlockGibbsSpec,
    BlockSamplingProgram, SamplingSchedule, sample_states
)
from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional

# 1. Create nodes (factory pattern - no args)
m1 = CategoricalNode()
m2 = CategoricalNode()
e1 = CategoricalNode()
e2 = CategoricalNode()

# 2. Create blocks for factors
m1_block = Block([m1])
m2_block = Block([m2])
e1_block = Block([e1])
e2_block = Block([e2])

# 3. Create factor with weight matrix
# Shape: [batch=1, n_states_m1=3, n_states_e1=3]
W = jnp.array([[
    [ 1.0, -0.5, -1.0],  # m1=0 (low)
    [-0.5,  0.0, -0.5],  # m1=1 (med)
    [-1.0, -0.5,  1.0],  # m1=2 (high)
]])

factor = CategoricalEBMFactor(
    node_groups=[m1_block, e1_block],  # Correct parameter name!
    weights=W
)

# 4. Create sampling blocks (for parallel updates)
meth_block = Block([m1, m2])
expr_block = Block([e1, e2])

# 5. Build BlockGibbsSpec
gibbs_spec = BlockGibbsSpec(
    free_super_blocks=[meth_block, expr_block],
    clamped_blocks=[]
)

# 6. Create samplers
samplers = [
    CategoricalGibbsConditional(n_categories=3),  # For meth_block
    CategoricalGibbsConditional(n_categories=3),  # For expr_block
]

# 7. Build interaction groups from factors
interaction_groups = []
for factor in [factor]:  # Add all your factors here
    interaction_groups.extend(factor.to_interaction_groups())

# 8. Create sampling program
program = BlockSamplingProgram(
    gibbs_spec=gibbs_spec,
    samplers=samplers,
    interaction_groups=interaction_groups
)

# 9. Setup sampling schedule
schedule = SamplingSchedule(
    n_warmup=100,
    n_samples=1000,
    steps_per_sample=1
)

# 10. Initialize state
key = jax.random.PRNGKey(42)
# Each free block needs initial state: shape [n_nodes_in_block]
init_state_free = [
    jnp.array([0, 0], dtype=jnp.uint8),  # meth_block: [m1, m2]
    jnp.array([0, 0], dtype=jnp.uint8),  # expr_block: [e1, e2]
]
state_clamp = []  # Empty since no clamped blocks

# 11. Sample!
samples = sample_states(
    key=key,
    program=program,
    schedule=schedule,
    init_state_free=init_state_free,
    state_clamp=state_clamp,
    nodes_to_sample=[meth_block, expr_block]
)

# samples is list of 2 arrays:
# samples[0]: methylation samples, shape [1000, 2] = [n_samples, n_nodes]
# samples[1]: expression samples, shape [1000, 2]

# 12. Compute energies separately (NOT returned by sample_states!)
from thrml.block_management import block_state_to_global

energies = []
for i in range(schedule.n_samples):
    # Reconstruct state dict for this sample
    state_dict = {
        meth_block: samples[0][i],  # [m1, m2] values for sample i
        expr_block: samples[1][i],  # [e1, e2] values for sample i
    }

    # Convert to global state
    global_state = block_state_to_global(
        [state_dict[meth_block], state_dict[expr_block]],
        gibbs_spec
    )

    # Compute energy
    energy = factor.energy(global_state, gibbs_spec)
    energies.append(energy)

energies = jnp.array(energies)
```

---

## Key Differences from Our Code

| Our Code | Correct API |
|----------|-------------|
| `sample_model(...)` returns `(samples, energies)` | No such method exists |
| Must use `sample_states(...)` | Returns ONLY samples |
| `categorical_node_groups=` | **Wrong!** Use `node_groups=` |
| `CategoricalNode(num_categories=3)` | **Wrong!** Use `CategoricalNode()` |
| Energies from sampling | Must compute separately with `factor.energy()` |

---

## What Needs Fixing in Our Code

1. **`core/thrml_model.py`**:
   - ✅ `CategoricalNode()` - already fixed
   - ✅ `node_groups=` - already fixed
   - ❌ `sample_from_model()` method - needs complete rewrite
   - ❌ Must compute energies separately after sampling

2. **`scripts/04_live_demo.py`**:
   - ❌ Lines 153-160, 171-178: calls non-existent `model.sample_model()`
   - ❌ Expects `(samples, energies)` tuple
   - ❌ Needs refactor to use `sample_states()` + separate energy computation

3. **`scripts/02_run_inference.py`**:
   - ❌ Likely has same issues
   - ❌ Needs same fixes

---

## Energy Computation Pattern

```python
# After sampling, compute energies for each sample
def compute_energy_for_sample(sample_dict, factors, block_spec):
    """Compute total energy for a single sample."""
    global_state = block_state_to_global(
        list(sample_dict.values()),
        block_spec
    )

    total_energy = 0.0
    for factor in factors:
        total_energy += factor.energy(global_state, block_spec)

    return total_energy
```

---

## Status

✅ API signatures verified from installed package
✅ Key parameters identified (`node_groups`, not `categorical_node_groups`)
✅ Sampling returns samples only (not energies)
✅ Energy computation pattern identified
❌ Need to reimplement `sample_from_model()` in `core/thrml_model.py`
❌ Need to update demo scripts

---

## References

- Installed package: `venv/lib/python3.14/site-packages/thrml/`
- Source files examined:
  - `thrml/__init__.py` - exports
  - `thrml/block_sampling.py` - sample_states
  - `thrml/models/discrete_ebm.py` - CategoricalEBMFactor, CategoricalGibbsConditional
  - `thrml/pgm.py` - CategoricalNode
