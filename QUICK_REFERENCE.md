# THRML Model Fixes - Quick Reference Card

## Summary: 8 Critical Fixes Applied

| # | Method | Status | Lines |
|---|--------|--------|-------|
| 1 | Imports | ✅ FIXED | 9-16 |
| 2 | `build_model_forward()` | ✅ FIXED | 54-133 |
| 3 | `build_model_backward()` | ✅ FIXED | 135-211 |
| 4 | `_get_cached_model()` | ✅ ADDED | 213-241 |
| 5 | `compute_energy()` | ✅ FIXED | 243-290 |
| 6 | `sample_from_model()` | ✅ FIXED | 292-374 |
| 7 | `compute_free_energy()` | ✅ FIXED | 376-409 |
| 8 | `_sample_to_state()` | ✅ FIXED | 472-492 |

---

## Fix 1: Imports

### Before
```python
from thrml.pgm import CategoricalNode
from thrml.block_management import Block
from thrml.models.discrete_ebm import CategoricalEBMFactor
# Missing critical imports!
```

### After
```python
from scipy.special import logsumexp
from thrml.pgm import CategoricalNode
from thrml.block_management import Block, BlockSpec
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram
from thrml.block_sampling import sample_with_observation
from thrml.observers import StateObserver
```

---

## Fix 2: build_model_forward()

### Before
```python
def build_model_forward(self, gene1: str, gene2: str) -> Tuple[List, List[Block]]:
    # TODO: Implement forward causal model
    factors = []
    blocks = [Block([m1, m2]), Block([e1, e2])]
    return factors, blocks  # Empty factors!
```

### After
```python
def build_model_forward(self, gene1: str, gene2: str) -> Tuple[List, List[Block]]:
    # Create single-node blocks for factors
    m1_block = Block([m1])
    e1_block = Block([e1])
    # ... etc

    # Factor 1: M1 -> E1 (anti-concordance)
    W_m1_e1 = jnp.array([[
        [ 1.0, -0.5, -1.0],
        [-0.5,  0.0, -0.5],
        [-1.0, -0.5,  1.0],
    ]])
    factor_m1_e1 = CategoricalEBMFactor(
        categorical_node_groups=[m1_block, e1_block],
        weights=W_m1_e1
    )

    # Factor 2: E1 -> E2 (INDRA-weighted)
    # Factor 3: M2 -> E2 (anti-concordance)

    factors = [factor_m1_e1, factor_e1_e2, factor_m2_e2]
    blocks = [Block([m1, m2]), Block([e1, e2])]
    return factors, blocks
```

**Key Change**: Creates 3 factors with proper weight matrices `[1, 3, 3]`

---

## Fix 3: build_model_backward()

### Before
```python
def build_model_backward(self, gene1: str, gene2: str) -> Tuple[List, List[Block]]:
    # TODO: Implement backward causal model
    factors = []
    blocks = []
    return factors, blocks  # Empty!
```

### After
```python
def build_model_backward(self, gene1: str, gene2: str) -> Tuple[List, List[Block]]:
    # Mirror of forward with REVERSED E2 -> E1 direction
    factor_m2_e2 = CategoricalEBMFactor(...)
    factor_e2_e1 = CategoricalEBMFactor(
        categorical_node_groups=[e2_block, e1_block],  # REVERSED!
        weights=W_e2_e1
    )
    factor_m1_e1 = CategoricalEBMFactor(...)

    factors = [factor_m2_e2, factor_e2_e1, factor_m1_e1]
    blocks = [Block([m1, m2]), Block([e1, e2])]
    return factors, blocks
```

**Key Change**: Reverses E1 ↔ E2 direction

---

## Fix 4: _get_cached_model() [NEW]

### Before
*Method did not exist!*

### After
```python
def _get_cached_model(self, gene1: str, gene2: str, direction: str) -> Dict:
    cache_key = (gene1, gene2, direction)
    if cache_key not in self._model_cache:
        if direction == 'forward':
            factors, blocks = self.build_model_forward(gene1, gene2)
        else:
            factors, blocks = self.build_model_backward(gene1, gene2)
        block_spec = BlockSpec(blocks)
        self._model_cache[cache_key] = {
            'factors': factors,
            'blocks': blocks,
            'block_spec': block_spec
        }
    return self._model_cache[cache_key]
```

**Key Change**: Enables efficient energy computation with cached Block references

---

## Fix 5: compute_energy()

### Before
```python
def compute_energy(self, factors: List, state: Dict[str, jnp.ndarray]) -> float:
    # TODO: Implement energy computation
    total_energy = 0.0
    for factor in factors:
        pass  # Empty!
    return total_energy
```

### After
```python
def compute_energy(self, gene1: str, gene2: str, direction: str,
                  state_dict: Dict[str, int]) -> float:
    # Get cached model
    model = self._get_cached_model(gene1, gene2, direction)
    factors, blocks, block_spec = model['factors'], model['blocks'], model['block_spec']

    # Convert state_dict -> block_state -> global_state
    block_state = {
        blocks[0]: jnp.array([state_dict[f'{gene1}_meth'], state_dict[f'{gene2}_meth']]),
        blocks[1]: jnp.array([state_dict[f'{gene1}_expr'], state_dict[f'{gene2}_expr']])
    }
    global_state = block_spec.block_state_to_global(block_state)

    # Sum energies
    total_energy = sum(factor.energy(global_state, block_spec) for factor in factors)
    return float(total_energy)
```

**Key Changes**:
1. Proper signature with gene/direction parameters
2. State conversion pipeline
3. Actual energy computation

---

## Fix 6: sample_from_model()

### Before
```python
def sample_from_model(self, factors: List, blocks: List[Block],
                     n_samples: int = 1000, n_warmup: int = 100) -> jnp.ndarray:
    # TODO: Implement THRML sampling
    spec = BlockGibbsSpec(...)
    samplers = {}  # Empty!
    program = None  # None!
    samples = None  # None!
    return samples
```

### After
```python
def sample_from_model(self, factors: List, blocks: List[Block],
                     n_samples: int = 1000, n_warmup: int = 100) -> jnp.ndarray:
    spec = BlockGibbsSpec(
        free_blocks=blocks,
        sampling_order=[0, 1],
        clamped_blocks=[]
    )

    # CRITICAL: Keys must be Block objects!
    samplers = {
        blocks[0]: CategoricalGibbsConditional(),
        blocks[1]: CategoricalGibbsConditional(),
    }

    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=samplers,
        factors=factors,
        other_interaction_groups=[]
    )

    schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=10)
    observer = StateObserver(blocks=spec.free_blocks)
    initial_state = program.block_spec.make_empty_block_state()

    result = sample_with_observation(
        key=jax.random.PRNGKey(42),
        program=program,
        schedule=schedule,
        observer=observer,
        initial_free_state=initial_state,
        clamped_state={}
    )

    meth_samples = result[blocks[0]]
    expr_samples = result[blocks[1]]
    return jnp.concatenate([meth_samples, expr_samples], axis=1)
```

**Key Changes**: Complete THRML sampling workflow

---

## Fix 7: compute_free_energy()

### Before
```python
def compute_free_energy(self, samples: jnp.ndarray, factors: List) -> float:
    # TODO: Implement free energy estimation
    energies = jnp.array([
        self.compute_energy(factors, self._sample_to_state(s))  # Wrong signature!
        for s in samples
    ])
    F = -jnp.log(jnp.mean(jnp.exp(-energies)))  # Numerically unstable!
    return float(F)
```

### After
```python
def compute_free_energy(self, gene1: str, gene2: str, direction: str,
                       samples: jnp.ndarray) -> float:
    energies = []
    for sample in samples:
        state_dict = self._sample_to_state(sample, gene1, gene2)
        energy = self.compute_energy(gene1, gene2, direction, state_dict)
        energies.append(energy)

    energies = jnp.array(energies)
    N = len(energies)
    F = -logsumexp(-energies) + jnp.log(N)  # Numerically stable!
    return float(F)
```

**Key Changes**:
1. Proper signature with gene/direction
2. Log-sum-exp trick for stability

---

## Fix 8: _sample_to_state()

### Before
```python
def _sample_to_state(self, sample: jnp.ndarray) -> Dict[str, int]:
    # TODO: Implement conversion
    state = {}
    return state  # Empty!
```

### After
```python
def _sample_to_state(self, sample: jnp.ndarray, gene1: str, gene2: str) -> Dict[str, int]:
    assert len(sample) == 4, f"Sample must have 4 values, got {len(sample)}"
    return {
        f'{gene1}_meth': int(sample[0]),
        f'{gene2}_meth': int(sample[1]),
        f'{gene1}_expr': int(sample[2]),
        f'{gene2}_expr': int(sample[3])
    }
```

**Key Change**: Proper array-to-dict conversion

---

## Quick Test

```python
from core.thrml_model import GeneNetworkModel

# Create model
model = GeneNetworkModel(['EGFR', 'KRAS'], {('EGFR', 'KRAS'): 0.9})

# Build model
factors, blocks = model.build_model_forward('EGFR', 'KRAS')
print(f"Factors: {len(factors)}, Blocks: {len(blocks)}")  # Should print: Factors: 3, Blocks: 2

# Sample (requires THRML environment)
samples = model.sample_from_model(factors, blocks, n_samples=10, n_warmup=5)
print(f"Samples shape: {samples.shape}")  # Should print: Samples shape: (10, 4)

# Compute energy
state = {'EGFR_meth': 0, 'KRAS_meth': 1, 'EGFR_expr': 2, 'KRAS_expr': 1}
energy = model.compute_energy('EGFR', 'KRAS', 'forward', state)
print(f"Energy: {energy}")  # Should print a finite number
```

---

## Critical Patterns to Remember

### ✅ Factor Construction
```python
# Single-node blocks for factors
m1_block = Block([m1])
e1_block = Block([e1])

factor = CategoricalEBMFactor(
    categorical_node_groups=[m1_block, e1_block],  # Blocks, not nodes!
    weights=jnp.array([[[...]]])  # Shape: [1, 3, 3]
)
```

### ✅ Sampler Keys
```python
samplers = {
    blocks[0]: CategoricalGibbsConditional(),  # Block object as key!
    blocks[1]: CategoricalGibbsConditional()
}
```

### ✅ State Conversion
```python
# state_dict -> block_state -> global_state
block_state = {blocks[0]: meth_values, blocks[1]: expr_values}
global_state = block_spec.block_state_to_global(block_state)
energy = factor.energy(global_state, block_spec)
```

### ✅ Free Energy
```python
F = -logsumexp(-energies) + jnp.log(N)  # Use logsumexp!
```

---

## Files Changed

1. `core/thrml_model.py` - All methods implemented
2. `test_thrml_fixes.py` - NEW comprehensive test suite
3. `FIXES_APPLIED.md` - Complete documentation
4. `QUICK_REFERENCE.md` - This file

**Syntax validated**: ✅ `python3 -m py_compile core/thrml_model.py` passes

---

## Next Steps

1. Set up THRML environment: `pip install -r requirements.txt`
2. Run tests: `python test_thrml_fixes.py`
3. Verify all 8 tests pass
4. Build data loader for TCGA data
5. Deploy to H100s for real inference

**Ready for hackathon! 🚀**
