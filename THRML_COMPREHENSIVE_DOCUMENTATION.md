# THRML Comprehensive Technical Documentation

## Executive Summary

This document provides a complete capture of the THRML library API, mathematical formalisms, and Extropic's hardware primitives for thermodynamic probabilistic computing.

---

## Part I: Hardware Primitives (Extropic TSU Architecture)

### 1. Probabilistic Circuit Primitives

Extropic's Thermodynamic Sampling Units (TSUs) implement four fundamental probabilistic circuit primitives:

#### **pbit (Probabilistic Bit)**
- **Type**: Discrete binary sampler
- **Distribution**: Bernoulli
- **Mathematical Form**: P(x = 1) = p, P(x = 0) = 1 - p
- **Control**: Single voltage parameter programs probability p
- **Hardware Implementation**:
  - Output voltage randomly switches between high (1) and low (0) states
  - Control voltage biases the probability with sigmoidal response
  - Sampling rate: millions to hundreds of millions flips/second
  - Energy: 10,000× less than single floating-point add per flip
- **Relaxation Time**: τ₀ ranges from ~1ms to ~1ns depending on design

#### **pdit (Probabilistic Discrete)**
- **Type**: Discrete categorical sampler
- **Distribution**: Categorical (k states)
- **Mathematical Form**: P(x = i) = pᵢ for i ∈ {1, ..., k}
- **Control**: k-1 independent control parameters
- **Hardware Implementation**:
  - Output voltage jumps between k discrete levels
  - Each level represents a category state
  - Parameters program arbitrary categorical distributions
- **Use Case**: Electronic loaded dice

#### **pmode (Probabilistic Mode)**
- **Type**: Continuous Gaussian sampler
- **Distribution**: Normal/Gaussian
- **Mathematical Form**: x ~ N(μ, Σ)
- **Control**: Single parameter for simple Gaussian
- **Hardware Implementation**:
  - Generates continuous-valued samples
  - Available in 1D and 2D configurations
  - Supports programmable covariance matrices
- **Applications**: Continuous probabilistic inference

#### **pmog (Probabilistic Mixture of Gaussians)**
- **Type**: Continuous mixture sampler
- **Distribution**: Gaussian Mixture Model (GMM)
- **Mathematical Form**: P(x) = Σᵢ wᵢ N(μᵢ, Σᵢ)
- **Control**: Mixture weights, means, and covariances
- **Applications**:
  - Clustering algorithms
  - 3D rendering
  - Complex continuous distributions

### 2. TSU Architecture

#### Hardware Organization
```
TSU Platform (XTR-0):
├── CPU (conventional processor)
├── FPGA (coordination/control)
└── Two TSU daughterboard sockets
    └── Sampling cell arrays
        ├── Input: neighbor states via wires
        ├── Computation: programmable resistor network
        ├── Sampling: probabilistic circuit (pbit/pdit/pmode/pmog)
        └── Storage: memory register
```

#### Gibbs Sampling Cell Design

Each sampling cell implements the conditional update:

1. **Input Stage**: Receive neighbor variable states x_nb(i)
2. **Parameter Computation**:
   - Resistor network computes bias γᵢ = Σⱼ∈nb(i) wᵢⱼxⱼ + bᵢ
   - Output voltage represents conditional parameter
3. **Sampling Stage**:
   - Probabilistic circuit biased by γᵢ
   - Generates sample from P(xᵢ | x_nb(i))
4. **Storage Stage**: Memory register saves state for neighbor communication

#### Scalability Properties

- **Bipartite Graph**: Enables parallel updates of half the nodes
- **Local Communication**: Only physically adjacent circuits communicate
- **Arbitrary Expansion**: Grid size scales without increasing iteration time
- **Energy Efficiency**: 10,000× improvement over GPU-based sampling (simulated)

---

## Part II: Mathematical Formalisms

### 1. Energy-Based Models (EBMs)

#### Fundamental Relationship
```
P(x) ∝ exp(-E(x))
```
where:
- P(x) is probability of state x
- E(x) is energy function
- Higher energy states → lower probability

#### Factorized Energy Functions

For hardware efficiency, energy must factorize:
```
E(x) = Σᵢ ψᵢ(x_cᵢ)
```
where:
- ψᵢ are factor functions
- x_cᵢ are variable subsets (cliques)
- Factors operate on **local, nearby variables** only

### 2. Gibbs Sampling Mathematics

#### Block Gibbs Update Rule

For variable i conditioned on neighbors:
```
P(xᵢ | x_nb(i)) ∝ exp(-E(xᵢ | x_nb(i)))
```

#### Conditional Energy for Gaussian Systems

Energy function:
```
E_G(x) = ½(x - μ)ᵀ A (x - μ)
```

Expanded form:
```
E_G(x) = ½Σᵢ Aᵢᵢxᵢ² + Σⱼ>ᵢ Aᵢⱼxᵢxⱼ + Σᵢ bᵢxᵢ + C
```

Conditional for node i:
```
Eᵢ(xᵢ, x_nb(i)) = ½Aᵢᵢxᵢ² + xᵢ(Σⱼ∈nb(i) Aᵢⱼxⱼ + bᵢ)
```

Conditional mean:
```
mᵢ = -(Σⱼ∈nb(i) (Aᵢⱼ/Aᵢᵢ)xⱼ + bᵢ/Aᵢᵢ)
```

#### Ising Model Energy

For spin systems (sᵢ ∈ {-1, 1}):
```
E(s) = -β(Σᵢ bᵢsᵢ + Σ₍ᵢ,ⱼ₎ Jᵢⱼsᵢsⱼ)
```
where:
- β is inverse temperature
- bᵢ are bias fields
- Jᵢⱼ are coupling strengths

Gibbs conditional probability:
```
P(sᵢ = 1 | s_nb(i)) = σ(2γᵢ)
```
where γᵢ = β(Σⱼ∈nb(i) Jᵢⱼsⱼ + bᵢ) and σ is sigmoid

### 3. Discrete EBM Factor Energy

General discrete interaction term:
```
E_factor(s, c) = s₁ · ... · sₘ · W[c₁, ..., cₙ]
```
where:
- sᵢ are spin variables (binary)
- cᵢ are categorical variables
- W is weight tensor with shape [b, x₁, ..., xₙ]
- b is batch dimension

#### Spin-Only Interactions
```
γ = Σᵢ s₁ⁱ · ... · sₖⁱ · Wⁱ[x₁ⁱ, ..., xₘⁱ]
```

#### Categorical-Only Interactions
```
θ = Σᵢ s₁ⁱ · ... · sₖⁱ · Wⁱ[:, x₁ⁱ, ..., xₘⁱ]
```

### 4. Training: KL Divergence Gradients

For Ising models, parameter gradients estimated via:

**Weight gradients**:
```
ΔW = -β(⟨sᵢsⱼ⟩₊ - ⟨sᵢsⱼ⟩₋)
```

**Bias gradients**:
```
Δb = -β(⟨sᵢ⟩₊ - ⟨sᵢ⟩₋)
```

where:
- ⟨·⟩₊ denotes expectations with data clamped (positive phase)
- ⟨·⟩₋ denotes expectations with free sampling (negative phase)

---

## Part III: THRML Software API

### 1. Node Primitives

THRML does NOT directly expose pbit/pdit/pmode/pmog. Instead it provides software abstractions for simulation:

#### **AbstractNode**
- Base class for all PGM nodes
- Assigns shape and datatype specifications
- JAX-compatible state organization

#### **SpinNode**
- Binary-valued variables: {-1, 1}
- Software simulation of pbit behavior
- For Ising models and spin systems

#### **CategoricalNode**
- K-state discrete variables: {0, 1, ..., K-1}
- Software simulation of pdit behavior
- Supports arbitrary cardinality

### 2. Block Management

#### **Block**
```python
class Block:
    """Collection of same-type nodes sampled simultaneously (SIMD)"""
```
- All nodes must share identical type
- Enables parallel JAX processing
- Fundamental unit of block sampling

#### **BlockSpec**
```python
class BlockSpec:
    """Manages block-local ↔ global state mappings"""
```

**Key Methods**:
- `block_state_to_global()`: Block representation → global stack
- `from_global_state()`: Extract block from global representation
- `get_node_locations()`: Find nodes in global state
- `make_empty_block_state()`: Allocate initialized state
- `verify_block_state()`: Validate state compatibility

**Design Rationale**:
- Block representation: user-facing API
- Global representation: JAX optimization (stacked arrays)

#### **BlockGibbsSpec**
```python
class BlockGibbsSpec:
    free_blocks: list[Block]        # Variables to sample
    sampling_order: list[int]       # Update sequence
    clamped_blocks: list[Block]     # Fixed/observed variables
```

Introduces **SuperBlocks**: groups of blocks sampled simultaneously algorithmically but executed separately computationally.

### 3. Factors and Interactions

#### **AbstractFactor**
```python
class AbstractFactor:
    node_groups: list[Block]

    def to_interaction_groups(self) -> list[InteractionGroup]:
        """Compile undirected factor → directed interactions"""
```

Represents batches of undirected interactions between variable sets.

#### **WeightedFactor**
```python
class WeightedFactor(AbstractFactor):
    weights: Array  # Leading dim must match batch dimension
```

Adds learnable parameters to factors.

#### **InteractionGroup**
```python
class InteractionGroup:
    head_nodes: Block              # Nodes being updated
    tail_nodes: list[Block]        # Neighbor information sources
    interaction: PyTree            # Static parametric data
```

**Functional Mechanism**:
When updating `head_nodes[i]`:
1. Gather states from `tail_nodes[k][i]` for all k
2. Extract ith element from interaction PyTree arrays
3. Pass to conditional sampler

**Critical Constraint**: First dimension of interaction arrays must equal length of head_nodes.

### 4. Discrete EBM Factors

#### **DiscreteEBMFactor**
```python
class DiscreteEBMFactor:
    """Energy terms: s₁ · ... · sₘ · W[c₁, ..., cₙ]"""
    spin_node_groups: list[Block]
    categorical_node_groups: list[Block]
    weights: Array  # Shape [b, x₁, ..., xₙ]
    is_spin: dict   # Maps variables to type
```

#### **SpinEBMFactor**
Specialized for spin-only interactions.

#### **CategoricalEBMFactor**
Specialized for categorical-only interactions.

#### **SquareDiscreteEBMFactor**
Optimizes square weight tensors for performance.

**Important Constraint**: Variables cannot appear multiple times in a single interaction (violates Boltzmann assumptions).

### 5. Conditional Samplers

#### **AbstractConditionalSampler**
```python
class AbstractConditionalSampler:
    def sample(
        self,
        key: PRNGKey,
        interactions: list[PyTree],
        active_flags: Array,
        states: Array,
        sampler_state: Any
    ) -> Array:
        """Generate samples for node block"""
```

#### **AbstractParametricConditionalSampler**
```python
class AbstractParametricConditionalSampler(AbstractConditionalSampler):
    def compute_parameters(
        self,
        interactions: list[PyTree],
        active_flags: Array,
        states: Array
    ) -> PyTree:
        """Compute conditional distribution parameters"""
```

Two-step workflow:
1. Compute parameters from neighbors
2. Sample from resulting distribution

#### **BernoulliConditional**
```python
class BernoulliConditional(AbstractParametricConditionalSampler):
    """Spin-valued Bernoulli: P(s) ∝ exp(γs), s ∈ {-1, 1}"""
```

#### **SoftmaxConditional**
```python
class SoftmaxConditional(AbstractParametricConditionalSampler):
    """Categorical softmax: P(k) ∝ exp(θₖ)"""
```

#### **SpinGibbsConditional**
Gibbs updates for spin variables:
```
γ = Σᵢ s₁ⁱ · ... · sₖⁱ · Wⁱ[x₁ⁱ, ..., xₘⁱ]
```

#### **CategoricalGibbsConditional**
Gibbs updates for categorical variables:
```
θ = Σᵢ s₁ⁱ · ... · sₖⁱ · Wⁱ[:, x₁ⁱ, ..., xₘⁱ]
```

### 6. Sampling Programs

#### **BlockSamplingProgram**
```python
class BlockSamplingProgram:
    """The beating heart of THRML"""
```

Handles:
- Reindexing between block and global representations
- Slicing and padding for vectorization
- Transforming implicit specs → executable programs

**Key Methods**:
- `sample_blocks()`: One complete iteration over all blocks
- `sample_single_block()`: Update individual block
- `sample_with_observation()`: Run with observer callbacks
- `sample_states()`: Convenience wrapper for specific nodes

**Design Philosophy**:
- JAX lacks ragged arrays → use padding
- Accept runtime overhead for compile-time efficiency
- Implicit indexing requires consistent ordering

#### **FactorSamplingProgram**
```python
class FactorSamplingProgram(BlockSamplingProgram):
    """Thin wrapper: factors → interaction groups → block sampling"""

    def __init__(
        self,
        gibbs_spec: BlockGibbsSpec,
        samplers: dict[Block, ConditionalSampler],
        factors: list[AbstractFactor],
        other_interaction_groups: list[InteractionGroup] = []
    ):
```

Decomposes factors into interaction groups for block sampling.

#### **SamplingSchedule**
```python
class SamplingSchedule:
    n_warmup: int            # Burn-in iterations
    n_samples: int           # Number of samples to collect
    steps_per_sample: int    # Iterations between recordings
```

### 7. Energy-Based Models

#### **AbstractEBM**
```python
class AbstractEBM:
    def energy(
        self,
        state: PyTree,
        block_spec: BlockSpec
    ) -> float:
        """Evaluate energy function E(x)"""
```

#### **AbstractFactorizedEBM**
```python
class AbstractFactorizedEBM(AbstractEBM):
    """Energy: E(x) = Σᵢ Eⁱ(x)"""

    @property
    def factors(self) -> list[EBMFactor]:
        """Return constituent factors"""
```

#### **FactorizedEBM**
```python
class FactorizedEBM(AbstractFactorizedEBM):
    """Concrete factorized EBM from factor list"""

    def __init__(
        self,
        factors: list[EBMFactor],
        spin_node_config: dict = {},
        categorical_node_config: dict = {}
    ):
```

Configurable node shapes/dtypes for SpinNode and CategoricalNode.

#### **EBMFactor**
```python
class EBMFactor:
    def energy(
        self,
        global_state: Array,
        block_spec: BlockSpec
    ) -> float:
        """Evaluate factor contribution to total energy"""
```

### 8. Ising Models

#### **IsingEBM**
```python
class IsingEBM:
    """Ising energy: E(s) = -β(Σᵢ bᵢsᵢ + Σ₍ᵢ,ⱼ₎ Jᵢⱼsᵢsⱼ)"""
    nodes: list[SpinNode]    # Spins with biases bᵢ
    edges: list[tuple]       # Pairs with couplings Jᵢⱼ
    beta: float              # Inverse temperature
```

#### **IsingSamplingProgram**
```python
class IsingSamplingProgram(FactorSamplingProgram):
    """Specialized for Ising model sampling"""
```

#### **IsingTrainingSpec**
```python
class IsingTrainingSpec:
    """Complete specification for trainable Ising EBMs"""
```

Includes sampling programs and schedules for positive/negative phases.

#### **Key Functions**

**hinton_init()**
```python
def hinton_init(
    key: PRNGKey,
    block_spec: BlockSpec,
    biases: Array,
    beta: float
) -> PyTree:
    """Initialize via marginal: P(sᵢ = 1) = σ(βhᵢ)"""
```

**estimate_moments()**
```python
def estimate_moments(
    samples: Array,
    edges: list[tuple]
) -> tuple[Array, Array]:
    """Compute ⟨sᵢ⟩ and ⟨sᵢsⱼ⟩ from samples"""
```

**estimate_kl_grad()**
```python
def estimate_kl_grad(
    positive_samples: Array,
    negative_samples: Array,
    edges: list[tuple],
    beta: float
) -> tuple[Array, Array]:
    """
    Estimate KL divergence gradients:
    ΔW = -β(⟨sᵢsⱼ⟩₊ - ⟨sᵢsⱼ⟩₋)
    Δb = -β(⟨sᵢ⟩₊ - ⟨sᵢ⟩₋)
    """
```

### 9. Observers

#### **AbstractObserver**
```python
class AbstractObserver:
    def init(self) -> Any:
        """Initialize observer memory (default: None)"""

    def __call__(
        self,
        block_state: PyTree,
        observer_state: Any
    ) -> Any:
        """Called once per sampling iteration"""
```

Maintains arbitrary state: running averages, histograms, log-probs, etc.

#### **StateObserver**
```python
class StateObserver(AbstractObserver):
    """Records raw node states for specified blocks"""

    def __init__(self, blocks: list[Block]):
```

#### **MomentAccumulatorObserver**
```python
class MomentAccumulatorObserver(AbstractObserver):
    """Computes running sums: Σᵢ f(x₁ⁱ)f(x₂ⁱ)...f(xₙⁱ)"""
```

**Moment Specification Structure**:
1. First level: different moment types
2. Second level: individual moments within type
3. Third level: node sequences defining each moment

**Optional Transform**:
```python
f_transform: Callable  # Element-wise transformation before accumulation
```

---

## Part IV: Software Abstractions Around Sampling

### 1. Clamping

**Concept**: Fix observed/visible variables during sampling

**Implementation**: `BlockGibbsSpec.clamped_blocks`

**Usage**:
```python
spec = BlockGibbsSpec(
    free_blocks=[latent_block],
    sampling_order=[0],
    clamped_blocks=[visible_block]  # Fixed during sampling
)
```

**Training Pattern**:
- Positive phase: clamp data, sample latents
- Negative phase: sample both data and latents

### 2. Factors

**Concept**: Undirected interactions in factor graphs

**Characteristics**:
- Batch operations over parallel node groups
- Define energy contributions
- Compiled to directed InteractionGroups for sampling

**Workflow**:
```
Factor (undirected)
    ↓ to_interaction_groups()
InteractionGroup (directed)
    ↓ used by
BlockSamplingProgram
```

**Factor Types**:
- `AbstractFactor`: Base interface
- `WeightedFactor`: Parameterized interactions
- `DiscreteEBMFactor`: Product of variables × weights
- `SpinEBMFactor`: Spin-specific optimization
- `CategoricalEBMFactor`: Categorical-specific optimization

### 3. Couplings

**Concept**: Pairwise interactions between variables

**Mathematical Form**:
```
E_coupling = Σ₍ᵢ,ⱼ₎ Jᵢⱼ xᵢ xⱼ
```

**Ising Implementation**:
```python
IsingEBM(
    nodes=[s1, s2, s3],
    edges=[(s1, s2, J12), (s2, s3, J23)],  # Coupling pairs with weights
    beta=1.0
)
```

**General Discrete EBM**:
```python
SpinEBMFactor(
    spin_node_groups=[block1, block2],
    weights=J  # Coupling tensor
)
```

**InteractionGroup Representation**:
- `head_nodes`: Variables being updated
- `tail_nodes`: Coupled neighbor variables
- `interaction`: Coupling parameters (weights/biases)

### 4. Graph Coloring and Parallelism

**Bipartite Graphs**:
- Two-color graphs enable maximal parallelism
- Update all nodes of one color simultaneously
- Alternate between colors

**SuperBlocks**:
- Algorithmic: grouped for sequential updating
- Computational: may execute separately due to different types

**Sampling Order**:
```python
BlockGibbsSpec(
    free_blocks=[red_nodes, blue_nodes],
    sampling_order=[0, 1]  # Sequential: red then blue
)
```

### 5. Padding Strategy

**Problem**: JAX lacks ragged array support

**Solution**: Pad variable-size blocks to uniform size

**Trade-off**:
- Runtime overhead from unnecessary computation
- Compile-time efficiency and vectorization
- Accepted for overall performance gain

**Implementation**: Handled internally by `BlockSamplingProgram`

### 6. State Representations

#### **Block State** (User-Facing)
```python
block_state = {
    block1: array([...]),
    block2: array([...])
}
```
- Organized by Block objects
- Convenient for API users
- Used in FactorSamplingProgram construction

#### **Global State** (Internal Optimization)
```python
global_state = [
    array([...]),  # All blocks of type 1 stacked
    array([...])   # All blocks of type 2 stacked
]
```
- Stacked by PyTree structure type
- Optimized for JAX operations
- Used internally during sampling

**Conversion Functions**:
- `BlockSpec.block_state_to_global()`
- `BlockSpec.from_global_state()`

---

## Part V: Complete API Surface Summary

### Core Modules

1. **thrml.pgm**: Graphical model components
   - `AbstractNode`, `SpinNode`, `CategoricalNode`

2. **thrml.block_management**: Block organization
   - `Block`, `BlockSpec`, `BlockGibbsSpec`

3. **thrml.interaction**: Interaction specifications
   - `InteractionGroup`

4. **thrml.factor**: Factor graph components
   - `AbstractFactor`, `WeightedFactor`, `FactorSamplingProgram`

5. **thrml.conditional_samplers**: Update rules
   - `AbstractConditionalSampler`, `BernoulliConditional`, `SoftmaxConditional`

6. **thrml.block_sampling**: Sampling infrastructure
   - `BlockSamplingProgram`, `SamplingSchedule`
   - `sample_blocks()`, `sample_with_observation()`

7. **thrml.models.ebm**: Energy-based models
   - `AbstractEBM`, `FactorizedEBM`, `EBMFactor`

8. **thrml.models.discrete_ebm**: Discrete EBMs
   - `DiscreteEBMFactor`, `SpinEBMFactor`, `CategoricalEBMFactor`
   - `SpinGibbsConditional`, `CategoricalGibbsConditional`

9. **thrml.models.ising**: Ising models
   - `IsingEBM`, `IsingSamplingProgram`, `IsingTrainingSpec`
   - `hinton_init()`, `estimate_moments()`, `estimate_kl_grad()`

10. **thrml.observers**: Sampling observation
    - `AbstractObserver`, `StateObserver`, `MomentAccumulatorObserver`

### Design Patterns

1. **Inheritance Hierarchies**:
   ```
   AbstractNode → SpinNode, CategoricalNode
   AbstractFactor → WeightedFactor → DiscreteEBMFactor → SpinEBMFactor
   AbstractConditionalSampler → AbstractParametricConditionalSampler → Concrete samplers
   AbstractEBM → AbstractFactorizedEBM → FactorizedEBM
   BlockSamplingProgram → FactorSamplingProgram → IsingSamplingProgram
   ```

2. **Compilation Pattern**:
   ```
   User Specification → Compiled Program → Executable Sampling
   Factors → InteractionGroups → BlockSamplingProgram
   ```

3. **Two-Phase Training**:
   ```
   Positive Phase: Clamp data, sample latents → Moments₊
   Negative Phase: Free sampling → Moments₋
   Gradient: Δθ ∝ (Moments₊ - Moments₋)
   ```

4. **Observer Pattern**:
   ```
   Observer.init() → initial_state
   For each sample iteration:
       new_state = observer(block_state, observer_state)
   ```

---

## Part VI: Key Architectural Principles

### 1. JAX-First Design
- Functional programming paradigm
- Pytree-based state representation
- JIT compilation for performance
- Vectorized operations via padding

### 2. Hardware-Software Co-Design
- Software abstractions mirror hardware primitives
- SpinNode ↔ pbit
- CategoricalNode ↔ pdit
- Block sampling ↔ parallel TSU cells

### 3. Locality and Factorization
- Energy functions must factorize
- Interactions between nearby variables only
- Matches hardware communication constraints
- Enables distributed processing

### 4. Separation of Concerns
- Nodes: variable definitions
- Blocks: parallel update units
- Factors: energy structure
- InteractionGroups: computational dependencies
- Samplers: update rules
- Programs: execution orchestration

### 5. Performance Trade-offs
- Padding overhead for vectorization
- Implicit indexing for efficiency
- Compile-time optimization over runtime flexibility

---

## Part VII: Relationship Between Hardware and Software

### Hardware → Software Mapping

| Hardware Primitive | Software Abstraction | THRML Implementation |
|-------------------|---------------------|----------------------|
| pbit | Binary stochastic variable | `SpinNode` |
| pdit | Categorical stochastic variable | `CategoricalNode` |
| pmode | Continuous Gaussian variable | (Not directly exposed) |
| pmog | Mixture of Gaussians | (Not directly exposed) |
| Sampling Cell | Conditional update unit | `ConditionalSampler` |
| TSU Grid | Parallel sampling array | `BlockSamplingProgram` |
| Resistor Network | Parameter computation | `InteractionGroup.interaction` |
| Memory Register | State storage | `block_state` PyTree |

### Software Simulation of Hardware

THRML provides:
1. **Exact simulation** of hardware behavior for discrete variables
2. **Performance benchmarking** before hardware deployment
3. **Algorithm prototyping** for 2026-2027 hardware release
4. **Educational framework** for thermodynamic computing concepts

### Current Limitations

1. **No continuous primitives**: pmode/pmog not directly exposed in THRML
2. **Discrete-only**: Current focus on SpinNode and CategoricalNode
3. **Simulation overhead**: GPU performance competitive but not 10,000× advantage
4. **JAX constraints**: Padding required, ragged arrays unsupported

---

## Part VIII: Usage Patterns and Examples

### Pattern 1: Building an Ising Model

```python
import thrml

# Define spin nodes
spins = [thrml.SpinNode(shape=()) for _ in range(N)]

# Define couplings
edges = [(spins[i], spins[j], J_ij) for i, j in edge_list]

# Create Ising EBM
ising = thrml.IsingEBM(
    nodes=spins,
    edges=edges,
    beta=1.0  # Inverse temperature
)

# Set up sampling
spec = thrml.BlockGibbsSpec(
    free_blocks=[thrml.Block(spins)],
    sampling_order=[0]
)

sampler = {spec.free_blocks[0]: thrml.BernoulliConditional()}

program = thrml.IsingSamplingProgram(
    gibbs_spec=spec,
    samplers=sampler,
    factors=ising.factors
)

# Sample
schedule = thrml.SamplingSchedule(
    n_warmup=100,
    n_samples=1000,
    steps_per_sample=10
)

samples = thrml.sample_with_observation(
    key=jax.random.PRNGKey(0),
    program=program,
    schedule=schedule,
    observer=thrml.StateObserver([spec.free_blocks[0]])
)
```

### Pattern 2: Categorical Model with Factors

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

# Build sampling program
spec = thrml.BlockGibbsSpec(
    free_blocks=[block],
    sampling_order=[0]
)

sampler = {block: thrml.CategoricalGibbsConditional()}

program = thrml.FactorSamplingProgram(
    gibbs_spec=spec,
    samplers=sampler,
    factors=[factor]
)
```

### Pattern 3: Clamped Variables (Training)

```python
# Separate visible and latent variables
visible_nodes = [thrml.SpinNode(shape=()) for _ in range(N_vis)]
latent_nodes = [thrml.SpinNode(shape=()) for _ in range(N_lat)]

visible_block = thrml.Block(visible_nodes)
latent_block = thrml.Block(latent_nodes)

# Positive phase: clamp visible
pos_spec = thrml.BlockGibbsSpec(
    free_blocks=[latent_block],
    sampling_order=[0],
    clamped_blocks=[visible_block]  # Fixed to data
)

# Negative phase: all free
neg_spec = thrml.BlockGibbsSpec(
    free_blocks=[latent_block, visible_block],
    sampling_order=[0, 1]
)

# Create training spec
training_spec = thrml.IsingTrainingSpec(
    positive_program=pos_program,
    negative_program=neg_program,
    positive_schedule=pos_schedule,
    negative_schedule=neg_schedule
)

# Estimate gradients
grad_W, grad_b = thrml.estimate_kl_grad(
    positive_samples=pos_samples,
    negative_samples=neg_samples,
    edges=edges,
    beta=1.0
)
```

### Pattern 4: Custom Observers

```python
# Compute first and second moments
moment_observer = thrml.MomentAccumulatorObserver(
    moment_spec=[
        [[node] for node in nodes],           # First moments
        [[nodes[i], nodes[j]] for i, j in edges]  # Second moments
    ]
)

# Run sampling with observation
result = thrml.sample_with_observation(
    key=key,
    program=program,
    schedule=schedule,
    observer=moment_observer
)

# Extract accumulated moments
first_moments = result[0]   # Σᵢ xᵢ
second_moments = result[1]  # Σᵢ xᵢ xⱼ
```

---

## Part IX: Performance Characteristics

### Computational Complexity

**Per Gibbs Iteration**:
- Factor parameter computation: O(E × d) where E = edges, d = feature dimension
- Sampling: O(N) where N = number of nodes
- Total: O(E × d + N)

**Scaling**:
- Linear in number of variables (N)
- Linear in number of edges (E)
- Parallelizable across blocks

### Memory Usage

**State Storage**:
- Global state: O(N × d) where d = node dimension
- Interaction parameters: O(E × d²) for pairwise factors

**Padding Overhead**:
- Worst case: 2× memory if block sizes highly variable
- Typical: ~20-30% overhead

### JAX JIT Compilation

**First Call**:
- Compilation time: seconds to minutes for large models
- Includes tracing, optimization, and code generation

**Subsequent Calls**:
- Near-native speed
- Microseconds per iteration for moderate-size models

### GPU Performance

**Current THRML (8 GPUs)**:
- Competitive with specialized hardware for discrete models
- Batch processing enables high throughput
- Useful for prototyping and benchmarking

**Future Hardware (TSU)**:
- 10,000× energy efficiency improvement (projected)
- Direct physical sampling vs. digital simulation
- Analog compute for parameter accumulation

---

## Part X: Comparison with Traditional Approaches

### vs. Variational Inference
- **THRML**: Asymptotically exact (given infinite samples)
- **VI**: Approximate with tractable optimization
- **Trade-off**: Sample quality vs. computational cost

### vs. Exact Inference
- **THRML**: Scalable to large models via sampling
- **Exact**: Limited to small tree-width graphs
- **Trade-off**: Accuracy vs. scalability

### vs. Neural Networks
- **THRML**: Probabilistic, interpretable, uncertainty quantification
- **NN**: Deterministic, black-box, point estimates
- **Trade-off**: Principled uncertainty vs. flexibility

### vs. Other MCMC
- **THRML**: Hardware-optimized Gibbs sampling
- **Other MCMC**: Hamiltonian MC, Langevin, Metropolis-Hastings
- **Trade-off**: Hardware efficiency vs. algorithmic sophistication

---

## Part XI: Future Directions and Roadmap

### Near-Term (THRML Software)
1. Continuous variable support (pmode/pmog simulation)
2. Higher-order factor support
3. Advanced training algorithms
4. Pre-built model zoo

### Medium-Term (XTR-0 Platform)
1. Hardware-in-the-loop prototyping
2. Hybrid CPU-TSU algorithms
3. Real-world application benchmarks
4. Developer tooling and debuggers

### Long-Term (Production TSUs)
1. Full pbit/pdit/pmode/pmog hardware deployment
2. 10,000× energy efficiency realization
3. Large-scale generative AI applications
4. Thermodynamic training at scale

---

## Part XII: Key Constraints and Limitations

### Mathematical Constraints
1. **Factorization requirement**: Energy must decompose into local terms
2. **No repeated variables**: Each variable appears once per interaction
3. **Discrete focus**: Current emphasis on binary/categorical variables
4. **Bipartite assumption**: Optimal for two-colorable graphs

### Software Constraints
1. **JAX dependency**: Requires JAX ecosystem familiarity
2. **Padding overhead**: Variable block sizes incur computational cost
3. **Implicit indexing**: Requires careful ordering of blocks/interactions
4. **Limited ragged support**: Uniform shapes preferred

### Hardware Constraints
1. **Local communication**: Only nearby variables can interact directly
2. **Discrete states**: pbit/pdit for discrete; pmode/pmog for continuous
3. **Thermal noise**: Physical sampling introduces irreducible noise
4. **Relaxation time**: Finite sampling rate (MHz to GHz range)

---

## Part XIII: Glossary

**Block**: Collection of same-type nodes updated simultaneously

**Clamping**: Fixing variables to observed values during sampling

**Coupling**: Pairwise interaction between variables (e.g., Jᵢⱼsᵢsⱼ)

**Energy Function**: E(x) mapping states to scalars; lower energy = higher probability

**Factor**: Undirected interaction between variable subsets in factor graph

**Gibbs Sampling**: Iterative conditional sampling from P(xᵢ | x₋ᵢ)

**Interaction Group**: Directed computational dependency for conditional updates

**pbit**: Probabilistic bit - hardware Bernoulli sampler

**pdit**: Probabilistic discrete - hardware categorical sampler

**pmode**: Probabilistic mode - hardware Gaussian sampler

**pmog**: Probabilistic mixture of Gaussians - hardware GMM sampler

**SuperBlock**: Algorithmic grouping of blocks for sequential update

**TSU**: Thermodynamic Sampling Unit - Extropic's probabilistic computing hardware

---

## Conclusion

THRML provides a comprehensive software framework for thermodynamic probabilistic computing that:

1. **Simulates** Extropic's pbit/pdit/pmode/pmog hardware primitives
2. **Exposes** mathematically rigorous energy-based modeling
3. **Enables** efficient block Gibbs sampling via JAX
4. **Supports** custom factors, couplings, and clamping
5. **Facilitates** algorithm prototyping before hardware deployment

The library bridges theoretical probabilistic graphical models, practical machine learning applications, and future thermodynamic computing hardware through a unified, performant API built on solid mathematical foundations.
