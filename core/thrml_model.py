"""
THRML Model - Energy Functions for Causal Inference
This module defines CategoricalNode-based energy functions for gene networks.

THRML 0.1.3 API Overview:
--------------------------
Nodes:
- CategoricalNode() - factory function, no args (categories determined by weight matrix)
- SpinNode() - factory function for binary variables

Blocks:
- Block([node1, node2, ...]) - groups nodes for parallel sampling
- BlockSpec(blocks) - defines structure of entire PGM

Factors (Energy Terms):
- CategoricalEBMFactor(node_groups=[block1, block2], weights=W)
  * node_groups: list of Block objects
  * weights: JAX array, shape [batch, n_cats1, n_cats2, ...]
  * energy() method computes energy for a global state

Sampling:
- BlockGibbsSpec(free_super_blocks, clamped_blocks) - defines sampling strategy
- BlockSamplingProgram(gibbs_spec, samplers, interaction_groups)
- SamplingSchedule(n_warmup, n_samples, steps_per_sample)
- sample_states(key, program, schedule, init_state_free, state_clamp, nodes_to_sample)
  * Returns: list of PyTrees with shape [n_samples, n_nodes, ...]

Samplers:
- CategoricalGibbsConditional(n_categories=3) - for categorical variables
- SpinGibbsConditional() - for spin variables

Key Pattern:
1. Create nodes with factory functions
2. Create Blocks grouping related nodes
3. Create Factors defining energies between blocks
4. Build BlockGibbsSpec and BlockSamplingProgram
5. Run sample_states() or sample_with_observation()
6. Compute energies from samples using factor.energy(global_state, block_spec)
"""

import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Optional
from scipy.special import logsumexp
import thrml
from thrml import CategoricalNode, Block, BlockSpec
from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml import BlockGibbsSpec, BlockSamplingProgram, SamplingSchedule, sample_states

class GeneNetworkModel:
    """
    Energy-based model for gene regulatory networks.

    Uses CategoricalNodes for methylation and expression (3 states each).
    Implements causal direction testing via free energy comparison.

    NOTE: This implementation currently has placeholders for proper THRML usage.
    The sample_from_model() method needs to be updated to match THRML 0.1.3 API.
    See module docstring for correct THRML patterns.
    """
    
    def __init__(self,
                 genes: List[str],
                 prior_network: Dict[Tuple[str, str], float],
                 n_states: int = 3):
        """
        Args:
            genes: List of gene symbols
            prior_network: Dict mapping (gene1, gene2) -> belief score from INDRA
            n_states: Number of discrete states (default 3: low/med/high)
        """
        self.genes = genes
        self.n_genes = len(genes)
        self.prior_network = prior_network
        self.n_states = n_states

        # Create nodes: methylation + expression for each gene
        # Note: CategoricalNode() in THRML 0.1.3 is a factory that creates node instances
        # The number of categories is determined by the weight matrix shape in factors
        self.meth_nodes = {
            gene: CategoricalNode()
            for gene in genes
        }
        self.expr_nodes = {
            gene: CategoricalNode()
            for gene in genes
        }

        # Cache for model components
        self._model_cache = {}
        
    def build_model_forward(self,
                           gene1: str,
                           gene2: str) -> Tuple[List, List[Block]]:
        """
        Build energy model for: M1 -> E1 -> E2 (with M2 -> E2)

        Creates three factors:
        1. M1 -> E1: Local methylation regulation (anti-concordance)
        2. E1 -> E2: Inter-gene expression regulation (INDRA-weighted)
        3. M2 -> E2: Local methylation regulation for gene2

        Returns:
            factors: List of CategoricalEBMFactor objects
            blocks: List of Block objects for sampling [meth_block, expr_block]
        """
        # Get nodes
        m1 = self.meth_nodes[gene1]
        m2 = self.meth_nodes[gene2]
        e1 = self.expr_nodes[gene1]
        e2 = self.expr_nodes[gene2]

        # Create single-node blocks for factors (CRITICAL: must be Block objects)
        m1_block = Block([m1])
        m2_block = Block([m2])
        e1_block = Block([e1])
        e2_block = Block([e2])

        # Factor 1: M1 -> E1 (methylation suppresses expression)
        # Weight matrix shape: [1, 3, 3] = [batch, m1_states, e1_states]
        # Anti-concordance: high methylation -> low expression
        W_m1_e1 = jnp.array([[
            [ 1.0, -0.5, -1.0],  # M1=0 (low meth): favor E1=2 (high expr)
            [-0.5,  0.0, -0.5],  # M1=1 (med meth): neutral
            [-1.0, -0.5,  1.0],  # M1=2 (high meth): favor E1=0 (low expr)
        ]])

        factor_m1_e1 = CategoricalEBMFactor(
            node_groups=[m1_block, e1_block],
            weights=W_m1_e1
        )

        # Factor 2: E1 -> E2 (inter-gene expression regulation)
        # Get INDRA prior strength (extract belief score from dict)
        prior_key = (gene1, gene2)
        prior_info = self.prior_network.get(prior_key, {'belief': 0.5})
        prior_strength = prior_info['belief'] if isinstance(prior_info, dict) else prior_info

        # Concordance matrix: activation relationship
        # (assumes activation; could be made more sophisticated)
        W_e1_e2 = prior_strength * jnp.array([[
            [-1.0, -0.5,  0.0],  # E1=0: favor E2=0
            [-0.5, -1.0, -0.5],  # E1=1: favor E2=1
            [ 0.0, -0.5, -1.0],  # E1=2: favor E2=2
        ]])

        factor_e1_e2 = CategoricalEBMFactor(
            node_groups=[e1_block, e2_block],
            weights=W_e1_e2
        )

        # Factor 3: M2 -> E2 (methylation suppresses expression)
        W_m2_e2 = jnp.array([[
            [ 1.0, -0.5, -1.0],
            [-0.5,  0.0, -0.5],
            [-1.0, -0.5,  1.0],
        ]])

        factor_m2_e2 = CategoricalEBMFactor(
            node_groups=[m2_block, e2_block],
            weights=W_m2_e2
        )

        factors = [factor_m1_e1, factor_e1_e2, factor_m2_e2]

        # For sampling: group by variable type (enables parallel updates)
        meth_sampling_block = Block([m1, m2])
        expr_sampling_block = Block([e1, e2])

        blocks = [meth_sampling_block, expr_sampling_block]

        return factors, blocks
    
    def build_model_backward(self,
                            gene1: str,
                            gene2: str) -> Tuple[List, List[Block]]:
        """
        Build energy model for: M2 -> E2 -> E1 (with M1 -> E1)

        Mirror of forward model with reversed E2 -> E1 direction.

        Creates three factors:
        1. M2 -> E2: Local methylation regulation
        2. E2 -> E1: Inter-gene expression regulation (REVERSED)
        3. M1 -> E1: Local methylation regulation for gene1

        Returns:
            factors: List of CategoricalEBMFactor objects
            blocks: List of Block objects for sampling [meth_block, expr_block]
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

        # Factor 1: M2 -> E2 (same as forward)
        W_m2_e2 = jnp.array([[
            [ 1.0, -0.5, -1.0],
            [-0.5,  0.0, -0.5],
            [-1.0, -0.5,  1.0],
        ]])

        factor_m2_e2 = CategoricalEBMFactor(
            node_groups=[m2_block, e2_block],
            weights=W_m2_e2
        )

        # Factor 2: E2 -> E1 (REVERSED direction)
        prior_key = (gene2, gene1)  # NOTE: reversed
        prior_info = self.prior_network.get(prior_key, {'belief': 0.5})
        prior_strength = prior_info['belief'] if isinstance(prior_info, dict) else prior_info

        W_e2_e1 = prior_strength * jnp.array([[
            [-1.0, -0.5,  0.0],
            [-0.5, -1.0, -0.5],
            [ 0.0, -0.5, -1.0],
        ]])

        factor_e2_e1 = CategoricalEBMFactor(
            node_groups=[e2_block, e1_block],  # REVERSED
            weights=W_e2_e1
        )

        # Factor 3: M1 -> E1 (same as forward)
        W_m1_e1 = jnp.array([[
            [ 1.0, -0.5, -1.0],
            [-0.5,  0.0, -0.5],
            [-1.0, -0.5,  1.0],
        ]])

        factor_m1_e1 = CategoricalEBMFactor(
            node_groups=[m1_block, e1_block],
            weights=W_m1_e1
        )

        factors = [factor_m2_e2, factor_e2_e1, factor_m1_e1]

        # Sampling blocks (same organization as forward)
        meth_sampling_block = Block([m1, m2])
        expr_sampling_block = Block([e1, e2])

        blocks = [meth_sampling_block, expr_sampling_block]

        return factors, blocks
    
    def _get_cached_model(self, gene1: str, gene2: str, direction: str) -> Dict:
        """
        Get or create cached model components.

        Args:
            gene1, gene2: Gene symbols
            direction: 'forward' or 'backward'

        Returns:
            Dict with 'factors', 'blocks'
        """
        cache_key = (gene1, gene2, direction)

        if cache_key not in self._model_cache:
            if direction == 'forward':
                factors, blocks = self.build_model_forward(gene1, gene2)
            else:
                factors, blocks = self.build_model_backward(gene1, gene2)

            self._model_cache[cache_key] = {
                'factors': factors,
                'blocks': blocks
            }

        return self._model_cache[cache_key]

    def compute_energy(self,
                      gene1: str,
                      gene2: str,
                      direction: str,
                      state_dict: Dict[str, int]) -> float:
        """
        Compute total energy for a given state.

        Uses THRML 0.1.3 pattern:
        1. Convert state_dict to block state arrays
        2. Use block_state_to_global() from thrml.block_management
        3. Sum factor energies with factor.energy(global_state, gibbs_spec)

        Args:
            gene1, gene2: Gene symbols
            direction: 'forward' or 'backward'
            state_dict: Dict mapping variable names to states
                       e.g., {'EGFR_meth': 1, 'EGFR_expr': 2, ...}

        Returns:
            Total energy value
        """
        from thrml.block_management import block_state_to_global

        # Get cached model components
        model = self._get_cached_model(gene1, gene2, direction)
        factors = model['factors']
        blocks = model['blocks']

        # Convert state_dict to block state arrays
        meth_values = jnp.array([
            state_dict.get(f'{gene1}_meth', 0),
            state_dict.get(f'{gene2}_meth', 0)
        ], dtype=jnp.uint8)

        expr_values = jnp.array([
            state_dict.get(f'{gene1}_expr', 0),
            state_dict.get(f'{gene2}_expr', 0)
        ], dtype=jnp.uint8)

        # Create BlockGibbsSpec for energy computation
        gibbs_spec = BlockGibbsSpec(
            free_super_blocks=blocks,
            clamped_blocks=[]
        )

        # Convert to global state using block_state_to_global
        # Takes list of block states in order of free_super_blocks
        global_state = block_state_to_global(
            [meth_values, expr_values],
            gibbs_spec
        )

        # Sum energies from all factors
        total_energy = 0.0
        for factor in factors:
            total_energy += factor.energy(global_state, gibbs_spec)

        return float(total_energy)
    
    def sample_from_model(self,
                         factors: List,
                         blocks: List[Block],
                         n_samples: int = 1000,
                         n_warmup: int = 100) -> jnp.ndarray:
        """
        Sample states from model using THRML 0.1.3 block Gibbs sampling.

        Uses the verified THRML pattern:
        1. Create BlockGibbsSpec with free_super_blocks and clamped_blocks
        2. Create CategoricalGibbsConditional samplers (one per free block)
        3. Build interaction_groups from factors
        4. Build BlockSamplingProgram
        5. Run sample_states() - returns samples ONLY (not energies)
        6. Extract and concatenate samples

        Args:
            factors: List of CategoricalEBMFactor objects defining energy
            blocks: List of Block objects for parallel sampling [meth_block, expr_block]
            n_samples: Number of samples to collect
            n_warmup: Burn-in iterations

        Returns:
            Array of samples, shape [n_samples, 4] = [m1, m2, e1, e2]
        """
        # 1. Create BlockGibbsSpec
        # free_super_blocks: blocks to sample (can be single or tuple for simultaneous)
        # clamped_blocks: blocks held fixed (empty in our case)
        gibbs_spec = BlockGibbsSpec(
            free_super_blocks=blocks,  # [meth_block, expr_block]
            clamped_blocks=[]
        )

        # 2. Create samplers for each free block
        # CategoricalGibbsConditional requires n_categories parameter
        samplers = [
            CategoricalGibbsConditional(n_categories=self.n_states),  # For meth_block
            CategoricalGibbsConditional(n_categories=self.n_states),  # For expr_block
        ]

        # 3. Build interaction_groups from factors
        interaction_groups = []
        for factor in factors:
            interaction_groups.extend(factor.to_interaction_groups())

        # 4. Create BlockSamplingProgram
        program = BlockSamplingProgram(
            gibbs_spec=gibbs_spec,
            samplers=samplers,
            interaction_groups=interaction_groups
        )

        # 5. Create sampling schedule
        schedule = SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=1  # No thinning by default
        )

        # 6. Initialize state
        # Each free block needs initial state: shape [n_nodes_in_block]
        # blocks[0] = meth_block with 2 nodes (m1, m2)
        # blocks[1] = expr_block with 2 nodes (e1, e2)
        key = jax.random.PRNGKey(42)
        init_state_free = [
            jnp.zeros(2, dtype=jnp.uint8),  # meth_block: [m1, m2] = [0, 0]
            jnp.zeros(2, dtype=jnp.uint8),  # expr_block: [e1, e2] = [0, 0]
        ]
        state_clamp = []  # Empty since no clamped blocks

        # 7. Sample!
        # sample_states returns: list[PyTree] with shape [n_samples, n_nodes, ...]
        samples = sample_states(
            key=key,
            program=program,
            schedule=schedule,
            init_state_free=init_state_free,
            state_clamp=state_clamp,
            nodes_to_sample=blocks  # Observe both blocks
        )

        # samples is list of 2 arrays:
        # samples[0]: methylation samples, shape [n_samples, 2] = [n_samples, [m1, m2]]
        # samples[1]: expression samples, shape [n_samples, 2] = [n_samples, [e1, e2]]

        # 8. Concatenate to single array: [m1, m2, e1, e2] for each sample
        all_samples = jnp.concatenate([samples[0], samples[1]], axis=1)
        # Shape: [n_samples, 4]

        return all_samples

    def estimate_free_energy(self, energies: jnp.ndarray) -> float:
        """
        Estimate free energy from array of energies using logsumexp trick.

        Args:
            energies: Array of energy values

        Returns:
            Free energy F = -logsumexp(-energies) + log(N)
        """
        N = len(energies)
        F = -logsumexp(-energies) + jnp.log(N)
        return float(F)
    
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

        # Free energy via log-sum-exp (numerically stable)
        N = len(energies)
        F = -logsumexp(-energies) + jnp.log(N)

        return float(F)
    
    def test_causal_direction(self,
                             gene1: str,
                             gene2: str,
                             data: Dict[str, jnp.ndarray],
                             n_samples: int = 1000) -> Dict:
        """
        Test causal direction between two genes.

        Implements free energy comparison:
        - Build forward model (G1 -> G2) and backward model (G2 -> G1)
        - Sample from both models
        - Compute free energies F_forward and F_backward
        - Lower free energy indicates better model fit

        Args:
            gene1: First gene symbol
            gene2: Second gene symbol
            data: Observed methylation + expression data (not used in current implementation)
            n_samples: Number of samples for free energy estimation

        Returns:
            Dict with 'direction', 'delta_F', 'confidence', etc.
        """
        # Build models
        factors_fwd, blocks_fwd = self.build_model_forward(gene1, gene2)
        factors_bwd, blocks_bwd = self.build_model_backward(gene1, gene2)

        # Sample from both
        samples_fwd = self.sample_from_model(factors_fwd, blocks_fwd, n_samples)
        samples_bwd = self.sample_from_model(factors_bwd, blocks_bwd, n_samples)

        # Compute free energies
        F_fwd = self.compute_free_energy(gene1, gene2, 'forward', samples_fwd)
        F_bwd = self.compute_free_energy(gene1, gene2, 'backward', samples_bwd)

        # Discrimination
        delta_F = F_bwd - F_fwd

        # Determine direction (threshold = 1.0 k_B T)
        threshold = 1.0
        if delta_F > threshold:
            direction = f"{gene1} -> {gene2}"
        elif delta_F < -threshold:
            direction = f"{gene2} -> {gene1}"
        else:
            direction = "undecided"

        # Confidence estimation (simplified)
        confidence = abs(delta_F) / (1.0 + abs(delta_F))

        return {
            'gene1': gene1,
            'gene2': gene2,
            'direction': direction,
            'delta_F': float(delta_F),
            'F_forward': float(F_fwd),
            'F_backward': float(F_bwd),
            'confidence': float(confidence),
            'n_samples': n_samples
        }
    
    def _sample_to_state(self, sample: jnp.ndarray, gene1: str, gene2: str) -> Dict[str, int]:
        """
        Convert flat sample array to state dict.

        Args:
            sample: Array of length 4: [m1, m2, e1, e2]
            gene1, gene2: Gene symbols

        Returns:
            State dict with variable names
        """
        assert len(sample) == 4, f"Sample must have 4 values, got {len(sample)}"

        state = {
            f'{gene1}_meth': int(sample[0]),
            f'{gene2}_meth': int(sample[1]),
            f'{gene1}_expr': int(sample[2]),
            f'{gene2}_expr': int(sample[3])
        }

        return state


# Example usage
if __name__ == "__main__":
    # Test with small network
    genes = ["EGFR", "KRAS", "BRAF"]
    
    # Mock INDRA priors
    priors = {
        ("EGFR", "KRAS"): 0.9,
        ("KRAS", "BRAF"): 0.8,
    }
    
    model = GeneNetworkModel(genes, priors)
    
    # Mock data
    data = {
        'EGFR_meth': jnp.array([1, 1, 2, 0]),  # 4 samples
        'EGFR_expr': jnp.array([1, 2, 2, 0]),
        'KRAS_meth': jnp.array([0, 1, 1, 0]),
        'KRAS_expr': jnp.array([0, 1, 2, 0]),
    }
    
    # Test causal direction
    result = model.test_causal_direction("EGFR", "KRAS", data, n_samples=100)
    print(f"Direction: {result['direction']}")
    print(f"ΔF = {result['delta_F']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
