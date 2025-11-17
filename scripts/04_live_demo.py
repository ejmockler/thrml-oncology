#!/usr/bin/env python3
"""
Live Thermodynamic Computing Demo
Real-time visualization of thermodynamic causal inference in action.
"""

import sys
import argparse
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thrml_model import GeneNetworkModel
from core.data_loader import generate_synthetic_data, prepare_model_input
from core.indra_client import IndraClient
from core.thermodynamic_visualizer import ThermodynamicVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_live_inference_demo(
    gene1: str,
    gene2: str,
    n_samples: int = 1000,
    n_warmup: int = 100,
    output_dir: str = 'results/live_demo',
    show_realtime: bool = False
):
    """
    Run live thermodynamic inference demo with real-time visualization.

    This demonstrates the complete thermodynamic computing pipeline:
    1. Energy landscape construction
    2. Block Gibbs sampling (simulating TSU hardware)
    3. Free energy estimation
    4. Causal direction discrimination

    Args:
        gene1: First gene (e.g., 'EGFR')
        gene2: Second gene (e.g., 'KRAS')
        n_samples: Number of samples to generate
        n_warmup: Burn-in iterations
        output_dir: Where to save visualizations
        show_realtime: Show animated sampling (slow, for presentations)
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("THERMODYNAMIC COMPUTING DEMO: LIVE CAUSAL INFERENCE")
    logger.info("=" * 70)
    logger.info(f"Gene pair: {gene1} → {gene2}")
    logger.info(f"Samples: {n_samples} (warmup: {n_warmup})")
    logger.info("")

    # ========== STEP 1: Generate synthetic data ==========
    logger.info("STEP 1: Generating synthetic biological data")
    logger.info("-" * 70)

    genes = [gene1, gene2]
    data = generate_synthetic_data(
        genes,
        n_sensitive=25,
        n_resistant=25,
        seed=42
    )

    model_input = prepare_model_input(data, genes, discretize=True)
    meth_data = model_input['methylation_discrete']
    expr_data = model_input['expression_discrete']

    logger.info(f"✓ Generated data for {len(genes)} genes")
    logger.info(f"  - Methylation states: {meth_data.shape}")
    logger.info(f"  - Expression states: {expr_data.shape}")
    logger.info("")

    # ========== STEP 2: Query INDRA for biological priors ==========
    logger.info("STEP 2: Querying INDRA biological knowledge base")
    logger.info("-" * 70)

    indra_client = IndraClient()

    try:
        prior_network = indra_client.build_prior_network(genes)
        if (gene1, gene2) in prior_network:
            prior_belief = prior_network[(gene1, gene2)]['belief']
            logger.info(f"✓ INDRA prior: {gene1} → {gene2} (belief: {prior_belief:.3f})")
        else:
            logger.info(f"  No prior knowledge for {gene1} → {gene2}")
            prior_belief = 0.5
    except Exception as e:
        logger.warning(f"INDRA query failed: {e}")
        prior_belief = 0.5

    logger.info("")

    # ========== STEP 3: Build thermodynamic models ==========
    logger.info("STEP 3: Building energy-based models")
    logger.info("-" * 70)

    model = GeneNetworkModel(genes, prior_network)

    # Forward model: gene1 → gene2
    logger.info(f"Building FORWARD model: {gene1} → {gene2}")
    factors_fwd, blocks_fwd = model.build_model_forward(gene1, gene2)
    logger.info(f"  ✓ Created {len(factors_fwd)} energy factors")
    logger.info(f"  ✓ Created {len(blocks_fwd)} sampling blocks")

    # Backward model: gene2 → gene1
    logger.info(f"Building BACKWARD model: {gene2} → {gene1}")
    factors_bwd, blocks_bwd = model.build_model_backward(gene1, gene2)
    logger.info(f"  ✓ Created {len(factors_bwd)} energy factors")
    logger.info(f"  ✓ Created {len(blocks_bwd)} sampling blocks")
    logger.info("")

    # ========== STEP 4: Initialize visualizer ==========
    logger.info("STEP 4: Initializing thermodynamic visualizer")
    logger.info("-" * 70)

    visualizer = ThermodynamicVisualizer(output_dir=str(output_path))

    # Create energy landscape visualizations
    logger.info("Creating energy landscape visualizations...")
    visualizer.visualize_energy_landscape(model, gene1, gene2, direction='forward')
    visualizer.visualize_energy_landscape(model, gene1, gene2, direction='backward')
    logger.info(f"  ✓ Saved energy landscapes to {output_path}")
    logger.info("")

    # ========== STEP 5: Run Block Gibbs sampling (simulating TSU) ==========
    logger.info("STEP 5: Running Block Gibbs sampling (TSU simulation)")
    logger.info("-" * 70)
    logger.info("This simulates the thermodynamic computing hardware:")
    logger.info("  - Each sample = parallel stochastic resistor network update")
    logger.info("  - Block updates = synchronized pdit oscillations")
    logger.info("  - Thermal noise = physical temperature of TSU circuits")
    logger.info("")

    # Run forward model sampling
    logger.info(f"Sampling FORWARD model ({n_samples} samples)...")
    start_time = time.time()

    # THRML 0.1.3: sample_from_model returns samples only (not energies)
    samples_fwd = model.sample_from_model(
        factors_fwd,
        blocks_fwd,
        n_samples=n_samples,
        n_warmup=n_warmup
    )

    time_fwd = time.time() - start_time
    logger.info(f"  ✓ Generated {len(samples_fwd)} samples in {time_fwd:.2f}s")

    # Compute energies separately for each sample
    logger.info("  Computing energies for forward samples...")
    energies_fwd = []
    for sample in samples_fwd:
        state_dict = model._sample_to_state(sample, gene1, gene2)
        energy = model.compute_energy(gene1, gene2, 'forward', state_dict)
        energies_fwd.append(energy)
    energies_fwd = np.array(energies_fwd)

    logger.info(f"  ✓ Mean energy: {np.mean(energies_fwd):.2f}")
    logger.info(f"  ✓ Energy std: {np.std(energies_fwd):.2f}")

    # Run backward model sampling
    logger.info(f"Sampling BACKWARD model ({n_samples} samples)...")
    start_time = time.time()

    # THRML 0.1.3: sample_from_model returns samples only (not energies)
    samples_bwd = model.sample_from_model(
        factors_bwd,
        blocks_bwd,
        n_samples=n_samples,
        n_warmup=n_warmup
    )

    time_bwd = time.time() - start_time
    logger.info(f"  ✓ Generated {len(samples_bwd)} samples in {time_bwd:.2f}s")

    # Compute energies separately for each sample
    logger.info("  Computing energies for backward samples...")
    energies_bwd = []
    for sample in samples_bwd:
        state_dict = model._sample_to_state(sample, gene1, gene2)
        energy = model.compute_energy(gene1, gene2, 'backward', state_dict)
        energies_bwd.append(energy)
    energies_bwd = np.array(energies_bwd)

    logger.info(f"  ✓ Mean energy: {np.mean(energies_bwd):.2f}")
    logger.info(f"  ✓ Energy std: {np.std(energies_bwd):.2f}")
    logger.info("")

    # ========== STEP 6: Visualize sampling dynamics ==========
    logger.info("STEP 6: Visualizing sampling dynamics")
    logger.info("-" * 70)

    visualizer.visualize_sampling_dynamics(samples_fwd, gene1, gene2,
                                          model_type='forward')
    visualizer.visualize_sampling_dynamics(samples_bwd, gene1, gene2,
                                          model_type='backward')
    logger.info(f"  ✓ Saved sampling trajectories to {output_path}")
    logger.info("")

    # ========== STEP 7: Free energy estimation ==========
    logger.info("STEP 7: Estimating free energies")
    logger.info("-" * 70)
    logger.info("Free energy F = -log(Z) where Z = partition function")
    logger.info("Lower F = more probable model given data")
    logger.info("")

    # Compute free energies
    F_fwd = model.estimate_free_energy(energies_fwd)
    F_bwd = model.estimate_free_energy(energies_bwd)

    logger.info(f"F_forward  = {F_fwd:.4f}")
    logger.info(f"F_backward = {F_bwd:.4f}")
    logger.info(f"ΔF = F_backward - F_forward = {F_bwd - F_fwd:.4f}")
    logger.info("")

    # ========== STEP 8: Causal direction discrimination ==========
    logger.info("STEP 8: Thermodynamic causal discrimination")
    logger.info("-" * 70)

    delta_F = F_bwd - F_fwd
    threshold = 1.0  # 1 k_B T

    if delta_F > threshold:
        direction = f"{gene1} → {gene2}"
        confidence = "HIGH" if abs(delta_F) > 2.0 else "MEDIUM"
    elif delta_F < -threshold:
        direction = f"{gene2} → {gene1}"
        confidence = "HIGH" if abs(delta_F) > 2.0 else "MEDIUM"
    else:
        direction = "UNDECIDED"
        confidence = "LOW"

    logger.info(f"Discrimination threshold: {threshold:.1f} k_B T")
    logger.info(f"")
    logger.info(f"  🎯 CAUSAL DIRECTION: {direction}")
    logger.info(f"  📊 CONFIDENCE: {confidence}")
    logger.info(f"  ⚡ ΔF = {delta_F:.4f} k_B T")
    logger.info("")

    if data.get('ground_truth'):
        true_direction = data['ground_truth'].get((gene1, gene2), 'unknown')
        if true_direction != 'unknown':
            correct = (direction.replace(' ', '').replace('→', '_to_') ==
                      true_direction.replace(' ', '').replace('→', '_to_'))
            logger.info(f"  ✓ Ground truth: {true_direction}")
            logger.info(f"  ✓ Prediction: {'CORRECT' if correct else 'INCORRECT'}")
            logger.info("")

    # ========== STEP 9: Create comprehensive visualization ==========
    logger.info("STEP 9: Creating comprehensive demo dashboard")
    logger.info("-" * 70)

    result = {
        'gene1': gene1,
        'gene2': gene2,
        'delta_F': delta_F,
        'F_forward': F_fwd,
        'F_backward': F_bwd,
        'direction': direction,
        'confidence': confidence,
        'n_samples': n_samples
    }

    # Create free energy discrimination plot
    visualizer.visualize_free_energy_discrimination(result)

    # Create TSU hardware mapping
    visualizer.visualize_tsu_hardware_mapping(gene1, gene2)

    # Create comprehensive dashboard
    visualizer.create_live_demo_dashboard(
        model, gene1, gene2,
        samples_fwd, samples_bwd,
        result
    )

    logger.info(f"  ✓ Saved comprehensive dashboard to {output_path}")
    logger.info("")

    # ========== STEP 10: TSU hardware advantage analysis ==========
    logger.info("STEP 10: TSU hardware advantage")
    logger.info("-" * 70)

    # GPU metrics (H100)
    gpu_time = time_fwd + time_bwd  # Actual measured time
    gpu_power = 500  # Watts (H100 TDP)
    gpu_energy = (gpu_power * gpu_time) / 3600  # Watt-hours
    gpu_cost = gpu_energy * 0.30  # $0.30/kWh

    # TSU projected metrics (from Extropic whitepaper)
    tsu_speedup = 600  # 600× faster for block Gibbs
    tsu_time = gpu_time / tsu_speedup
    tsu_power = 5  # Watts (projected)
    tsu_energy = (tsu_power * tsu_time) / 3600
    tsu_cost = tsu_energy * 0.30

    logger.info("Current GPU Performance (2× H100):")
    logger.info(f"  - Time: {gpu_time:.2f} seconds")
    logger.info(f"  - Power: {gpu_power}W")
    logger.info(f"  - Energy: {gpu_energy:.4f} Wh")
    logger.info(f"  - Cost: ${gpu_cost:.4f}")
    logger.info("")
    logger.info("Projected TSU Performance:")
    logger.info(f"  - Time: {tsu_time:.4f} seconds ({tsu_speedup}× faster)")
    logger.info(f"  - Power: {tsu_power}W ({gpu_power/tsu_power:.0f}× more efficient)")
    logger.info(f"  - Energy: {tsu_energy:.6f} Wh")
    logger.info(f"  - Cost: ${tsu_cost:.6f} ({gpu_cost/tsu_cost:.0f}× cheaper)")
    logger.info("")
    logger.info("Why TSU is faster:")
    logger.info("  1. Block Gibbs = native pdit operation (no compilation)")
    logger.info("  2. Parallel sampling via stochastic resistor networks")
    logger.info("  3. Physical temperature provides thermal noise (free)")
    logger.info("  4. No memory bandwidth bottleneck (local connectivity)")
    logger.info("")

    # ========== FINAL SUMMARY ==========
    logger.info("=" * 70)
    logger.info("DEMO COMPLETE: THERMODYNAMIC CAUSAL INFERENCE")
    logger.info("=" * 70)
    logger.info(f"Gene pair: {gene1} ↔ {gene2}")
    logger.info(f"Inferred direction: {direction}")
    logger.info(f"Confidence: {confidence} (ΔF = {delta_F:.4f})")
    logger.info("")
    logger.info("Generated visualizations:")
    logger.info(f"  - {output_path}/energy_landscape_forward.png")
    logger.info(f"  - {output_path}/energy_landscape_backward.png")
    logger.info(f"  - {output_path}/sampling_dynamics_forward.png")
    logger.info(f"  - {output_path}/sampling_dynamics_backward.png")
    logger.info(f"  - {output_path}/free_energy_discrimination.png")
    logger.info(f"  - {output_path}/tsu_hardware_mapping.png")
    logger.info(f"  - {output_path}/live_demo_dashboard.png")
    logger.info("")
    logger.info("This demo demonstrates:")
    logger.info("  ✓ Energy-based causal inference")
    logger.info("  ✓ Thermodynamic computing on THRML")
    logger.info("  ✓ Block Gibbs sampling (TSU-native)")
    logger.info("  ✓ Free energy discrimination (ΔF)")
    logger.info("  ✓ Hardware mapping to pdit circuits")
    logger.info("  ✓ 600× speedup potential on TSU hardware")
    logger.info("")
    logger.info("Next: Scale to full gene networks and validate with IC50 data")
    logger.info("=" * 70)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Live thermodynamic computing demo'
    )

    parser.add_argument('--gene1', type=str, default='EGFR',
                       help='First gene (default: EGFR)')
    parser.add_argument('--gene2', type=str, default='KRAS',
                       help='Second gene (default: KRAS)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples (default: 1000)')
    parser.add_argument('--warmup', type=int, default=100,
                       help='Warmup iterations (default: 100)')
    parser.add_argument('--output-dir', type=str, default='results/live_demo',
                       help='Output directory (default: results/live_demo)')
    parser.add_argument('--show-realtime', action='store_true',
                       help='Show animated sampling (slow)')

    args = parser.parse_args()

    result = run_live_inference_demo(
        gene1=args.gene1,
        gene2=args.gene2,
        n_samples=args.samples,
        n_warmup=args.warmup,
        output_dir=args.output_dir,
        show_realtime=args.show_realtime
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
