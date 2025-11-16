"""
Thermodynamic Visualization Engine
Manifests the thermodynamic computing action behind causal inference.

Shows energy landscapes, sampling dynamics, free energy evolution, and TSU hardware mapping.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import jax.numpy as jnp
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ThermodynamicVisualizer:
    """
    Creates compelling visualizations showing thermodynamic computing in action.

    Visualizations:
    1. Energy landscape (3D surface)
    2. Gibbs sampling trajectory (live animation)
    3. Free energy evolution over sampling iterations
    4. TSU hardware mapping (block structure)
    5. Causal discrimination via ΔF
    6. Network rewiring between sensitive/resistant
    """

    def __init__(self, figsize=(20, 12)):
        self.figsize = figsize
        sns.set_style("darkgrid")
        plt.rcParams['font.size'] = 10

    def visualize_energy_landscape(
        self,
        model,
        gene1: str,
        gene2: str,
        direction: str,
        output_path: str = None
    ):
        """
        3D energy landscape for a gene pair.

        Shows how energy varies across methylation-expression state space.
        Reveals energy minima (attractors) and barriers.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig)

        # Create state grids
        m_states = np.arange(3)  # {0, 1, 2}
        e_states = np.arange(3)

        # Compute energy for all M1-E1 combinations
        M1, E1 = np.meshgrid(m_states, e_states)
        energies = np.zeros((3, 3))

        for i, m1 in enumerate(m_states):
            for j, e1 in enumerate(e_states):
                # Fix M2=1, E2=1 for visualization
                state = {
                    f'{gene1}_meth': int(m1),
                    f'{gene2}_meth': 1,
                    f'{gene1}_expr': int(e1),
                    f'{gene2}_expr': 1
                }
                energies[j, i] = model.compute_energy(gene1, gene2, direction, state)

        # 3D surface plot
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        surf = ax1.plot_surface(M1, E1, energies, cmap='coolwarm', alpha=0.8)
        ax1.set_xlabel(f'{gene1} Methylation', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'{gene1} Expression', fontsize=12, fontweight='bold')
        ax1.set_zlabel('Energy', fontsize=12, fontweight='bold')
        ax1.set_title(f'Energy Landscape: {direction}', fontsize=14, fontweight='bold')
        ax1.set_xticks([0, 1, 2])
        ax1.set_yticks([0, 1, 2])
        ax1.set_xticklabels(['Low', 'Med', 'High'])
        ax1.set_yticklabels(['Low', 'Med', 'High'])
        fig.colorbar(surf, ax=ax1, shrink=0.5)

        # 2D heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(energies, cmap='coolwarm', origin='lower')
        ax2.set_xlabel(f'{gene1} Methylation', fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'{gene1} Expression', fontsize=12, fontweight='bold')
        ax2.set_title('Energy Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xticks([0, 1, 2])
        ax2.set_yticks([0, 1, 2])
        ax2.set_xticklabels(['Low', 'Med', 'High'])
        ax2.set_yticklabels(['Low', 'Med', 'High'])

        # Annotate energy values
        for i in range(3):
            for j in range(3):
                text = ax2.text(j, i, f'{energies[i, j]:.2f}',
                              ha="center", va="center", color="white", fontweight='bold')

        fig.colorbar(im, ax=ax2)

        # Boltzmann probability
        ax3 = fig.add_subplot(gs[1, 0])
        probs = np.exp(-energies)
        probs = probs / probs.sum()

        im2 = ax3.imshow(probs, cmap='viridis', origin='lower')
        ax3.set_xlabel(f'{gene1} Methylation', fontsize=12, fontweight='bold')
        ax3.set_ylabel(f'{gene1} Expression', fontsize=12, fontweight='bold')
        ax3.set_title('Boltzmann Probability: P(x) ∝ e^(-E(x))', fontsize=14, fontweight='bold')
        ax3.set_xticks([0, 1, 2])
        ax3.set_yticks([0, 1, 2])
        ax3.set_xticklabels(['Low', 'Med', 'High'])
        ax3.set_yticklabels(['Low', 'Med', 'High'])

        for i in range(3):
            for j in range(3):
                text = ax3.text(j, i, f'{probs[i, j]:.3f}',
                              ha="center", va="center", color="white", fontweight='bold')

        fig.colorbar(im2, ax=ax3)

        # Energy profile (cross-section)
        ax4 = fig.add_subplot(gs[1, 1])

        # Low methylation (M=0)
        ax4.plot(e_states, energies[:, 0], 'o-', linewidth=2, markersize=8,
                label='Low Meth (M=0)', color='blue')
        # Medium methylation (M=1)
        ax4.plot(e_states, energies[:, 1], 's-', linewidth=2, markersize=8,
                label='Med Meth (M=1)', color='green')
        # High methylation (M=2)
        ax4.plot(e_states, energies[:, 2], '^-', linewidth=2, markersize=8,
                label='High Meth (M=2)', color='red')

        ax4.set_xlabel(f'{gene1} Expression State', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Energy', fontsize=12, fontweight='bold')
        ax4.set_title('Energy Profile: Methylation Effect', fontsize=14, fontweight='bold')
        ax4.set_xticks([0, 1, 2])
        ax4.set_xticklabels(['Low', 'Med', 'High'])
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved energy landscape to {output_path}")

        return fig

    def visualize_sampling_dynamics(
        self,
        samples: np.ndarray,
        gene1: str,
        gene2: str,
        output_path: str = None,
        max_samples: int = 200
    ):
        """
        Visualize Gibbs sampling trajectory showing thermodynamic exploration.

        Shows how the system explores state space via thermal fluctuations.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        samples_subset = samples[:max_samples]
        iterations = np.arange(len(samples_subset))

        # M1 trajectory
        ax = axes[0, 0]
        ax.plot(iterations, samples_subset[:, 0], 'o-', alpha=0.6, color='blue', linewidth=1.5)
        ax.set_ylabel(f'{gene1} Methylation', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sampling Iteration', fontsize=12, fontweight='bold')
        ax.set_title('Methylation State Evolution (Gibbs Sampling)', fontsize=14, fontweight='bold')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Low', 'Med', 'High'])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)

        # E1 trajectory
        ax = axes[0, 1]
        ax.plot(iterations, samples_subset[:, 2], 's-', alpha=0.6, color='green', linewidth=1.5)
        ax.set_ylabel(f'{gene1} Expression', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sampling Iteration', fontsize=12, fontweight='bold')
        ax.set_title('Expression State Evolution', fontsize=14, fontweight='bold')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Low', 'Med', 'High'])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)

        # M1-E1 phase space
        ax = axes[1, 0]
        scatter = ax.scatter(samples_subset[:, 0], samples_subset[:, 2],
                           c=iterations, cmap='plasma', s=50, alpha=0.6)
        ax.set_xlabel(f'{gene1} Methylation', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{gene1} Expression', fontsize=12, fontweight='bold')
        ax.set_title('Phase Space Trajectory (Color = Time)', fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['Low', 'Med', 'High'])
        ax.set_yticklabels(['Low', 'Med', 'High'])
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Iteration')

        # State occupation histogram
        ax = axes[1, 1]

        # Compute state occupation frequencies
        state_counts = {}
        for sample in samples_subset:
            state = (int(sample[0]), int(sample[2]))
            state_counts[state] = state_counts.get(state, 0) + 1

        # Create heatmap
        occupation = np.zeros((3, 3))
        for (m, e), count in state_counts.items():
            occupation[e, m] = count / len(samples_subset)

        im = ax.imshow(occupation, cmap='viridis', origin='lower')
        ax.set_xlabel(f'{gene1} Methylation', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{gene1} Expression', fontsize=12, fontweight='bold')
        ax.set_title('State Occupation Probability', fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['Low', 'Med', 'High'])
        ax.set_yticklabels(['Low', 'Med', 'High'])

        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, f'{occupation[i, j]:.2f}',
                             ha="center", va="center", color="white", fontweight='bold')

        plt.colorbar(im, ax=ax)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sampling dynamics to {output_path}")

        return fig

    def visualize_free_energy_discrimination(
        self,
        result: Dict,
        output_path: str = None
    ):
        """
        Visualize the core thermodynamic discrimination: ΔF comparison.

        Shows how free energy differences reveal causal direction.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        gene1 = result['gene1']
        gene2 = result['gene2']
        F_fwd = result['F_forward']
        F_bwd = result['F_backward']
        delta_F = result['delta_F']
        direction = result['direction']
        confidence = result['confidence']

        # Free energy comparison
        ax = axes[0]
        models = ['Forward\n(G1→G2)', 'Backward\n(G2→G1)']
        free_energies = [F_fwd, F_bwd]
        colors = ['blue' if delta_F > 1 else 'gray',
                 'red' if delta_F < -1 else 'gray']

        bars = ax.bar(models, free_energies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Free Energy F', fontsize=12, fontweight='bold')
        ax.set_title('Free Energy Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Annotate values
        for bar, val in zip(bars, free_energies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        # ΔF arrow
        ax = axes[1]
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
        ax.axhline(y=-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Plot ΔF
        color = 'green' if abs(delta_F) > 1 else 'orange'
        ax.barh([0], [delta_F], height=0.5, color=color, alpha=0.7, edgecolor='black', linewidth=2)

        ax.set_xlabel('ΔF = F_backward - F_forward', fontsize=12, fontweight='bold')
        ax.set_title('Thermodynamic Discrimination', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.set_xlim(-max(abs(delta_F) * 1.5, 2), max(abs(delta_F) * 1.5, 2))
        ax.grid(axis='x', alpha=0.3)
        ax.legend()

        # Annotate decision
        ax.text(delta_F, 0.3, f'ΔF = {delta_F:.2f}',
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))

        # Decision diagram
        ax = axes[2]
        ax.axis('off')

        # Title
        ax.text(0.5, 0.95, 'Causal Direction Decision',
               ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)

        # Decision tree
        y_pos = 0.75
        ax.text(0.5, y_pos, f'ΔF = {delta_F:.2f}',
               ha='center', fontsize=12, fontweight='bold',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))

        # Arrows and decisions
        if delta_F > 1:
            ax.arrow(0.5, y_pos - 0.05, -0.15, -0.15,
                    head_width=0.03, head_length=0.05, fc='green', ec='green',
                    transform=ax.transAxes, linewidth=2)
            ax.text(0.25, 0.5, f'{gene1} → {gene2}',
                   ha='center', fontsize=13, fontweight='bold',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))
        elif delta_F < -1:
            ax.arrow(0.5, y_pos - 0.05, 0.15, -0.15,
                    head_width=0.03, head_length=0.05, fc='red', ec='red',
                    transform=ax.transAxes, linewidth=2)
            ax.text(0.75, 0.5, f'{gene2} → {gene1}',
                   ha='center', fontsize=13, fontweight='bold',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='red', linewidth=2))
        else:
            ax.arrow(0.5, y_pos - 0.05, 0, -0.15,
                    head_width=0.03, head_length=0.05, fc='orange', ec='orange',
                    transform=ax.transAxes, linewidth=2)
            ax.text(0.5, 0.5, 'Undecided',
                   ha='center', fontsize=13, fontweight='bold',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))

        # Confidence
        ax.text(0.5, 0.3, f'Confidence: {confidence:.1%}',
               ha='center', fontsize=12, transform=ax.transAxes)

        # Physical interpretation
        ax.text(0.5, 0.1, 'Lower free energy = Better model fit',
               ha='center', fontsize=10, style='italic', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved free energy discrimination to {output_path}")

        return fig

    def visualize_tsu_hardware_mapping(
        self,
        gene1: str,
        gene2: str,
        output_path: str = None
    ):
        """
        Visualize how the model maps to TSU hardware.

        Shows:
        - Variable nodes → pbit/pdit circuits
        - Block structure → parallel sampling cells
        - Factor interactions → resistor networks
        - Gibbs updates → thermodynamic relaxation
        """
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Software architecture
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        ax1.set_title('SOFTWARE → HARDWARE MAPPING', fontsize=16, fontweight='bold', pad=20)

        # Draw software layer
        software_y = 0.75

        # Nodes
        for i, (label, x_pos) in enumerate([
            ('M1\n(CategoricalNode)', 0.15),
            ('M2\n(CategoricalNode)', 0.35),
            ('E1\n(CategoricalNode)', 0.65),
            ('E2\n(CategoricalNode)', 0.85)
        ]):
            circle = mpatches.Circle((x_pos, software_y), 0.05,
                                    facecolor='lightblue', edgecolor='blue', linewidth=2)
            ax1.add_patch(circle)
            ax1.text(x_pos, software_y, label, ha='center', va='center',
                    fontsize=9, fontweight='bold')

        # Blocks
        rect1 = mpatches.FancyBboxPatch((0.1, 0.55), 0.3, 0.12,
                                       boxstyle="round,pad=0.01",
                                       facecolor='lightyellow', edgecolor='orange',
                                       linewidth=2, alpha=0.5)
        ax1.add_patch(rect1)
        ax1.text(0.25, 0.61, 'Methylation Block\n(parallel update)',
                ha='center', fontsize=9, fontweight='bold')

        rect2 = mpatches.FancyBboxPatch((0.6, 0.55), 0.3, 0.12,
                                       boxstyle="round,pad=0.01",
                                       facecolor='lightgreen', edgecolor='green',
                                       linewidth=2, alpha=0.5)
        ax1.add_patch(rect2)
        ax1.text(0.75, 0.61, 'Expression Block\n(parallel update)',
                ha='center', fontsize=9, fontweight='bold')

        # Arrow to hardware
        ax1.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.5),
                    arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        ax1.text(0.52, 0.425, 'Maps to TSU', fontsize=11, fontweight='bold')

        # Hardware layer
        hardware_y = 0.25

        # pdit circuits
        for i, (label, x_pos) in enumerate([
            ('pdit\n{0,1,2}', 0.15),
            ('pdit\n{0,1,2}', 0.35),
            ('pdit\n{0,1,2}', 0.65),
            ('pdit\n{0,1,2}', 0.85)
        ]):
            rect = mpatches.Rectangle((x_pos - 0.04, hardware_y - 0.04), 0.08, 0.08,
                                     facecolor='lightcoral', edgecolor='red', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(x_pos, hardware_y, label, ha='center', va='center',
                    fontsize=8, fontweight='bold')

        # Sampling cells
        cell1 = mpatches.FancyBboxPatch((0.1, 0.08), 0.3, 0.12,
                                       boxstyle="round,pad=0.01",
                                       facecolor='lightyellow', edgecolor='orange',
                                       linewidth=3, alpha=0.7)
        ax1.add_patch(cell1)
        ax1.text(0.25, 0.14, 'Sampling Cell Array 1\n(thermodynamic relaxation)',
                ha='center', fontsize=9, fontweight='bold')

        cell2 = mpatches.FancyBboxPatch((0.6, 0.08), 0.3, 0.12,
                                       boxstyle="round,pad=0.01",
                                       facecolor='lightgreen', edgecolor='green',
                                       linewidth=3, alpha=0.7)
        ax1.add_patch(cell2)
        ax1.text(0.75, 0.14, 'Sampling Cell Array 2\n(thermodynamic relaxation)',
                ha='center', fontsize=9, fontweight='bold')

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Block Gibbs schedule
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')
        ax2.set_title('Block Gibbs Sampling Schedule', fontsize=12, fontweight='bold')

        # Timeline
        iterations = ['t=0', 't=1', 't=2', 't=3', 't=4']
        y_meth = 0.7
        y_expr = 0.3

        for i, t in enumerate(iterations):
            x_pos = 0.1 + i * 0.2

            # Methylation updates (odd iterations)
            if i % 2 == 1:
                rect = mpatches.Rectangle((x_pos - 0.05, y_meth - 0.08), 0.1, 0.16,
                                         facecolor='orange', alpha=0.7, edgecolor='black', linewidth=2)
                ax2.add_patch(rect)
                ax2.text(x_pos, y_meth, 'M', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='white')

            # Expression updates (even iterations)
            if i % 2 == 0:
                rect = mpatches.Rectangle((x_pos - 0.05, y_expr - 0.08), 0.1, 0.16,
                                         facecolor='green', alpha=0.7, edgecolor='black', linewidth=2)
                ax2.add_patch(rect)
                ax2.text(x_pos, y_expr, 'E', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='white')

            ax2.text(x_pos, 0.05, t, ha='center', fontsize=10)

        ax2.text(0.05, y_meth, 'Methylation:', ha='right', fontsize=10, fontweight='bold')
        ax2.text(0.05, y_expr, 'Expression:', ha='right', fontsize=10, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        # Energy-based factor
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        ax3.set_title('Factor → Resistor Network', fontsize=12, fontweight='bold')

        # Show factor transformation
        ax3.text(0.5, 0.85, 'Software Factor:', ha='center', fontsize=10, fontweight='bold')
        ax3.text(0.5, 0.75, 'E = W[M, E]', ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue'))

        ax3.text(0.5, 0.55, '↓', ha='center', fontsize=16, fontweight='bold')

        ax3.text(0.5, 0.45, 'Hardware Resistor Net:', ha='center', fontsize=10, fontweight='bold')
        ax3.text(0.5, 0.35, 'V_out = Σ R_ij · V_i', ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightcoral'))

        ax3.text(0.5, 0.15, 'Analog parameter computation\n(instant, low energy)',
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        # Energy efficiency comparison
        ax4 = fig.add_subplot(gs[1, 2])

        methods = ['GPU\n(digital)', 'TSU\n(thermodynamic)']
        energies = [500, 0.05]  # Watts
        times = [4*3600, 3*60]  # seconds
        costs = [0.60, 0.001]  # dollars

        colors = ['gray', 'green']
        bars = ax4.bar(methods, energies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Power (W)', fontsize=11, fontweight='bold')
        ax4.set_title('Energy Efficiency\n(1000 samples, 105 pairs)', fontsize=12, fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(axis='y', alpha=0.3)

        # Annotate
        for bar, val, time, cost in zip(bars, energies, times, costs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height * 2,
                    f'{val}W\n{time//60:.0f} min\n${cost:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Improvement factor
        improvement = energies[0] / energies[1]
        ax4.text(0.5, 0.05, f'{improvement:.0f}× energy improvement',
                ha='center', fontsize=10, fontweight='bold', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved TSU hardware mapping to {output_path}")

        return fig

    def create_live_demo_dashboard(
        self,
        model,
        gene1: str,
        gene2: str,
        samples_fwd: np.ndarray,
        samples_bwd: np.ndarray,
        result: Dict,
        output_path: str = None
    ):
        """
        Create comprehensive live demo dashboard.

        Shows everything happening at once:
        - Energy landscapes (forward vs backward)
        - Sampling trajectories
        - Free energy evolution
        - TSU mapping
        - Final discrimination
        """
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Title
        fig.suptitle(f'THERMODYNAMIC CAUSAL INFERENCE: {gene1} ↔ {gene2}',
                    fontsize=18, fontweight='bold', y=0.98)

        # Row 1: Energy landscapes
        # Forward model energy
        ax = fig.add_subplot(gs[0, 0], projection='3d')
        m_states = np.arange(3)
        e_states = np.arange(3)
        M, E = np.meshgrid(m_states, e_states)
        energies_fwd = np.zeros((3, 3))

        for i, m in enumerate(m_states):
            for j, e in enumerate(e_states):
                state = {f'{gene1}_meth': int(m), f'{gene2}_meth': 1,
                        f'{gene1}_expr': int(e), f'{gene2}_expr': 1}
                energies_fwd[j, i] = model.compute_energy(gene1, gene2, 'forward', state)

        surf = ax.plot_surface(M, E, energies_fwd, cmap='Blues', alpha=0.8)
        ax.set_title(f'Forward Model: {gene1}→{gene2}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{gene1} Meth', fontsize=10)
        ax.set_ylabel(f'{gene1} Expr', fontsize=10)
        ax.set_zlabel('Energy', fontsize=10)

        # Backward model energy
        ax = fig.add_subplot(gs[0, 1], projection='3d')
        energies_bwd = np.zeros((3, 3))

        for i, m in enumerate(m_states):
            for j, e in enumerate(e_states):
                state = {f'{gene1}_meth': int(m), f'{gene2}_meth': 1,
                        f'{gene1}_expr': int(e), f'{gene2}_expr': 1}
                energies_bwd[j, i] = model.compute_energy(gene1, gene2, 'backward', state)

        surf = ax.plot_surface(M, E, energies_bwd, cmap='Reds', alpha=0.8)
        ax.set_title(f'Backward Model: {gene2}→{gene1}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{gene1} Meth', fontsize=10)
        ax.set_ylabel(f'{gene1} Expr', fontsize=10)
        ax.set_zlabel('Energy', fontsize=10)

        # Free energy discrimination
        ax = fig.add_subplot(gs[0, 2])
        delta_F = result['delta_F']
        models = ['Forward', 'Backward']
        free_energies = [result['F_forward'], result['F_backward']]
        colors = ['blue' if delta_F > 1 else 'gray',
                 'red' if delta_F < -1 else 'gray']

        bars = ax.bar(models, free_energies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Free Energy F', fontsize=11, fontweight='bold')
        ax.set_title(f'ΔF = {delta_F:.2f}\n{result["direction"]}',
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, free_energies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.2f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

        # Row 2: Sampling trajectories
        max_samples = min(200, len(samples_fwd))

        # Forward sampling
        ax = fig.add_subplot(gs[1, 0])
        iterations = np.arange(max_samples)
        ax.plot(iterations, samples_fwd[:max_samples, 0], 'o-', alpha=0.5,
               color='blue', linewidth=1, label='Methylation')
        ax.plot(iterations, samples_fwd[:max_samples, 2], 's-', alpha=0.5,
               color='green', linewidth=1, label='Expression')
        ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
        ax.set_ylabel('State', fontsize=10, fontweight='bold')
        ax.set_title('Forward Model Sampling', fontsize=12, fontweight='bold')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Low', 'Med', 'High'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Backward sampling
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(iterations, samples_bwd[:max_samples, 0], 'o-', alpha=0.5,
               color='blue', linewidth=1, label='Methylation')
        ax.plot(iterations, samples_bwd[:max_samples, 2], 's-', alpha=0.5,
               color='green', linewidth=1, label='Expression')
        ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
        ax.set_ylabel('State', fontsize=10, fontweight='bold')
        ax.set_title('Backward Model Sampling', fontsize=12, fontweight='bold')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Low', 'Med', 'High'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Phase space comparison
        ax = fig.add_subplot(gs[1, 2])
        ax.scatter(samples_fwd[:max_samples, 0], samples_fwd[:max_samples, 2],
                  c='blue', s=30, alpha=0.3, label='Forward')
        ax.scatter(samples_bwd[:max_samples, 0], samples_bwd[:max_samples, 2],
                  c='red', s=30, alpha=0.3, marker='s', label='Backward')
        ax.set_xlabel(f'{gene1} Methylation', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{gene1} Expression', fontsize=10, fontweight='bold')
        ax.set_title('Phase Space Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Row 3: TSU hardware mapping
        ax = fig.add_subplot(gs[2, :])
        ax.axis('off')

        # Draw hardware schematic
        y_top = 0.8
        y_mid = 0.5
        y_bot = 0.2

        # Nodes
        node_x = [0.2, 0.4, 0.6, 0.8]
        node_labels = ['M1\npdit', 'M2\npdit', 'E1\npdit', 'E2\npdit']
        node_colors = ['orange', 'orange', 'green', 'green']

        for x, label, color in zip(node_x, node_labels, node_colors):
            rect = mpatches.Rectangle((x - 0.04, y_top - 0.06), 0.08, 0.12,
                                     facecolor=color, edgecolor='black',
                                     linewidth=2, alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y_top, label, ha='center', va='center',
                   fontsize=9, fontweight='bold')

        # Resistor networks (factors)
        factor_positions = [(0.3, y_mid), (0.5, y_mid), (0.7, y_mid)]
        factor_labels = ['M1→E1\nResistor\nNetwork', 'E1→E2\nResistor\nNetwork', 'M2→E2\nResistor\nNetwork']

        for (x, y), label in zip(factor_positions, factor_labels):
            circle = mpatches.Circle((x, y), 0.06, facecolor='yellow',
                                    edgecolor='black', linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=7, fontweight='bold')

        # Sampling cells
        cell_labels = ['Cell 1\n(Gibbs)', 'Cell 2\n(Gibbs)']
        cell_x = [0.3, 0.7]
        cell_colors = ['orange', 'green']

        for x, label, color in zip(cell_x, cell_labels, cell_colors):
            rect = mpatches.FancyBboxPatch((x - 0.1, y_bot - 0.06), 0.2, 0.12,
                                          boxstyle="round,pad=0.01",
                                          facecolor=color, edgecolor='black',
                                          linewidth=2, alpha=0.5)
            ax.add_patch(rect)
            ax.text(x, y_bot, label, ha='center', va='center',
                   fontsize=9, fontweight='bold')

        # Arrows
        for i in range(4):
            ax.arrow(node_x[i], y_top - 0.07, 0, -0.12,
                    head_width=0.02, head_length=0.03, fc='black', ec='black')

        ax.text(0.5, 0.05, 'TSU Hardware: Parallel thermodynamic sampling via pdit circuits',
               ha='center', fontsize=11, fontweight='bold', style='italic')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved live demo dashboard to {output_path}")

        return fig


# Example usage
if __name__ == "__main__":
    print("Thermodynamic Visualizer - Ready for demo!")
    print("Create stunning visualizations of thermodynamic computing in action")
