#!/usr/bin/env python3
"""
Results Analysis and Visualization
Compares networks, predicts drugs, validates, and generates figures.
"""

import sys
import os
import argparse
import pickle
import json
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.inference import compare_networks, summarize_network
from core.validation import (
    predict_drugs_from_changes,
    validate_predictions,
    summarize_results
)
from core.indra_client import IndraClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_network_comparison(
    network_sensitive,
    network_resistant,
    changed_edges,
    output_dir
):
    """Create network comparison figure"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Build NetworkX graphs
    G_sens = nx.DiGraph()
    G_res = nx.DiGraph()

    for (g1, g2), info in network_sensitive.items():
        if info['direction'] != 'undecided':
            weight = abs(info['delta_F'])
            if info['direction'].startswith(g1):
                G_sens.add_edge(g1, g2, weight=weight)
            else:
                G_sens.add_edge(g2, g1, weight=weight)

    for (g1, g2), info in network_resistant.items():
        if info['direction'] != 'undecided':
            weight = abs(info['delta_F'])
            if info['direction'].startswith(g1):
                G_res.add_edge(g1, g2, weight=weight)
            else:
                G_res.add_edge(g2, g1, weight=weight)

    # Layout
    pos = nx.spring_layout(G_sens, seed=42)

    # Sensitive network
    ax1.set_title('Sensitive Cells Network', fontsize=14, fontweight='bold')
    nx.draw_networkx(
        G_sens, pos, ax=ax1,
        node_color='lightblue',
        node_size=1000,
        font_size=10,
        arrows=True,
        arrowsize=20,
        width=2
    )

    # Resistant network
    ax2.set_title('Resistant Cells Network', fontsize=14, fontweight='bold')

    # Highlight changed edges
    changed_edge_list = (
        changed_edges['edge_flips'] +
        changed_edges['new_edges'] +
        changed_edges['lost_edges']
    )

    edge_colors = []
    for edge in G_res.edges():
        if edge in changed_edge_list or edge[::-1] in changed_edge_list:
            edge_colors.append('red')
        else:
            edge_colors.append('black')

    nx.draw_networkx(
        G_res, pos, ax=ax2,
        node_color='lightcoral',
        node_size=1000,
        font_size=10,
        arrows=True,
        arrowsize=20,
        edge_color=edge_colors,
        width=2
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Changed edges'),
        Line2D([0], [0], color='black', lw=2, label='Stable edges')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    output_path = Path(output_dir) / 'network_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved network comparison to {output_path}")

    plt.close()


def visualize_precision(validation_results, output_dir):
    """Create precision comparison bar chart"""

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['THRML Predictions', 'Random Baseline']
    precisions = [
        validation_results['precision'] * 100,
        validation_results['baseline_precision'] * 100
    ]

    bars = ax.bar(categories, precisions, color=['#2ecc71', '#95a5a6'])

    # Add value labels on bars
    for bar, val in zip(bars, precisions):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{val:.1f}%',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )

    ax.set_ylabel('Precision (%)', fontsize=12)
    ax.set_title('Drug Prediction Precision vs Baseline', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(precisions) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    # Add improvement factor
    improvement = validation_results['precision'] / validation_results['baseline_precision']
    ax.text(
        0.5, 0.95,
        f'{improvement:.1f}× improvement over random',
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    output_path = Path(output_dir) / 'precision_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved precision comparison to {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze THRML inference results')

    parser.add_argument('--input', type=str, required=True,
                       help='Input pickle file from 02_run_inference.py')
    parser.add_argument('--output-dir', type=str, default='results/figures',
                       help='Output directory for figures')
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Threshold for significant edge changes (default: 2.0)')
    parser.add_argument('--top-drugs', type=int, default=10,
                       help='Number of top drugs to predict (default: 10)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== STEP 1: Load results ==========
    logger.info("=" * 60)
    logger.info("STEP 1: Loading inference results")
    logger.info("=" * 60)

    with open(args.input, 'rb') as f:
        results = pickle.load(f)

    genes = results['genes']
    network_sensitive = results['network_sensitive']
    network_resistant = results['network_resistant']

    logger.info(f"Loaded results for {len(genes)} genes")
    logger.info(f"  - Sensitive network: {len(network_sensitive)} edges")
    logger.info(f"  - Resistant network: {len(network_resistant)} edges")

    # ========== STEP 2: Compare networks ==========
    logger.info("=" * 60)
    logger.info("STEP 2: Comparing networks")
    logger.info("=" * 60)

    changed_edges = compare_networks(
        network_sensitive,
        network_resistant,
        threshold=args.threshold
    )

    logger.info(f"Network changes detected:")
    logger.info(f"  - Edge flips: {len(changed_edges['edge_flips'])}")
    logger.info(f"  - Edge weakening: {len(changed_edges['edge_weakening'])}")
    logger.info(f"  - Edge strengthening: {len(changed_edges['edge_strengthening'])}")
    logger.info(f"  - New edges: {len(changed_edges['new_edges'])}")
    logger.info(f"  - Lost edges: {len(changed_edges['lost_edges'])}")

    # ========== STEP 3: Predict drugs ==========
    logger.info("=" * 60)
    logger.info("STEP 3: Predicting drug targets")
    logger.info("=" * 60)

    indra_client = IndraClient()

    predicted_drugs = predict_drugs_from_changes(
        changed_edges,
        indra_client,
        top_n=args.top_drugs
    )

    logger.info(f"Predicted {len(predicted_drugs)} drug candidates:")
    for i, drug in enumerate(predicted_drugs[:5], 1):
        logger.info(f"  {i}. {drug['drug_name']}")
        logger.info(f"     Targets: {', '.join(drug['target_genes'])}")
        logger.info(f"     Confidence: {drug['confidence']:.3f}")

    # ========== STEP 4: Validate predictions ==========
    logger.info("=" * 60)
    logger.info("STEP 4: Validating predictions")
    logger.info("=" * 60)

    # Use mock IC50 data (in real version, load from file)
    from core.validation import MOCK_IC50_DATA

    validation_results = validate_predictions(
        predicted_drugs,
        MOCK_IC50_DATA,
        threshold=1.0
    )

    logger.info(f"Validation results:")
    logger.info(f"  - Precision: {validation_results['precision']:.1%}")
    logger.info(f"  - Baseline: {validation_results['baseline_precision']:.1%}")
    logger.info(f"  - Improvement: {validation_results['precision'] / validation_results['baseline_precision']:.1f}×")
    logger.info(f"  - Validated drugs: {len(validation_results['validated_drugs'])}")

    # ========== STEP 5: Create summary ==========
    logger.info("=" * 60)
    logger.info("STEP 5: Creating summary")
    logger.info("=" * 60)

    summary = summarize_results(
        network_sensitive,
        network_resistant,
        changed_edges,
        predicted_drugs,
        validation_results
    )

    # Save summary
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    # ========== STEP 6: Generate figures ==========
    logger.info("=" * 60)
    logger.info("STEP 6: Generating figures")
    logger.info("=" * 60)

    visualize_network_comparison(
        network_sensitive,
        network_resistant,
        changed_edges,
        output_dir
    )

    visualize_precision(validation_results, output_dir)

    # ========== FINAL SUMMARY ==========
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results directory: {output_dir}")
    logger.info(f"  - network_comparison.png")
    logger.info(f"  - precision_comparison.png")
    logger.info(f"  - summary.json")
    logger.info("")
    logger.info("Key findings:")
    logger.info(f"  - {summary['n_edges_changed']} edges changed between networks")
    logger.info(f"  - {summary['n_drugs_predicted']} drugs predicted")
    logger.info(f"  - {summary['precision']:.1%} precision ({summary['improvement_factor']:.1f}× better than random)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
