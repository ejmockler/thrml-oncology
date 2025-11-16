#!/usr/bin/env python3
"""
Test script for inference pipeline.

Demonstrates:
1. Network inference on mock data
2. Network comparison between conditions
3. Checkpoint/resume functionality
4. Export to Cytoscape format
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from pathlib import Path
import logging

from core.inference import (
    infer_network_structure,
    compare_networks,
    run_inference_with_progress,
    export_network_to_cytoscape,
    summarize_network
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_mock_data(genes, n_samples=200, seed=42):
    """Generate mock discretized methylation and expression data"""
    np.random.seed(seed)

    # Simulate realistic patterns:
    # - Methylation is relatively stable (biased toward low/medium)
    # - Expression is more variable
    # - Add some correlation between meth and expr (anti-correlation)

    methylation = {}
    expression = {}

    for gene in genes:
        # Methylation: biased toward 0-1 (low-medium)
        meth = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.35, 0.15])
        methylation[gene] = meth

        # Expression: anti-correlated with methylation + noise
        expr = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            if meth[i] == 2:  # High methylation
                expr[i] = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            elif meth[i] == 1:  # Medium methylation
                expr[i] = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
            else:  # Low methylation
                expr[i] = np.random.choice([0, 1, 2], p=[0.1, 0.3, 0.6])

        expression[gene] = expr

    meth_df = pd.DataFrame(methylation)
    expr_df = pd.DataFrame(expression)

    return meth_df, expr_df


def generate_resistant_data(meth_df, expr_df, genes, n_samples=200, seed=43):
    """
    Generate 'resistant' data with modified relationships.

    Simulates resistance by:
    1. Flipping some methylation-expression relationships
    2. Adding noise to break some correlations
    """
    np.random.seed(seed)

    methylation = {}
    expression = {}

    for gene in genes:
        # Keep methylation similar
        meth = meth_df[gene].values
        methylation[gene] = meth

        # Modify expression patterns
        expr = np.zeros(n_samples, dtype=int)

        # For "EGFR", flip the relationship (simulating resistance mechanism)
        if gene == "EGFR":
            for i in range(n_samples):
                if meth[i] == 2:  # High methylation now activates (paradoxical)
                    expr[i] = np.random.choice([0, 1, 2], p=[0.1, 0.3, 0.6])
                elif meth[i] == 1:
                    expr[i] = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
                else:
                    expr[i] = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        else:
            # Keep others similar to sensitive
            expr = expr_df[gene].values

        expression[gene] = expr

    meth_df_res = pd.DataFrame(methylation)
    expr_df_res = pd.DataFrame(expression)

    return meth_df_res, expr_df_res


def test_basic_inference():
    """Test basic network inference on small gene set"""
    logger.info("=" * 70)
    logger.info("TEST 1: Basic Network Inference")
    logger.info("=" * 70)

    # Small gene set for quick testing
    genes = ["EGFR", "KRAS", "BRAF", "PIK3CA"]

    # Mock INDRA priors (realistic pathway: EGFR -> KRAS -> BRAF)
    prior_network = {
        ("EGFR", "KRAS"): 0.9,
        ("KRAS", "BRAF"): 0.85,
        ("EGFR", "PIK3CA"): 0.7,
        ("PIK3CA", "KRAS"): 0.6,
    }

    # Generate data
    meth_df, expr_df = generate_mock_data(genes, n_samples=100)

    logger.info(f"Generated data: {meth_df.shape[0]} samples, {len(genes)} genes")
    logger.info(f"Prior network: {len(prior_network)} edges")

    # Run inference
    results = infer_network_structure(
        genes=genes,
        methylation_data=meth_df,
        expression_data=expr_df,
        prior_network=prior_network,
        n_samples=500,  # Moderate sample count
        n_warmup=200,
        parallel=False  # Sequential for debugging
    )

    # Display results
    logger.info("\nInference Results:")
    logger.info("-" * 70)

    for (g1, g2), result in sorted(results.items(), key=lambda x: abs(x[1]['delta_F']), reverse=True):
        logger.info(
            f"{g1:10s} <-> {g2:10s} : {result['direction']:20s} "
            f"| ΔF={result['delta_F']:6.2f} | conf={result['confidence']:.3f} "
            f"| prior={result.get('prior_belief', 0.0):.2f}"
        )

    # Summary statistics
    summary = summarize_network(results)
    logger.info("\nNetwork Summary:")
    logger.info("-" * 70)
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    return results


def test_network_comparison():
    """Test comparing sensitive vs resistant networks"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Network Comparison (Sensitive vs Resistant)")
    logger.info("=" * 70)

    genes = ["EGFR", "KRAS", "BRAF"]

    prior_network = {
        ("EGFR", "KRAS"): 0.9,
        ("KRAS", "BRAF"): 0.85,
    }

    # Generate sensitive data
    meth_sens, expr_sens = generate_mock_data(genes, n_samples=100, seed=42)

    # Generate resistant data (with EGFR relationship flipped)
    meth_res, expr_res = generate_resistant_data(
        meth_sens, expr_sens, genes, n_samples=100, seed=43
    )

    logger.info("Running inference on SENSITIVE cells...")
    network_sensitive = infer_network_structure(
        genes=genes,
        methylation_data=meth_sens,
        expression_data=expr_sens,
        prior_network=prior_network,
        n_samples=500,
        n_warmup=200,
        parallel=False
    )

    logger.info("Running inference on RESISTANT cells...")
    network_resistant = infer_network_structure(
        genes=genes,
        methylation_data=meth_res,
        expression_data=expr_res,
        prior_network=prior_network,
        n_samples=500,
        n_warmup=200,
        parallel=False
    )

    # Compare networks
    logger.info("\nComparing networks...")
    comparison = compare_networks(
        network_sensitive,
        network_resistant,
        threshold=1.5,  # Free energy change threshold
        min_delta_F=0.5  # Minimum edge strength
    )

    # Display comparison
    logger.info("\nNetwork Comparison Results:")
    logger.info("-" * 70)

    for change_type, edges in comparison.items():
        logger.info(f"\n{change_type.upper()} ({len(edges)} edges):")
        for edge in edges[:10]:  # Show first 10
            logger.info(f"  {edge[0]} <-> {edge[1]}")

    return comparison


def test_checkpointing():
    """Test checkpoint and resume functionality"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Checkpoint and Resume")
    logger.info("=" * 70)

    genes = ["EGFR", "KRAS", "BRAF"]

    prior_network = {
        ("EGFR", "KRAS"): 0.9,
        ("KRAS", "BRAF"): 0.85,
    }

    # Generate data
    meth_df, expr_df = generate_mock_data(genes, n_samples=100)

    data_dict = {
        'methylation': meth_df,
        'expression': expr_df
    }

    # Output files
    output_dir = Path("/tmp/thrml_inference_test")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "network_results.json"

    logger.info(f"Output directory: {output_dir}")
    logger.info("Running inference with checkpointing every 2 pairs...")

    # Run inference with checkpointing
    results = run_inference_with_progress(
        genes=genes,
        data=data_dict,
        prior_network=prior_network,
        output_file=str(output_file),
        checkpoint_every=2,  # Checkpoint frequently
        n_samples=300,
        n_warmup=100,
        parallel=False
    )

    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Total pairs processed: {len(results)}")

    # Export to Cytoscape
    cytoscape_file = output_dir / "network_cytoscape.csv"
    export_network_to_cytoscape(
        results,
        str(cytoscape_file),
        min_confidence=0.3,
        min_delta_F=0.5
    )

    logger.info(f"Cytoscape network exported to: {cytoscape_file}")

    # Display exported file preview
    if cytoscape_file.exists():
        df = pd.read_csv(cytoscape_file)
        logger.info(f"\nExported {len(df)} edges to Cytoscape format:")
        logger.info(df.head(10).to_string())

    return results


def test_parallel_execution():
    """Test parallel execution with ThreadPoolExecutor"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Parallel Execution")
    logger.info("=" * 70)

    genes = ["EGFR", "KRAS", "BRAF", "PIK3CA", "PTEN"]

    prior_network = {
        ("EGFR", "KRAS"): 0.9,
        ("KRAS", "BRAF"): 0.85,
        ("EGFR", "PIK3CA"): 0.7,
        ("PIK3CA", "PTEN"): 0.6,
    }

    # Generate larger dataset
    meth_df, expr_df = generate_mock_data(genes, n_samples=150)

    logger.info(f"Testing parallel execution with {len(genes)} genes")
    logger.info("Note: JAX manages GPU resources internally")

    import time

    # Time parallel execution
    start_time = time.time()

    results_parallel = infer_network_structure(
        genes=genes,
        methylation_data=meth_df,
        expression_data=expr_df,
        prior_network=prior_network,
        n_samples=300,
        n_warmup=100,
        parallel=True,
        max_workers=4  # Use 4 workers
    )

    parallel_time = time.time() - start_time

    logger.info(f"\nParallel execution completed in {parallel_time:.2f} seconds")
    logger.info(f"Processed {len(results_parallel)} gene pairs")

    # Summary
    summary = summarize_network(results_parallel)
    logger.info("\nParallel Inference Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    return results_parallel


def main():
    """Run all tests"""
    logger.info("THRML Inference Pipeline Test Suite")
    logger.info("=" * 70)

    try:
        # Test 1: Basic inference
        results_basic = test_basic_inference()

        # Test 2: Network comparison
        comparison = test_network_comparison()

        # Test 3: Checkpointing
        results_checkpoint = test_checkpointing()

        # Test 4: Parallel execution
        results_parallel = test_parallel_execution()

        logger.info("\n" + "=" * 70)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
