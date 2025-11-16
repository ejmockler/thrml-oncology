#!/usr/bin/env python3
"""
Main Inference Pipeline
Runs causal network inference on gene pairs with parallel GPU support.
"""

import sys
import os
import argparse
import pickle
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thrml_model import GeneNetworkModel
from core.data_loader import generate_synthetic_data, prepare_model_input, discretize_values
from core.indra_client import IndraClient
from core.inference import infer_network_structure, run_inference_with_progress
from core.validation import predict_drugs_from_changes, validate_predictions, summarize_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run THRML causal inference')

    # Gene selection
    parser.add_argument('--genes', type=int, default=15,
                       help='Number of genes to analyze (default: 15)')
    parser.add_argument('--gene-list', type=str, nargs='+',
                       help='Specific genes to analyze (overrides --genes)')

    # Sample configuration
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples per model (default: 1000)')
    parser.add_argument('--warmup', type=int, default=100,
                       help='Burn-in iterations (default: 100)')

    # Pair selection
    parser.add_argument('--pairs-start', type=int, default=None,
                       help='Start index for gene pairs (for parallel execution)')
    parser.add_argument('--pairs-end', type=int, default=None,
                       help='End index for gene pairs (for parallel execution)')

    # Data source
    parser.add_argument('--synthetic-data', action='store_true',
                       help='Use synthetic data instead of CCLE')
    parser.add_argument('--data-dir', type=str, default='data/ccle',
                       help='Directory containing CCLE data')

    # Output
    parser.add_argument('--output', type=str, default='results/inference.pkl',
                       help='Output file for results')
    parser.add_argument('--checkpoint-every', type=int, default=10,
                       help='Checkpoint frequency (pairs)')

    # Testing
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (5 genes, 100 samples)')

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        logger.info("Running in QUICK TEST mode")
        args.genes = 5
        args.samples = 100
        args.warmup = 50

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ========== STEP 1: Load and prepare data ==========
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preparing data")
    logger.info("=" * 60)

    if args.synthetic_data or args.quick_test:
        logger.info("Generating synthetic data...")

        if args.gene_list:
            genes = args.gene_list
        else:
            # Common EGFR pathway genes
            all_genes = [
                'EGFR', 'KRAS', 'BRAF', 'MEK1', 'ERK1',
                'PIK3CA', 'AKT1', 'MTOR', 'TP53', 'PTEN',
                'MYC', 'JUN', 'FOS', 'STAT3', 'JAK1'
            ]
            genes = all_genes[:args.genes]

        data = generate_synthetic_data(
            genes,
            n_sensitive=25,
            n_resistant=25,
            seed=42
        )

        logger.info(f"Generated data for {len(genes)} genes:")
        logger.info(f"  - {len(data['sensitive_idx'])} sensitive cell lines")
        logger.info(f"  - {len(data['resistant_idx'])} resistant cell lines")
        logger.info(f"  - Ground truth: {len(data['ground_truth'])} causal edges")
    else:
        logger.info(f"Loading CCLE data from {args.data_dir}...")
        from core.data_loader import load_ccle_data

        if args.gene_list:
            genes = args.gene_list
        else:
            genes = None  # Will use all genes in CCLE data

        data = load_ccle_data(genes, args.data_dir)
        genes = data['genes']

    # Discretize if needed
    model_input = prepare_model_input(data, genes, discretize=True)
    meth_data = model_input['methylation_discrete']
    expr_data = model_input['expression_discrete']

    logger.info(f"Data prepared:")
    logger.info(f"  - Methylation: {meth_data.shape}")
    logger.info(f"  - Expression: {expr_data.shape}")

    # ========== STEP 2: Build INDRA prior network ==========
    logger.info("=" * 60)
    logger.info("STEP 2: Building INDRA prior network")
    logger.info("=" * 60)

    indra_client = IndraClient()

    try:
        logger.info(f"Querying INDRA for {len(genes)} genes...")
        prior_network = indra_client.build_prior_network(genes)
        logger.info(f"Built prior network with {len(prior_network)} edges")
    except Exception as e:
        logger.warning(f"INDRA query failed: {e}")
        logger.warning("Using empty prior network")
        prior_network = {}

    # ========== STEP 3: Run inference on SENSITIVE cells ==========
    logger.info("=" * 60)
    logger.info("STEP 3: Inferring network structure (SENSITIVE cells)")
    logger.info("=" * 60)

    # Split data
    sensitive_idx = data['sensitive_idx']
    resistant_idx = data['resistant_idx']

    meth_sensitive = meth_data.iloc[sensitive_idx]
    expr_sensitive = expr_data.iloc[sensitive_idx]

    logger.info(f"Running inference on {len(sensitive_idx)} sensitive cell lines...")

    network_sensitive = run_inference_with_progress(
        genes=genes,
        methylation_data=meth_sensitive,
        expression_data=expr_sensitive,
        prior_network=prior_network,
        n_samples=args.samples,
        n_warmup=args.warmup,
        output_file=str(output_path).replace('.pkl', '_sensitive.pkl'),
        checkpoint_every=args.checkpoint_every,
        pairs_start=args.pairs_start,
        pairs_end=args.pairs_end
    )

    logger.info(f"Sensitive network: {len(network_sensitive)} edges inferred")

    # ========== STEP 4: Run inference on RESISTANT cells ==========
    logger.info("=" * 60)
    logger.info("STEP 4: Inferring network structure (RESISTANT cells)")
    logger.info("=" * 60)

    meth_resistant = meth_data.iloc[resistant_idx]
    expr_resistant = expr_data.iloc[resistant_idx]

    logger.info(f"Running inference on {len(resistant_idx)} resistant cell lines...")

    network_resistant = run_inference_with_progress(
        genes=genes,
        methylation_data=meth_resistant,
        expression_data=expr_resistant,
        prior_network=prior_network,
        n_samples=args.samples,
        n_warmup=args.warmup,
        output_file=str(output_path).replace('.pkl', '_resistant.pkl'),
        checkpoint_every=args.checkpoint_every,
        pairs_start=args.pairs_start,
        pairs_end=args.pairs_end
    )

    logger.info(f"Resistant network: {len(network_resistant)} edges inferred")

    # ========== STEP 5: Save results ==========
    logger.info("=" * 60)
    logger.info("STEP 5: Saving results")
    logger.info("=" * 60)

    results = {
        'genes': genes,
        'network_sensitive': network_sensitive,
        'network_resistant': network_resistant,
        'prior_network': prior_network,
        'ground_truth': data.get('ground_truth', {}),
        'config': {
            'n_samples': args.samples,
            'n_warmup': args.warmup,
            'n_genes': len(genes),
            'n_sensitive': len(sensitive_idx),
            'n_resistant': len(resistant_idx)
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"Results saved to {output_path}")

    # ========== SUMMARY ==========
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Genes analyzed: {len(genes)}")
    logger.info(f"Edges in sensitive network: {len(network_sensitive)}")
    logger.info(f"Edges in resistant network: {len(network_resistant)}")
    logger.info(f"Total runtime: See above")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  python scripts/03_analyze_results.py --input {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
