"""
Inference Orchestration Pipeline for Causal Network Discovery

This module coordinates batch inference across gene pairs, manages parallel
execution, implements network comparison, and provides progress tracking.
"""

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import time

from core.thrml_model import GeneNetworkModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def infer_network_structure(
    genes: List[str],
    methylation_data: pd.DataFrame,
    expression_data: pd.DataFrame,
    prior_network: Dict[Tuple[str, str], float],
    n_samples: int = 1000,
    n_warmup: int = 100,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    threshold: float = 1.0
) -> Dict[Tuple[str, str], Dict]:
    """
    Infer causal directions for all gene pairs.

    Args:
        genes: List of gene symbols
        methylation_data: Discretized methylation [samples × genes]
        expression_data: Discretized expression [samples × genes]
        prior_network: INDRA priors mapping (gene1, gene2) -> belief
        n_samples: Samples per model
        n_warmup: Burn-in iterations
        parallel: Use multiprocessing (if True)
        max_workers: Number of parallel workers (None = use all cores)
        threshold: Free energy threshold for direction decision

    Returns:
        {
            (gene1, gene2): {
                'direction': 'gene1 -> gene2' | 'gene2 -> gene1' | 'undecided',
                'delta_F': float,
                'F_forward': float,
                'F_backward': float,
                'confidence': float,
                'prior_belief': float
            }
        }
    """
    logger.info(f"Starting network inference for {len(genes)} genes")
    logger.info(f"Total possible pairs: {len(genes) * (len(genes) - 1)}")

    # Create model instance
    model = GeneNetworkModel(genes, prior_network)

    # Prepare data dictionary
    data = _prepare_data_dict(genes, methylation_data, expression_data)

    # Generate all gene pairs (order matters for causal direction)
    gene_pairs = [(g1, g2) for g1 in genes for g2 in genes if g1 != g2]

    # Filter pairs that have prior evidence (optional optimization)
    if prior_network:
        gene_pairs_filtered = [
            (g1, g2) for g1, g2 in gene_pairs
            if (g1, g2) in prior_network or (g2, g1) in prior_network
        ]
        logger.info(f"Filtered to {len(gene_pairs_filtered)} pairs with prior evidence")
        gene_pairs = gene_pairs_filtered

    # Run inference
    if parallel and len(gene_pairs) > 10:
        results = _infer_parallel(
            gene_pairs, model, data, n_samples, n_warmup,
            threshold, max_workers
        )
    else:
        results = _infer_sequential(
            gene_pairs, model, data, n_samples, n_warmup, threshold
        )

    logger.info(f"Completed inference for {len(results)} gene pairs")

    # Convert to dict keyed by gene pair
    network_results = {}
    for result in results:
        if result is not None:
            gene_pair = (result['gene1'], result['gene2'])
            network_results[gene_pair] = result

    return network_results


def _prepare_data_dict(
    genes: List[str],
    methylation_data: pd.DataFrame,
    expression_data: pd.DataFrame
) -> Dict[str, jnp.ndarray]:
    """
    Convert pandas DataFrames to JAX arrays.

    Args:
        genes: List of gene symbols
        methylation_data: DataFrame with columns matching gene names
        expression_data: DataFrame with columns matching gene names

    Returns:
        Dict mapping 'GENE_meth' and 'GENE_expr' to JAX arrays
    """
    data = {}

    for gene in genes:
        if gene in methylation_data.columns:
            data[f'{gene}_meth'] = jnp.array(methylation_data[gene].values)
        else:
            logger.warning(f"Gene {gene} not found in methylation data")

        if gene in expression_data.columns:
            data[f'{gene}_expr'] = jnp.array(expression_data[gene].values)
        else:
            logger.warning(f"Gene {gene} not found in expression data")

    return data


def _infer_sequential(
    gene_pairs: List[Tuple[str, str]],
    model: GeneNetworkModel,
    data: Dict[str, jnp.ndarray],
    n_samples: int,
    n_warmup: int,
    threshold: float
) -> List[Dict]:
    """Run inference sequentially with progress bar"""
    results = []

    with tqdm(total=len(gene_pairs), desc="Testing gene pairs") as pbar:
        for gene1, gene2 in gene_pairs:
            try:
                result = model.test_causal_direction(
                    gene1, gene2, data, n_samples=n_samples
                )

                # Add prior belief if available
                prior_belief = model.prior_network.get((gene1, gene2), 0.0)
                result['prior_belief'] = prior_belief

                results.append(result)

                # Log strong directional signals
                if abs(result['delta_F']) > threshold:
                    logger.info(
                        f"{result['direction']}: ΔF={result['delta_F']:.2f}, "
                        f"conf={result['confidence']:.3f}"
                    )

            except Exception as e:
                logger.error(f"Failed to test {gene1} -> {gene2}: {e}")
                results.append(None)

            pbar.update(1)

    return results


def _infer_parallel(
    gene_pairs: List[Tuple[str, str]],
    model: GeneNetworkModel,
    data: Dict[str, jnp.ndarray],
    n_samples: int,
    n_warmup: int,
    threshold: float,
    max_workers: Optional[int] = None
) -> List[Dict]:
    """Run inference in parallel using ThreadPoolExecutor (JAX handles GPU)"""
    results = []

    # Use ThreadPoolExecutor since JAX manages GPU resources
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pair = {
            executor.submit(
                _safe_test_direction, model, g1, g2, data, n_samples
            ): (g1, g2)
            for g1, g2 in gene_pairs
        }

        # Collect results with progress bar
        with tqdm(total=len(gene_pairs), desc="Testing gene pairs (parallel)") as pbar:
            for future in as_completed(future_to_pair):
                gene1, gene2 = future_to_pair[future]

                try:
                    result = future.result()

                    if result is not None:
                        # Add prior belief
                        prior_belief = model.prior_network.get((gene1, gene2), 0.0)
                        result['prior_belief'] = prior_belief

                        results.append(result)

                        # Log strong directional signals
                        if abs(result['delta_F']) > threshold:
                            logger.info(
                                f"{result['direction']}: ΔF={result['delta_F']:.2f}, "
                                f"conf={result['confidence']:.3f}"
                            )
                    else:
                        results.append(None)

                except Exception as e:
                    logger.error(f"Failed to get result for {gene1} -> {gene2}: {e}")
                    results.append(None)

                pbar.update(1)

    return results


def _safe_test_direction(
    model: GeneNetworkModel,
    gene1: str,
    gene2: str,
    data: Dict[str, jnp.ndarray],
    n_samples: int
) -> Optional[Dict]:
    """Safely test causal direction with error handling"""
    try:
        return model.test_causal_direction(gene1, gene2, data, n_samples)
    except Exception as e:
        logger.error(f"Error testing {gene1} -> {gene2}: {e}")
        return None


def compare_networks(
    network_sensitive: Dict[Tuple[str, str], Dict],
    network_resistant: Dict[Tuple[str, str], Dict],
    threshold: float = 2.0,
    min_delta_F: float = 1.0
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Find edges that changed between sensitive and resistant cells.

    Args:
        network_sensitive: Inference results for sensitive cells
        network_resistant: Inference results for resistant cells
        threshold: Threshold for identifying edge strength changes
        min_delta_F: Minimum |ΔF| to consider an edge significant

    Returns:
        {
            'edge_flips': [(g1, g2), ...],  # Direction reversed
            'edge_weakening': [(g1, g2), ...],  # |ΔF| decreased >threshold
            'edge_strengthening': [(g1, g2), ...],  # |ΔF| increased >threshold
            'new_edges': [(g1, g2), ...],  # Present in resistant only
            'lost_edges': [(g1, g2), ...],  # Present in sensitive only
            'stable_edges': [(g1, g2), ...]  # No significant change
        }
    """
    logger.info("Comparing sensitive vs resistant networks")

    edge_flips = []
    edge_weakening = []
    edge_strengthening = []
    new_edges = []
    lost_edges = []
    stable_edges = []

    # Get all gene pairs from both networks
    all_pairs = set(network_sensitive.keys()) | set(network_resistant.keys())

    for pair in all_pairs:
        sens_result = network_sensitive.get(pair)
        resist_result = network_resistant.get(pair)

        # Check if edge is new or lost
        if sens_result is None:
            if resist_result and abs(resist_result['delta_F']) > min_delta_F:
                new_edges.append(pair)
            continue

        if resist_result is None:
            if abs(sens_result['delta_F']) > min_delta_F:
                lost_edges.append(pair)
            continue

        # Both networks have this pair - compare them
        sens_delta = sens_result['delta_F']
        resist_delta = resist_result['delta_F']

        sens_direction = sens_result['direction']
        resist_direction = resist_result['direction']

        # Check for direction flip
        if _directions_flipped(sens_direction, resist_direction):
            edge_flips.append(pair)
            logger.info(
                f"Edge flip: {pair} changed from '{sens_direction}' to '{resist_direction}'"
            )
            continue

        # Check for strength changes
        delta_change = abs(resist_delta) - abs(sens_delta)
        percent_change = delta_change / (abs(sens_delta) + 1e-6)

        if abs(delta_change) > threshold:
            if delta_change > 0:
                edge_strengthening.append(pair)
                logger.info(
                    f"Edge strengthening: {pair} |ΔF| {abs(sens_delta):.2f} -> "
                    f"{abs(resist_delta):.2f} (+{percent_change*100:.1f}%)"
                )
            else:
                edge_weakening.append(pair)
                logger.info(
                    f"Edge weakening: {pair} |ΔF| {abs(sens_delta):.2f} -> "
                    f"{abs(resist_delta):.2f} ({percent_change*100:.1f}%)"
                )
        else:
            # Edge is stable
            if abs(sens_delta) > min_delta_F or abs(resist_delta) > min_delta_F:
                stable_edges.append(pair)

    results = {
        'edge_flips': edge_flips,
        'edge_weakening': edge_weakening,
        'edge_strengthening': edge_strengthening,
        'new_edges': new_edges,
        'lost_edges': lost_edges,
        'stable_edges': stable_edges
    }

    # Log summary
    logger.info("Network comparison summary:")
    for key, edges in results.items():
        logger.info(f"  {key}: {len(edges)}")

    return results


def _directions_flipped(dir1: str, dir2: str) -> bool:
    """Check if two direction strings represent opposite directions"""
    if dir1 == "undecided" or dir2 == "undecided":
        return False

    # Extract gene order from direction strings like "GENE1 -> GENE2"
    if " -> " not in dir1 or " -> " not in dir2:
        return False

    genes1 = dir1.split(" -> ")
    genes2 = dir2.split(" -> ")

    # Flip if gene order is reversed
    return genes1[0] == genes2[1] and genes1[1] == genes2[0]


def process_gene_pairs_batch(
    gene_pairs: List[Tuple[str, str]],
    model: GeneNetworkModel,
    data: Dict[str, jnp.ndarray],
    n_samples: int = 1000,
    n_warmup: int = 100
) -> List[Dict]:
    """
    Process multiple gene pairs in batch.

    Useful for parallelization across GPUs or distributing work.

    Args:
        gene_pairs: List of (gene1, gene2) tuples
        model: GeneNetworkModel instance
        data: Observed data dictionary
        n_samples: Number of samples for free energy estimation
        n_warmup: Burn-in samples

    Returns:
        List of result dictionaries (one per pair)
    """
    results = []

    for gene1, gene2 in gene_pairs:
        try:
            result = model.test_causal_direction(
                gene1, gene2, data, n_samples=n_samples
            )

            # Add prior belief
            prior_belief = model.prior_network.get((gene1, gene2), 0.0)
            result['prior_belief'] = prior_belief

            results.append(result)

        except Exception as e:
            logger.error(f"Error processing {gene1} -> {gene2}: {e}")
            results.append({
                'gene1': gene1,
                'gene2': gene2,
                'direction': 'error',
                'delta_F': 0.0,
                'F_forward': 0.0,
                'F_backward': 0.0,
                'confidence': 0.0,
                'error': str(e)
            })

    return results


def run_inference_with_progress(
    genes: List[str],
    data: Dict[str, Any],
    prior_network: Dict[Tuple[str, str], float],
    output_file: str,
    checkpoint_every: int = 10,
    n_samples: int = 1000,
    n_warmup: int = 100,
    parallel: bool = True
) -> Dict[Tuple[str, str], Dict]:
    """
    Run inference with progress bars and checkpointing.

    Saves intermediate results every N pairs to allow resuming from crashes.

    Args:
        genes: List of gene symbols
        data: Dict with 'methylation' and 'expression' DataFrames
        prior_network: INDRA prior network
        output_file: Path to save final results (JSON)
        checkpoint_every: Save checkpoint every N gene pairs
        n_samples: Samples per model
        n_warmup: Burn-in samples
        parallel: Use parallel execution

    Returns:
        Complete network inference results
    """
    output_path = Path(output_file)
    checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.pkl"

    # Check for existing checkpoint
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        completed_pairs = set(checkpoint['completed_pairs'])
        results = checkpoint['results']
        logger.info(f"Resuming from checkpoint: {len(completed_pairs)} pairs completed")
    else:
        completed_pairs = set()
        results = {}

    # Create model
    model = GeneNetworkModel(genes, prior_network)

    # Prepare data
    meth_df = data['methylation']
    expr_df = data['expression']
    data_dict = _prepare_data_dict(genes, meth_df, expr_df)

    # Generate gene pairs (excluding completed ones)
    all_pairs = [(g1, g2) for g1 in genes for g2 in genes if g1 != g2]
    remaining_pairs = [p for p in all_pairs if p not in completed_pairs]

    logger.info(f"Total pairs: {len(all_pairs)}, Remaining: {len(remaining_pairs)}")

    # Process in batches with checkpointing
    batch_size = checkpoint_every
    num_batches = (len(remaining_pairs) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(remaining_pairs))
        batch_pairs = remaining_pairs[start_idx:end_idx]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")

        # Process batch
        batch_results = process_gene_pairs_batch(
            batch_pairs, model, data_dict, n_samples, n_warmup
        )

        # Update results
        for i, pair in enumerate(batch_pairs):
            if batch_results[i] is not None:
                results[pair] = batch_results[i]
                completed_pairs.add(pair)

        # Save checkpoint
        checkpoint = {
            'completed_pairs': list(completed_pairs),
            'results': results,
            'timestamp': time.time()
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        logger.info(f"Checkpoint saved: {len(completed_pairs)} pairs completed")

    # Save final results as JSON
    logger.info(f"Saving final results to {output_file}")

    # Convert tuple keys to strings for JSON serialization
    results_json = {
        f"{k[0]}->{k[1]}": v for k, v in results.items()
    }

    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    # Remove checkpoint file
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint file removed")

    logger.info(f"Inference complete: {len(results)} gene pairs processed")

    return results


def export_network_to_cytoscape(
    network_results: Dict[Tuple[str, str], Dict],
    output_file: str,
    min_confidence: float = 0.5,
    min_delta_F: float = 1.0
):
    """
    Export network to Cytoscape-compatible format.

    Args:
        network_results: Inference results
        output_file: Path to save CSV file
        min_confidence: Minimum confidence to include edge
        min_delta_F: Minimum |ΔF| to include edge
    """
    edges = []

    for (gene1, gene2), result in network_results.items():
        # Filter by quality thresholds
        if result['confidence'] < min_confidence:
            continue
        if abs(result['delta_F']) < min_delta_F:
            continue
        if result['direction'] == 'undecided':
            continue

        # Determine source and target
        if gene1 in result['direction'].split(" -> ")[0]:
            source = gene1
            target = gene2
        else:
            source = gene2
            target = gene1

        edges.append({
            'source': source,
            'target': target,
            'delta_F': result['delta_F'],
            'confidence': result['confidence'],
            'F_forward': result['F_forward'],
            'F_backward': result['F_backward'],
            'prior_belief': result.get('prior_belief', 0.0)
        })

    # Create DataFrame and save
    df = pd.DataFrame(edges)
    df.to_csv(output_file, index=False)

    logger.info(f"Exported {len(edges)} edges to {output_file}")


def summarize_network(network_results: Dict[Tuple[str, str], Dict]) -> Dict[str, Any]:
    """
    Generate summary statistics for network inference results.

    Returns:
        Dict with statistics about network structure and quality
    """
    total_pairs = len(network_results)

    if total_pairs == 0:
        return {'error': 'No results to summarize'}

    # Extract metrics
    delta_Fs = [abs(r['delta_F']) for r in network_results.values()]
    confidences = [r['confidence'] for r in network_results.values()]

    # Count directions
    directions = [r['direction'] for r in network_results.values()]
    undecided = sum(1 for d in directions if d == 'undecided')
    decided = total_pairs - undecided

    # Count strong edges (|ΔF| > 2.0)
    strong_edges = sum(1 for df in delta_Fs if df > 2.0)

    # Count edges with prior support
    with_prior = sum(1 for r in network_results.values() if r.get('prior_belief', 0) > 0.5)

    summary = {
        'total_pairs_tested': total_pairs,
        'decided_directions': decided,
        'undecided_directions': undecided,
        'strong_edges': strong_edges,
        'edges_with_prior_support': with_prior,
        'mean_delta_F': float(np.mean(delta_Fs)),
        'median_delta_F': float(np.median(delta_Fs)),
        'std_delta_F': float(np.std(delta_Fs)),
        'mean_confidence': float(np.mean(confidences)),
        'median_confidence': float(np.median(confidences))
    }

    return summary


# Example usage
if __name__ == "__main__":
    # Example: Small network inference
    import pandas as pd

    logger.info("=" * 60)
    logger.info("EXAMPLE: Inference Pipeline Demonstration")
    logger.info("=" * 60)

    # Define genes
    genes = ["EGFR", "KRAS", "BRAF"]

    # Mock INDRA priors
    prior_network = {
        ("EGFR", "KRAS"): 0.9,
        ("KRAS", "BRAF"): 0.8,
        ("EGFR", "BRAF"): 0.5,
    }

    # Generate mock data (100 samples, 3 genes, 3 states)
    np.random.seed(42)
    n_samples_data = 100

    methylation_data = pd.DataFrame({
        gene: np.random.randint(0, 3, n_samples_data)
        for gene in genes
    })

    expression_data = pd.DataFrame({
        gene: np.random.randint(0, 3, n_samples_data)
        for gene in genes
    })

    logger.info(f"Mock data: {n_samples_data} samples, {len(genes)} genes")
    logger.info(f"Prior network: {len(prior_network)} edges")

    # Run inference (sequential for demo)
    logger.info("\n--- Running Network Inference ---")

    network_results = infer_network_structure(
        genes=genes,
        methylation_data=methylation_data,
        expression_data=expression_data,
        prior_network=prior_network,
        n_samples=100,  # Small for demo
        n_warmup=50,
        parallel=False  # Sequential for demo
    )

    # Display results
    logger.info("\n--- Inference Results ---")
    for (gene1, gene2), result in network_results.items():
        logger.info(
            f"{gene1} <-> {gene2}: {result['direction']} "
            f"(ΔF={result['delta_F']:.2f}, conf={result['confidence']:.3f})"
        )

    # Generate summary
    logger.info("\n--- Network Summary ---")
    summary = summarize_network(network_results)
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    # Test network comparison (simulate resistant cells)
    logger.info("\n--- Network Comparison Demo ---")

    # Simulate resistant network (flip some directions)
    network_resistant = {}
    for (g1, g2), result in network_results.items():
        # Copy result and modify
        resistant_result = result.copy()

        # Flip EGFR -> KRAS edge
        if (g1, g2) == ("EGFR", "KRAS"):
            resistant_result['delta_F'] *= -1.5  # Flip and strengthen
            resistant_result['direction'] = "KRAS -> EGFR"
            logger.info(f"Simulated flip: {g1}<->{g2} direction reversed")

        network_resistant[(g1, g2)] = resistant_result

    # Compare networks
    comparison = compare_networks(network_results, network_resistant, threshold=1.0)

    logger.info("\n--- Comparison Results ---")
    for change_type, edges in comparison.items():
        if edges:
            logger.info(f"{change_type}: {edges}")

    # Test checkpointing
    logger.info("\n--- Testing Checkpoint System ---")
    output_file = "/tmp/network_inference_results.json"

    data_dict = {
        'methylation': methylation_data,
        'expression': expression_data
    }

    results_with_checkpoint = run_inference_with_progress(
        genes=genes,
        data=data_dict,
        prior_network=prior_network,
        output_file=output_file,
        checkpoint_every=2,  # Checkpoint every 2 pairs
        n_samples=100,
        parallel=False
    )

    logger.info(f"Results saved to {output_file}")

    # Export to Cytoscape
    cytoscape_file = "/tmp/network_cytoscape.csv"
    export_network_to_cytoscape(
        results_with_checkpoint,
        cytoscape_file,
        min_confidence=0.3,
        min_delta_F=0.5
    )

    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE COMPLETE")
    logger.info("=" * 60)
