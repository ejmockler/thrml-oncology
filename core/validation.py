"""
Validation and Drug Prediction Module
Maps network changes to drug targets and validates predictions.
"""

import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
from core.indra_client import IndraClient
from core.thrml_model import GeneNetworkModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_drugs_from_changes(
    changed_edges: Dict,
    indra_client: IndraClient,
    top_n: int = 10
) -> List[Dict]:
    """
    Map changed edges to drug targets.

    Args:
        changed_edges: Output from compare_networks() with keys:
            'weakened': [(gene1, gene2, old_strength, new_strength, delta_F), ...]
            'lost': [(gene1, gene2, old_strength, delta_F), ...]
            'new': [(gene1, gene2, new_strength, delta_F), ...]
            'strengthened': [(gene1, gene2, old_strength, new_strength, delta_F), ...]
        indra_client: INDRA API client
        top_n: Number of top drugs to return

    Returns:
        List of dicts:
        [
            {
                'drug_name': str,
                'target_genes': [str],
                'mechanism': str,
                'confidence': float,
                'indra_belief': float,
                'mechanism_type': str,  # 'bypass_inhibitor' or 'pathway_modulator'
                'edge_context': str  # Description of the edge change
            }
        ]

    Strategy:
    1. For each weakened/lost edge (bypass mechanism):
       - Identify which gene is upregulated (resistance mechanism)
       - Query INDRA for drugs inhibiting that gene
       - These drugs should restore sensitivity by blocking bypass
    2. For each new/strengthened edge:
       - Identify alternative pathway activation
       - Find drugs targeting pathway nodes
       - These drugs target compensatory mechanisms
    3. Rank by: mechanism match × INDRA belief × |edge strength change|
    """
    logger.info("Predicting drugs from network changes...")

    # Collect all drug candidates
    drug_candidates = []

    # Process weakened/lost edges (bypass mechanisms)
    # These represent resistance mechanisms where signal bypasses blocked pathway
    bypass_edges = changed_edges.get('weakened', []) + changed_edges.get('lost', [])

    for edge_data in bypass_edges:
        if len(edge_data) == 5:  # Weakened: (gene1, gene2, old, new, delta_F)
            gene1, gene2, old_strength, new_strength, delta_F = edge_data
            strength_change = abs(old_strength - new_strength)
        else:  # Lost: (gene1, gene2, old, delta_F)
            gene1, gene2, old_strength, delta_F = edge_data
            new_strength = 0.0
            strength_change = old_strength

        # The weakened edge suggests gene2 is now being activated through bypass
        # We want to inhibit gene2 to restore sensitivity
        logger.info(f"Finding drugs targeting {gene2} (bypass via weakened {gene1}->{gene2})")

        drugs = indra_client.get_drug_targets(gene2)

        for drug in drugs:
            confidence = (
                drug['belief'] *  # INDRA evidence strength
                strength_change *  # How much the edge changed
                abs(delta_F)  # Confidence in causal direction change
            )

            drug_candidates.append({
                'drug_name': drug['drug_name'],
                'target_genes': [gene2],
                'mechanism': f"Inhibits {gene2} to block bypass pathway (weakened {gene1}->{gene2} edge)",
                'confidence': confidence,
                'indra_belief': drug['belief'],
                'mechanism_type': 'bypass_inhibitor',
                'edge_context': f"Weakened edge: {gene1}->{gene2} (Δstrength={strength_change:.3f}, ΔF={delta_F:.3f})"
            })

    # Process new/strengthened edges (compensatory mechanisms)
    # These represent new pathways activated in resistance
    compensatory_edges = changed_edges.get('new', []) + changed_edges.get('strengthened', [])

    for edge_data in compensatory_edges:
        if len(edge_data) == 5:  # Strengthened: (gene1, gene2, old, new, delta_F)
            gene1, gene2, old_strength, new_strength, delta_F = edge_data
            strength_change = abs(new_strength - old_strength)
        else:  # New: (gene1, gene2, new, delta_F)
            gene1, gene2, new_strength, delta_F = edge_data
            old_strength = 0.0
            strength_change = new_strength

        # New/strengthened edge suggests compensatory pathway
        # Target both genes in the new pathway
        logger.info(f"Finding drugs for compensatory pathway {gene1}->{gene2}")

        # Try to inhibit the downstream gene (direct effect)
        drugs_gene2 = indra_client.get_drug_targets(gene2)

        for drug in drugs_gene2:
            confidence = (
                drug['belief'] *
                strength_change *
                abs(delta_F) *
                0.8  # Slightly lower weight for compensatory vs bypass
            )

            drug_candidates.append({
                'drug_name': drug['drug_name'],
                'target_genes': [gene2],
                'mechanism': f"Inhibits {gene2} in compensatory pathway (new/strengthened {gene1}->{gene2})",
                'confidence': confidence,
                'indra_belief': drug['belief'],
                'mechanism_type': 'pathway_modulator',
                'edge_context': f"New/strengthened edge: {gene1}->{gene2} (Δstrength={strength_change:.3f}, ΔF={delta_F:.3f})"
            })

        # Also try upstream gene (indirect effect)
        drugs_gene1 = indra_client.get_drug_targets(gene1)

        for drug in drugs_gene1:
            confidence = (
                drug['belief'] *
                strength_change *
                abs(delta_F) *
                0.6  # Lower weight for upstream targeting
            )

            drug_candidates.append({
                'drug_name': drug['drug_name'],
                'target_genes': [gene1],
                'mechanism': f"Inhibits {gene1} to block compensatory {gene1}->{gene2} pathway",
                'confidence': confidence,
                'indra_belief': drug['belief'],
                'mechanism_type': 'pathway_modulator',
                'edge_context': f"New/strengthened edge: {gene1}->{gene2} (Δstrength={strength_change:.3f}, ΔF={delta_F:.3f})"
            })

    # Deduplicate and aggregate drugs
    drug_dict = defaultdict(lambda: {
        'target_genes': set(),
        'mechanisms': [],
        'confidences': [],
        'beliefs': [],
        'edge_contexts': [],
        'mechanism_types': set()
    })

    for candidate in drug_candidates:
        drug_name = candidate['drug_name']
        drug_dict[drug_name]['target_genes'].update(candidate['target_genes'])
        drug_dict[drug_name]['mechanisms'].append(candidate['mechanism'])
        drug_dict[drug_name]['confidences'].append(candidate['confidence'])
        drug_dict[drug_name]['beliefs'].append(candidate['indra_belief'])
        drug_dict[drug_name]['edge_contexts'].append(candidate['edge_context'])
        drug_dict[drug_name]['mechanism_types'].add(candidate['mechanism_type'])

    # Aggregate and rank
    final_drugs = []
    for drug_name, data in drug_dict.items():
        final_drugs.append({
            'drug_name': drug_name,
            'target_genes': sorted(list(data['target_genes'])),
            'mechanism': ' | '.join(set(data['mechanisms'])),
            'confidence': max(data['confidences']),  # Take best confidence
            'indra_belief': max(data['beliefs']),
            'mechanism_type': ', '.join(sorted(data['mechanism_types'])),
            'edge_context': ' | '.join(set(data['edge_contexts'])),
            'n_supporting_edges': len(data['mechanisms'])
        })

    # Sort by confidence
    final_drugs.sort(key=lambda x: x['confidence'], reverse=True)

    logger.info(f"Predicted {len(final_drugs)} unique drugs from {len(drug_candidates)} candidates")

    return final_drugs[:top_n]


def validate_predictions(
    predicted_drugs: List[Dict],
    ic50_data: Dict[str, float],
    threshold: float = 1.0  # IC50 threshold for "effective" in μM
) -> Dict:
    """
    Validate drug predictions against IC50 ground truth.

    Args:
        predicted_drugs: From predict_drugs_from_changes()
        ic50_data: {drug_name -> IC50 value in μM}
            Lower IC50 = more effective
            Typical effective range: 0.001 - 1.0 μM
        threshold: IC50 below this = effective (default 1.0 μM)

    Returns:
        {
            'precision': float,  # % of predictions that work (IC50 < threshold)
            'recall': float,  # % of effective drugs that were predicted
            'f1_score': float,
            'validated_drugs': [str],  # Drugs that passed validation
            'failed_drugs': [str],  # Drugs that failed validation
            'missing_drugs': [str],  # Drugs not in IC50 data
            'baseline_precision': float,  # Random selection rate
            'improvement_factor': float,  # How much better than random
            'mean_ic50_predicted': float,  # Mean IC50 of predicted drugs
            'mean_ic50_all': float  # Mean IC50 of all drugs in dataset
        }
    """
    logger.info(f"Validating {len(predicted_drugs)} predictions against IC50 data...")

    # Extract drug names from predictions
    predicted_names = [d['drug_name'] for d in predicted_drugs]

    # Categorize predictions
    validated_drugs = []  # Predicted AND effective
    failed_drugs = []  # Predicted but NOT effective
    missing_drugs = []  # Predicted but no IC50 data

    predicted_ic50s = []

    for drug_name in predicted_names:
        if drug_name in ic50_data:
            ic50 = ic50_data[drug_name]
            predicted_ic50s.append(ic50)

            if ic50 < threshold:
                validated_drugs.append(drug_name)
            else:
                failed_drugs.append(drug_name)
        else:
            missing_drugs.append(drug_name)

    # Calculate metrics
    n_predicted = len(predicted_names)
    n_with_data = len(validated_drugs) + len(failed_drugs)
    n_validated = len(validated_drugs)

    # Precision: Of drugs we predicted, how many are actually effective?
    precision = n_validated / n_with_data if n_with_data > 0 else 0.0

    # Recall: Of all effective drugs, how many did we predict?
    all_effective_drugs = [name for name, ic50 in ic50_data.items() if ic50 < threshold]
    n_effective_total = len(all_effective_drugs)
    recall = n_validated / n_effective_total if n_effective_total > 0 else 0.0

    # F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # Baseline: What if we randomly selected drugs?
    baseline_precision = n_effective_total / len(ic50_data) if len(ic50_data) > 0 else 0.0

    # Improvement over random
    improvement_factor = precision / baseline_precision if baseline_precision > 0 else float('inf')

    # Mean IC50 analysis
    mean_ic50_predicted = np.mean(predicted_ic50s) if predicted_ic50s else float('nan')
    mean_ic50_all = np.mean(list(ic50_data.values())) if ic50_data else float('nan')

    result = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'validated_drugs': validated_drugs,
        'failed_drugs': failed_drugs,
        'missing_drugs': missing_drugs,
        'baseline_precision': baseline_precision,
        'improvement_factor': improvement_factor,
        'mean_ic50_predicted': mean_ic50_predicted,
        'mean_ic50_all': mean_ic50_all,
        'n_predicted': n_predicted,
        'n_validated': n_validated,
        'n_effective_total': n_effective_total
    }

    logger.info(f"Validation results: Precision={precision:.1%}, Recall={recall:.1%}, F1={f1_score:.3f}")
    logger.info(f"Improvement over random: {improvement_factor:.2f}x")

    return result


def bootstrap_confidence(
    gene1: str,
    gene2: str,
    model: GeneNetworkModel,
    data: Dict[str, jnp.ndarray],
    n_bootstrap: int = 100,
    n_samples: int = 500
) -> Dict:
    """
    Estimate confidence in causal direction via bootstrap resampling.

    Args:
        gene1: Source gene
        gene2: Target gene
        model: GeneNetworkModel instance
        data: Observed methylation + expression data
            Keys: '{gene}_meth', '{gene}_expr'
            Values: Arrays of observations
        n_bootstrap: Number of bootstrap iterations
        n_samples: Number of samples per free energy estimate

    Returns:
        {
            'mean_delta_F': float,  # Mean ΔF across bootstrap samples
            'std_delta_F': float,  # Standard deviation of ΔF
            'confidence_interval': (float, float),  # 95% CI
            'p_value': float,  # Probability direction is wrong (ΔF crosses 0)
            'bootstrap_delta_Fs': [float]  # All bootstrap ΔF values
        }
    """
    logger.info(f"Computing bootstrap confidence for {gene1} -> {gene2}...")

    # Get data sizes
    meth1_data = data.get(f'{gene1}_meth', jnp.array([]))
    expr1_data = data.get(f'{gene1}_expr', jnp.array([]))
    meth2_data = data.get(f'{gene2}_meth', jnp.array([]))
    expr2_data = data.get(f'{gene2}_expr', jnp.array([]))

    n_observations = min(len(meth1_data), len(expr1_data), len(meth2_data), len(expr2_data))

    if n_observations == 0:
        logger.warning("No data available for bootstrap")
        return {
            'mean_delta_F': 0.0,
            'std_delta_F': 0.0,
            'confidence_interval': (0.0, 0.0),
            'p_value': 1.0,
            'bootstrap_delta_Fs': []
        }

    bootstrap_delta_Fs = []

    for i in range(n_bootstrap):
        # Resample data with replacement
        indices = np.random.choice(n_observations, size=n_observations, replace=True)

        resampled_data = {
            f'{gene1}_meth': meth1_data[indices],
            f'{gene1}_expr': expr1_data[indices],
            f'{gene2}_meth': meth2_data[indices],
            f'{gene2}_expr': expr2_data[indices],
        }

        # Test causal direction on resampled data
        try:
            result = model.test_causal_direction(gene1, gene2, resampled_data, n_samples=n_samples)
            delta_F = result['delta_F']
            bootstrap_delta_Fs.append(delta_F)
        except Exception as e:
            logger.warning(f"Bootstrap iteration {i} failed: {e}")
            continue

        if (i + 1) % 20 == 0:
            logger.info(f"Bootstrap progress: {i+1}/{n_bootstrap}")

    if not bootstrap_delta_Fs:
        logger.warning("All bootstrap iterations failed")
        return {
            'mean_delta_F': 0.0,
            'std_delta_F': 0.0,
            'confidence_interval': (0.0, 0.0),
            'p_value': 1.0,
            'bootstrap_delta_Fs': []
        }

    # Compute statistics
    mean_delta_F = float(np.mean(bootstrap_delta_Fs))
    std_delta_F = float(np.std(bootstrap_delta_Fs))

    # 95% confidence interval
    ci_lower = float(np.percentile(bootstrap_delta_Fs, 2.5))
    ci_upper = float(np.percentile(bootstrap_delta_Fs, 97.5))

    # P-value: fraction of bootstrap samples with opposite sign
    if mean_delta_F > 0:
        p_value = float(np.mean([df < 0 for df in bootstrap_delta_Fs]))
    else:
        p_value = float(np.mean([df > 0 for df in bootstrap_delta_Fs]))

    result = {
        'mean_delta_F': mean_delta_F,
        'std_delta_F': std_delta_F,
        'confidence_interval': (ci_lower, ci_upper),
        'p_value': p_value,
        'bootstrap_delta_Fs': bootstrap_delta_Fs
    }

    logger.info(f"Bootstrap complete: ΔF = {mean_delta_F:.3f} ± {std_delta_F:.3f}, p = {p_value:.3f}")

    return result


def summarize_results(
    network_sensitive: Dict,
    network_resistant: Dict,
    changed_edges: Dict,
    predicted_drugs: List[Dict],
    validation: Dict
) -> Dict:
    """
    Create comprehensive results summary.

    Args:
        network_sensitive: Network structure for sensitive cell line
            Keys: 'edges' (list of (gene1, gene2, strength, delta_F))
        network_resistant: Network structure for resistant cell line
        changed_edges: Dict with 'weakened', 'lost', 'new', 'strengthened' edges
        predicted_drugs: List of predicted drugs from predict_drugs_from_changes()
        validation: Validation results from validate_predictions()

    Returns:
        {
            'n_genes': int,
            'n_edges_sensitive': int,
            'n_edges_resistant': int,
            'n_edges_tested': int,
            'n_edges_changed': int,
            'change_breakdown': {
                'weakened': int,
                'lost': int,
                'new': int,
                'strengthened': int
            },
            'n_drugs_predicted': int,
            'precision': float,
            'recall': float,
            'f1_score': float,
            'baseline_precision': float,
            'improvement_factor': float,
            'top_predictions': [Dict],  # Top 5 drugs with details
            'mechanism_breakdown': {
                'bypass_inhibitor': int,
                'pathway_modulator': int
            }
        }
    """
    logger.info("Summarizing analysis results...")

    # Extract gene count
    all_genes = set()
    for edge in network_sensitive.get('edges', []):
        all_genes.add(edge[0])
        all_genes.add(edge[1])
    for edge in network_resistant.get('edges', []):
        all_genes.add(edge[0])
        all_genes.add(edge[1])

    n_genes = len(all_genes)
    n_edges_sensitive = len(network_sensitive.get('edges', []))
    n_edges_resistant = len(network_resistant.get('edges', []))

    # Count changed edges
    n_weakened = len(changed_edges.get('weakened', []))
    n_lost = len(changed_edges.get('lost', []))
    n_new = len(changed_edges.get('new', []))
    n_strengthened = len(changed_edges.get('strengthened', []))
    n_edges_changed = n_weakened + n_lost + n_new + n_strengthened

    # All edges that were tested
    n_edges_tested = max(n_edges_sensitive, n_edges_resistant)

    # Mechanism breakdown
    mechanism_counts = defaultdict(int)
    for drug in predicted_drugs:
        mech_types = drug.get('mechanism_type', '').split(', ')
        for mech in mech_types:
            if mech:
                mechanism_counts[mech] += 1

    # Top predictions (top 5)
    top_predictions = predicted_drugs[:5]

    # Format top predictions for readability
    formatted_top = []
    for i, drug in enumerate(top_predictions, 1):
        formatted_top.append({
            'rank': i,
            'drug_name': drug['drug_name'],
            'target_genes': drug['target_genes'],
            'confidence': round(drug['confidence'], 4),
            'indra_belief': round(drug['indra_belief'], 3),
            'mechanism_type': drug['mechanism_type'],
            'n_supporting_edges': drug.get('n_supporting_edges', 1)
        })

    summary = {
        'n_genes': n_genes,
        'n_edges_sensitive': n_edges_sensitive,
        'n_edges_resistant': n_edges_resistant,
        'n_edges_tested': n_edges_tested,
        'n_edges_changed': n_edges_changed,
        'change_breakdown': {
            'weakened': n_weakened,
            'lost': n_lost,
            'new': n_new,
            'strengthened': n_strengthened
        },
        'n_drugs_predicted': len(predicted_drugs),
        'precision': validation.get('precision', 0.0),
        'recall': validation.get('recall', 0.0),
        'f1_score': validation.get('f1_score', 0.0),
        'baseline_precision': validation.get('baseline_precision', 0.0),
        'improvement_factor': validation.get('improvement_factor', 0.0),
        'top_predictions': formatted_top,
        'mechanism_breakdown': dict(mechanism_counts)
    }

    logger.info(f"Summary complete: {n_edges_changed} edges changed, {len(predicted_drugs)} drugs predicted, "
                f"{validation.get('precision', 0):.1%} precision")

    return summary


# Mock IC50 data for testing
MOCK_IC50_DATA = {
    # EGFR inhibitors (effective against EGFR-driven cancers)
    'Erlotinib': 0.002,  # Highly effective
    'Gefitinib': 0.003,
    'Afatinib': 0.001,
    'Osimertinib': 0.005,

    # MEK inhibitors
    'Trametinib': 0.008,
    'Selumetinib': 0.012,
    'Cobimetinib': 0.015,

    # BRAF inhibitors
    'Vemurafenib': 0.010,
    'Dabrafenib': 0.007,

    # PI3K/AKT/mTOR pathway
    'Alpelisib': 0.020,
    'Everolimus': 0.030,
    'Temsirolimus': 0.040,

    # MET inhibitors
    'Crizotinib': 0.018,
    'Cabozantinib': 0.025,

    # HER2 inhibitors
    'Lapatinib': 0.009,
    'Neratinib': 0.011,

    # Multi-kinase inhibitors (less specific, higher IC50)
    'Sorafenib': 0.150,
    'Sunitinib': 0.120,
    'Pazopanib': 0.180,

    # Ineffective/off-target drugs (IC50 > 1.0 μM)
    'DrugX': 2.5,
    'DrugY': 5.0,
    'DrugZ': 10.0,
}


# Example usage
if __name__ == "__main__":
    from core.indra_client import IndraClient
    from core.thrml_model import GeneNetworkModel
    import jax.numpy as jnp

    logger.info("Testing validation module...")

    # Mock network comparison results
    changed_edges = {
        'weakened': [
            ('EGFR', 'KRAS', 0.9, 0.3, 2.5),  # EGFR->KRAS weakened
            ('KRAS', 'BRAF', 0.8, 0.4, 1.8),
        ],
        'lost': [
            ('EGFR', 'AKT1', 0.7, 1.5),  # EGFR->AKT1 lost
        ],
        'new': [
            ('MET', 'KRAS', 0.6, 2.0),  # New MET->KRAS bypass
        ],
        'strengthened': [
            ('PI3K', 'AKT1', 0.5, 0.9, 1.2),  # PI3K->AKT strengthened
        ]
    }

    # Initialize INDRA client
    indra_client = IndraClient()

    # Predict drugs
    logger.info("\n=== DRUG PREDICTION ===")
    predicted_drugs = predict_drugs_from_changes(
        changed_edges=changed_edges,
        indra_client=indra_client,
        top_n=10
    )

    print(f"\nPredicted {len(predicted_drugs)} drugs:")
    for i, drug in enumerate(predicted_drugs[:5], 1):
        print(f"\n{i}. {drug['drug_name']}")
        print(f"   Targets: {', '.join(drug['target_genes'])}")
        print(f"   Mechanism: {drug['mechanism']}")
        print(f"   Confidence: {drug['confidence']:.4f}")
        print(f"   Type: {drug['mechanism_type']}")

    # Validate predictions
    logger.info("\n=== VALIDATION ===")
    validation = validate_predictions(
        predicted_drugs=predicted_drugs,
        ic50_data=MOCK_IC50_DATA,
        threshold=1.0
    )

    print(f"\nValidation Results:")
    print(f"  Precision: {validation['precision']:.1%}")
    print(f"  Recall: {validation['recall']:.1%}")
    print(f"  F1 Score: {validation['f1_score']:.3f}")
    print(f"  Baseline (random): {validation['baseline_precision']:.1%}")
    print(f"  Improvement: {validation['improvement_factor']:.2f}x")
    print(f"  Validated drugs: {', '.join(validation['validated_drugs'][:5])}")

    # Bootstrap confidence (using mock model)
    logger.info("\n=== BOOTSTRAP CONFIDENCE ===")
    genes = ['EGFR', 'KRAS', 'BRAF']
    mock_priors = {('EGFR', 'KRAS'): 0.9}
    model = GeneNetworkModel(genes, mock_priors)

    # Mock data
    mock_data = {
        'EGFR_meth': jnp.array([1, 1, 2, 0, 1, 2]),
        'EGFR_expr': jnp.array([1, 2, 2, 0, 1, 2]),
        'KRAS_meth': jnp.array([0, 1, 1, 0, 1, 1]),
        'KRAS_expr': jnp.array([0, 1, 2, 0, 2, 2]),
    }

    # Note: This will use the TODO-marked methods in thrml_model.py
    # Uncomment when those are implemented
    # bootstrap_result = bootstrap_confidence(
    #     gene1='EGFR',
    #     gene2='KRAS',
    #     model=model,
    #     data=mock_data,
    #     n_bootstrap=20,
    #     n_samples=100
    # )
    # print(f"\nBootstrap Results:")
    # print(f"  Mean ΔF: {bootstrap_result['mean_delta_F']:.3f} ± {bootstrap_result['std_delta_F']:.3f}")
    # print(f"  95% CI: ({bootstrap_result['confidence_interval'][0]:.3f}, {bootstrap_result['confidence_interval'][1]:.3f})")
    # print(f"  P-value: {bootstrap_result['p_value']:.3f}")

    # Results summary
    logger.info("\n=== RESULTS SUMMARY ===")
    mock_network_sensitive = {
        'edges': [('EGFR', 'KRAS', 0.9, 2.5), ('KRAS', 'BRAF', 0.8, 1.8)]
    }
    mock_network_resistant = {
        'edges': [('EGFR', 'KRAS', 0.3, 0.5), ('MET', 'KRAS', 0.6, 2.0)]
    }

    summary = summarize_results(
        network_sensitive=mock_network_sensitive,
        network_resistant=mock_network_resistant,
        changed_edges=changed_edges,
        predicted_drugs=predicted_drugs,
        validation=validation
    )

    print(f"\nAnalysis Summary:")
    print(f"  Genes analyzed: {summary['n_genes']}")
    print(f"  Edges tested: {summary['n_edges_tested']}")
    print(f"  Edges changed: {summary['n_edges_changed']}")
    print(f"    - Weakened: {summary['change_breakdown']['weakened']}")
    print(f"    - Lost: {summary['change_breakdown']['lost']}")
    print(f"    - New: {summary['change_breakdown']['new']}")
    print(f"    - Strengthened: {summary['change_breakdown']['strengthened']}")
    print(f"  Drugs predicted: {summary['n_drugs_predicted']}")
    print(f"  Precision: {summary['precision']:.1%} ({summary['improvement_factor']:.2f}x better than random)")
    print(f"\n  Top 3 Predictions:")
    for pred in summary['top_predictions'][:3]:
        print(f"    {pred['rank']}. {pred['drug_name']} (confidence={pred['confidence']:.4f})")

    logger.info("\nValidation module testing complete!")
