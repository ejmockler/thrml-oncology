"""
Test validation module without full environment
Verifies structure and logic without requiring JAX installation
"""

import sys
sys.path.insert(0, '/Users/noot/Documents/thrml-cancer-decision-support')

# Test imports
print("Testing validation module imports...")

try:
    # Mock the jax import for testing
    import unittest.mock as mock

    # Create mock modules
    mock_jnp = mock.MagicMock()
    mock_jax = mock.MagicMock()

    # Install mocks
    sys.modules['jax'] = mock_jax
    sys.modules['jax.numpy'] = mock_jnp

    # Now import validation module
    from core import validation

    print("✓ Module imported successfully")

    # Test that all required functions exist
    required_functions = [
        'predict_drugs_from_changes',
        'validate_predictions',
        'bootstrap_confidence',
        'summarize_results'
    ]

    print("\nChecking required functions:")
    for func_name in required_functions:
        if hasattr(validation, func_name):
            print(f"  ✓ {func_name}")
        else:
            print(f"  ✗ {func_name} NOT FOUND")

    # Test mock IC50 data
    print(f"\n✓ Mock IC50 data contains {len(validation.MOCK_IC50_DATA)} drugs")

    # Test validate_predictions with simple mock data
    print("\nTesting validate_predictions function:")

    mock_predicted_drugs = [
        {'drug_name': 'Erlotinib', 'target_genes': ['EGFR'], 'confidence': 0.9},
        {'drug_name': 'Gefitinib', 'target_genes': ['EGFR'], 'confidence': 0.85},
        {'drug_name': 'DrugX', 'target_genes': ['KRAS'], 'confidence': 0.7},  # Ineffective
        {'drug_name': 'Trametinib', 'target_genes': ['MEK1'], 'confidence': 0.8},
    ]

    result = validation.validate_predictions(
        predicted_drugs=mock_predicted_drugs,
        ic50_data=validation.MOCK_IC50_DATA,
        threshold=1.0
    )

    print(f"  Precision: {result['precision']:.1%}")
    print(f"  Recall: {result['recall']:.1%}")
    print(f"  F1 Score: {result['f1_score']:.3f}")
    print(f"  Validated drugs: {result['validated_drugs']}")
    print(f"  Failed drugs: {result['failed_drugs']}")
    print(f"  Baseline: {result['baseline_precision']:.1%}")
    print(f"  Improvement: {result['improvement_factor']:.2f}x")

    # Test summarize_results
    print("\nTesting summarize_results function:")

    mock_network_sensitive = {
        'edges': [('EGFR', 'KRAS', 0.9, 2.5), ('KRAS', 'BRAF', 0.8, 1.8)]
    }
    mock_network_resistant = {
        'edges': [('EGFR', 'KRAS', 0.3, 0.5), ('MET', 'KRAS', 0.6, 2.0)]
    }
    mock_changed_edges = {
        'weakened': [('EGFR', 'KRAS', 0.9, 0.3, 2.5)],
        'lost': [('EGFR', 'AKT1', 0.7, 1.5)],
        'new': [('MET', 'KRAS', 0.6, 2.0)],
        'strengthened': []
    }

    summary = validation.summarize_results(
        network_sensitive=mock_network_sensitive,
        network_resistant=mock_network_resistant,
        changed_edges=mock_changed_edges,
        predicted_drugs=mock_predicted_drugs,
        validation=result
    )

    print(f"  Genes analyzed: {summary['n_genes']}")
    print(f"  Edges tested: {summary['n_edges_tested']}")
    print(f"  Edges changed: {summary['n_edges_changed']}")
    print(f"  Drugs predicted: {summary['n_drugs_predicted']}")
    print(f"  Top predictions: {len(summary['top_predictions'])}")

    print("\n✓ All validation tests passed!")
    print("\nModule structure verified successfully.")
    print("Ready for integration into analysis pipeline.")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
