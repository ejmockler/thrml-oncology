#!/usr/bin/env python3
"""
Test script for data_loader module
Run with: python3 scripts/test_data_loader.py
"""

import sys
sys.path.insert(0, '/Users/noot/Documents/thrml-cancer-decision-support')

from core.data_loader import (
    discretize_values,
    generate_synthetic_data,
    load_ccle_data,
    validate_data,
    prepare_model_input
)
import numpy as np
import pandas as pd


def run_all_tests():
    """Run comprehensive test suite for data_loader"""
    print("=" * 70)
    print("Data Loader Test Suite")
    print("=" * 70)

    passed = 0
    failed = 0

    # Test 1: Discretization
    print("\n[Test 1] Discretization")
    try:
        df = pd.DataFrame({
            'EGFR': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'KRAS': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        })
        disc = discretize_values(df)
        assert disc.shape == df.shape
        assert set(disc.values.flatten()) <= {0, 1, 2}
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 2: Synthetic data generation
    print("\n[Test 2] Synthetic data generation")
    try:
        genes = ['EGFR', 'KRAS', 'BRAF']
        data = generate_synthetic_data(genes, n_sensitive=10, n_resistant=10, seed=42)
        assert data['methylation'].shape == (20, 3)
        assert data['expression'].shape == (20, 3)
        assert len(data['cell_lines']) == 20
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 3: Data validation
    print("\n[Test 3] Data validation")
    try:
        validate_data(data)
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 4: Prepare model input
    print("\n[Test 4] Prepare model input")
    try:
        model_input = prepare_model_input(data, discretize=True)
        assert model_input['methylation_discrete'].max().max() <= 2
        assert model_input['expression_discrete'].max().max() <= 2
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 5: Ground truth structure
    print("\n[Test 5] Ground truth structure")
    try:
        gt = data['ground_truth']
        assert len(gt['edges']) > 0
        assert 'EGFR' in gt['methylation_expression']
        print(f"  Causal edges: {len(gt['edges'])}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
