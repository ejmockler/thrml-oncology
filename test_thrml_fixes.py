#!/usr/bin/env python3
"""
Test script to verify THRML model fixes.
Tests basic functionality without crashing.
"""

import sys
import traceback
import jax
import jax.numpy as jnp

# Set JAX to use CPU for testing
jax.config.update('jax_platform_name', 'cpu')

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        from core.thrml_model import GeneNetworkModel
        from thrml.pgm import CategoricalNode
        from thrml.block_management import Block, BlockSpec
        from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
        from thrml.factor import FactorSamplingProgram
        from thrml.block_sampling import sample_with_observation
        from thrml.observers import StateObserver
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test that we can create a GeneNetworkModel."""
    print("\nTesting model creation...")
    try:
        from core.thrml_model import GeneNetworkModel

        genes = ['EGFR', 'KRAS']
        priors = {('EGFR', 'KRAS'): 0.9}

        model = GeneNetworkModel(genes, priors)
        print(f"✓ Created model with {len(model.genes)} genes")
        print(f"  - Methylation nodes: {list(model.meth_nodes.keys())}")
        print(f"  - Expression nodes: {list(model.expr_nodes.keys())}")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_build_model_forward():
    """Test building forward model."""
    print("\nTesting build_model_forward...")
    try:
        from core.thrml_model import GeneNetworkModel

        genes = ['EGFR', 'KRAS']
        priors = {('EGFR', 'KRAS'): 0.9}
        model = GeneNetworkModel(genes, priors)

        factors, blocks = model.build_model_forward('EGFR', 'KRAS')

        print(f"✓ Built forward model")
        print(f"  - Number of factors: {len(factors)}")
        print(f"  - Number of blocks: {len(blocks)}")
        print(f"  - Block 0 (meth) has {len(blocks[0].nodes)} nodes")
        print(f"  - Block 1 (expr) has {len(blocks[1].nodes)} nodes")

        # Verify structure
        assert len(factors) == 3, f"Expected 3 factors, got {len(factors)}"
        assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}"
        assert len(blocks[0].nodes) == 2, "Expected 2 nodes in methylation block"
        assert len(blocks[1].nodes) == 2, "Expected 2 nodes in expression block"

        return True
    except Exception as e:
        print(f"✗ build_model_forward failed: {e}")
        traceback.print_exc()
        return False


def test_build_model_backward():
    """Test building backward model."""
    print("\nTesting build_model_backward...")
    try:
        from core.thrml_model import GeneNetworkModel

        genes = ['EGFR', 'KRAS']
        priors = {('KRAS', 'EGFR'): 0.8}
        model = GeneNetworkModel(genes, priors)

        factors, blocks = model.build_model_backward('EGFR', 'KRAS')

        print(f"✓ Built backward model")
        print(f"  - Number of factors: {len(factors)}")
        print(f"  - Number of blocks: {len(blocks)}")

        assert len(factors) == 3, f"Expected 3 factors, got {len(factors)}"
        assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}"

        return True
    except Exception as e:
        print(f"✗ build_model_backward failed: {e}")
        traceback.print_exc()
        return False


def test_sample_to_state():
    """Test sample to state conversion."""
    print("\nTesting _sample_to_state...")
    try:
        from core.thrml_model import GeneNetworkModel

        genes = ['EGFR', 'KRAS']
        model = GeneNetworkModel(genes, {})

        sample = jnp.array([0, 1, 2, 1])
        state = model._sample_to_state(sample, 'EGFR', 'KRAS')

        print(f"✓ Converted sample to state dict")
        print(f"  - Sample: {sample}")
        print(f"  - State: {state}")

        assert state['EGFR_meth'] == 0
        assert state['KRAS_meth'] == 1
        assert state['EGFR_expr'] == 2
        assert state['KRAS_expr'] == 1

        return True
    except Exception as e:
        print(f"✗ _sample_to_state failed: {e}")
        traceback.print_exc()
        return False


def test_compute_energy():
    """Test energy computation."""
    print("\nTesting compute_energy...")
    try:
        from core.thrml_model import GeneNetworkModel

        genes = ['EGFR', 'KRAS']
        priors = {('EGFR', 'KRAS'): 0.5}
        model = GeneNetworkModel(genes, priors)

        state = {
            'EGFR_meth': 0,
            'KRAS_meth': 1,
            'EGFR_expr': 2,
            'KRAS_expr': 1
        }

        energy = model.compute_energy('EGFR', 'KRAS', 'forward', state)

        print(f"✓ Computed energy")
        print(f"  - State: {state}")
        print(f"  - Energy: {energy:.4f}")

        assert not jnp.isnan(energy), "Energy is NaN"
        assert not jnp.isinf(energy), "Energy is infinite"

        return True
    except Exception as e:
        print(f"✗ compute_energy failed: {e}")
        traceback.print_exc()
        return False


def test_sampling():
    """Test sampling from model (quick test with minimal samples)."""
    print("\nTesting sample_from_model...")
    print("  (This may take a moment for JAX compilation...)")
    try:
        from core.thrml_model import GeneNetworkModel

        genes = ['EGFR', 'KRAS']
        priors = {('EGFR', 'KRAS'): 0.5}
        model = GeneNetworkModel(genes, priors)

        factors, blocks = model.build_model_forward('EGFR', 'KRAS')

        # Quick test with minimal samples
        samples = model.sample_from_model(factors, blocks, n_samples=10, n_warmup=5)

        print(f"✓ Sampling successful")
        print(f"  - Sample shape: {samples.shape}")
        print(f"  - Expected shape: (10, 4)")
        print(f"  - First sample: {samples[0]}")
        print(f"  - Value range: [{samples.min()}, {samples.max()}]")

        assert samples.shape == (10, 4), f"Expected shape (10, 4), got {samples.shape}"
        assert jnp.all(samples >= 0) and jnp.all(samples < 3), "Samples out of valid range [0, 3)"

        return True
    except Exception as e:
        print(f"✗ sample_from_model failed: {e}")
        traceback.print_exc()
        return False


def test_free_energy():
    """Test free energy computation."""
    print("\nTesting compute_free_energy...")
    try:
        from core.thrml_model import GeneNetworkModel

        genes = ['EGFR', 'KRAS']
        priors = {('EGFR', 'KRAS'): 0.5}
        model = GeneNetworkModel(genes, priors)

        # Create some mock samples
        samples = jnp.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [0, 1, 2, 1],
            [1, 0, 1, 2],
        ])

        F = model.compute_free_energy('EGFR', 'KRAS', 'forward', samples)

        print(f"✓ Computed free energy")
        print(f"  - Free energy: {F:.4f}")

        assert not jnp.isnan(F), "Free energy is NaN"
        assert not jnp.isinf(F), "Free energy is infinite"

        return True
    except Exception as e:
        print(f"✗ compute_free_energy failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("THRML Model Implementation Test Suite")
    print("=" * 60)

    tests = [
        test_imports,
        test_model_creation,
        test_build_model_forward,
        test_build_model_backward,
        test_sample_to_state,
        test_compute_energy,
        test_sampling,
        test_free_energy,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
