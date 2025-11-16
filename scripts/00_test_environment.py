#!/usr/bin/env python3
"""
Environment Test Script
Run this BEFORE starting the hackathon to verify everything works.
"""

import sys

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        import jax
        import jax.numpy as jnp
        print(f"✓ JAX {jax.__version__}")
        print(f"  Devices: {jax.devices()}")
    except ImportError as e:
        print(f"✗ JAX import failed: {e}")
        return False
    
    try:
        import thrml
        print(f"✓ THRML installed")
    except ImportError as e:
        print(f"✗ THRML import failed: {e}")
        return False
    
    try:
        import numpy, pandas, scipy, matplotlib, networkx, requests
        print("✓ Scientific stack installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False
    
    return True

def test_thrml_spin():
    """Test THRML SpinNode functionality"""
    print("\nTesting THRML SpinNode...")
    
    try:
        import jax
        import jax.numpy as jnp
        from thrml import SpinNode, Block

        # Create nodes (THRML 0.1.3 uses factory pattern - no args needed)
        nodes = [SpinNode() for _ in range(5)]
        block = Block(nodes)

        print("✓ SpinNode creation: OK")
        return True

    except Exception as e:
        print(f"✗ SpinNode test failed: {e}")
        return False

def test_thrml_categorical():
    """Test THRML CategoricalNode functionality"""
    print("\nTesting THRML CategoricalNode...")
    
    try:
        from thrml import CategoricalNode, Block

        # Create categorical nodes (THRML 0.1.3 uses factory pattern)
        nodes = [CategoricalNode() for _ in range(5)]
        block = Block(nodes)

        print("✓ CategoricalNode creation: OK")
        return True

    except Exception as e:
        print(f"✗ CategoricalNode test failed: {e}")
        return False

def test_thrml_sampling():
    """Test basic THRML sampling"""
    print("\nTesting THRML sampling...")
    
    try:
        import jax
        import jax.numpy as jnp
        from thrml import SpinNode, Block, SamplingSchedule

        # Simple test
        nodes = [SpinNode() for _ in range(5)]
        block = Block(nodes)

        # Note: Full sampling test requires more setup
        # This just verifies imports work
        schedule = SamplingSchedule(n_warmup=10, n_samples=100, steps_per_sample=1)

        print("✓ THRML sampling imports: OK")
        return True

    except Exception as e:
        print(f"✗ THRML sampling test failed: {e}")
        return False

def test_indra_api():
    """Test INDRA REST API connectivity"""
    print("\nTesting INDRA REST API...")

    try:
        import requests

        # Test query: EGFR phosphorylates something
        url = "http://api.indra.bio:8000/statements/from_agents"
        params = {
            "subject": "EGFR",
            "format": "json",
            "max_stmts": 5
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            stmt_count = len(data.get('statements', []))
            print(f"✓ INDRA API: OK (returned {stmt_count} statements)")
            return True
        else:
            print(f"⚠ INDRA API returned status {response.status_code}")
            print("  This is OK - can use synthetic data if INDRA is unavailable")
            return True  # Not a blocker - we have fallback

    except Exception as e:
        print(f"⚠ INDRA API test failed: {e}")
        print("  This is OK - can use synthetic data if INDRA is unavailable")
        return True  # Not a blocker - we have fallback

def test_jax_gpu():
    """Test JAX GPU availability"""
    print("\nTesting JAX GPU...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        
        if gpu_devices:
            print(f"✓ Found {len(gpu_devices)} GPU(s)")
            for i, dev in enumerate(gpu_devices):
                print(f"  GPU {i}: {dev}")
            
            # Test simple computation
            x = jnp.ones((1000, 1000))
            y = jnp.dot(x, x)
            _ = y.block_until_ready()
            
            print("✓ JAX GPU computation: OK")
            return True
        else:
            print("⚠ No GPUs found (CPU-only mode)")
            print("  Inference will be slower but should work")
            return True  # Not a failure, just slower
            
    except Exception as e:
        print(f"✗ JAX GPU test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("XTR-0 HACKATHON ENVIRONMENT TEST")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("SpinNode", test_thrml_spin),
        ("CategoricalNode", test_thrml_categorical),
        ("Sampling", test_thrml_sampling),
        ("INDRA API", test_indra_api),
        ("JAX GPU", test_jax_gpu),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Ready for hackathon!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Fix before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())
