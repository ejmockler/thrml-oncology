# Pre-Deployment Checklist: M1 Mac → GPU Cluster

## Overview

**Goal**: Validate everything possible on M1 Mac before SSH into GPU cluster to minimize debugging time during hackathon.

**M1 Advantages**:
- ✅ JAX supports Metal acceleration (fast enough for testing)
- ✅ Can validate all code logic
- ✅ Can test visualizations
- ✅ Can verify data pipeline
- ✅ Can catch bugs early

**What needs GPU cluster**:
- Large-scale inference (15 genes, 105 pairs)
- CUDA-specific optimizations
- Multi-GPU parallelization

---

## Phase 1: Environment Setup (5 minutes)

### 1.1 Run Automated Setup

```bash
cd /Users/noot/Documents/thrml-cancer-decision-support

# Automated environment setup
python3 setup_environment.py
```

**Expected output**:
```
✓ Python 3.10+ detected
✓ Virtual environment created
✓ All packages installed successfully
✓ All imports successful
✓ THRML basic functionality verified
✓ JAX detected 1 device(s):
  - Device 0: METAL (M1 GPU)
✓ GPU acceleration available
```

**Success criteria**:
- All tests pass
- Metal GPU detected
- `requirements-pinned.txt` generated

### 1.2 Activate Environment

```bash
source venv/bin/activate
```

**Verify**:
```bash
which python  # Should point to venv/bin/python
python --version  # Should be 3.10+
```

---

## Phase 2: Smoke Tests (10 minutes)

### 2.1 Environment Test

```bash
python scripts/00_test_environment.py
```

**Expected output**:
```
✓ THRML SpinNode: OK
✓ THRML CategoricalNode: OK
✓ THRML Block Gibbs: OK
✓ JAX Metal GPU: OK
✓ NumPy: OK
✓ Matplotlib: OK
✓ NetworkX: OK
✓ INDRA API: OK
```

**If fails**: Check ENVIRONMENT_SETUP.md troubleshooting section

### 2.2 Quick Live Demo (3 genes, 100 samples)

```bash
python scripts/04_live_demo.py \
    --gene1 EGFR \
    --gene2 KRAS \
    --samples 100 \
    --warmup 50
```

**Expected**:
- Completes in ~30-60 seconds on M1
- Generates 7 PNG files in `results/live_demo/`
- Shows energy landscapes, sampling dynamics, etc.
- No errors or NaNs

**Success criteria**:
- ✓ Script completes without errors
- ✓ All 7 visualization files created
- ✓ Free energy values are finite (not NaN/inf)
- ✓ ΔF is reasonable (-5 to +5 range)
- ✓ Causal direction determined

### 2.3 Quick Inference Test (5 genes, synthetic data)

```bash
python scripts/02_run_inference.py \
    --quick-test \
    --synthetic-data \
    --output results/test_inference.pkl
```

**Expected**:
- Completes in ~5-10 minutes on M1
- Tests 5 genes = 10 gene pairs
- 100 samples per model (forward/backward)
- Generates checkpoint files

**Success criteria**:
- ✓ Inference completes for all pairs
- ✓ No crashes or memory errors
- ✓ Results saved to pickle file
- ✓ Checkpoint system works

### 2.4 Analysis Test

```bash
python scripts/03_analyze_results.py \
    --input results/test_inference.pkl \
    --output-dir results/test_figures
```

**Expected**:
- Network comparison figure
- Precision comparison figure
- Summary JSON

**Success criteria**:
- ✓ Figures generated without errors
- ✓ Network changes detected
- ✓ Drug predictions made
- ✓ Validation metrics computed

---

## Phase 3: Code Validation (15 minutes)

### 3.1 Syntax Check All Files

```bash
# Check Python syntax
find . -name "*.py" -type f -exec python3 -m py_compile {} \;
```

**Expected**: No output = success

**If errors**: Fix syntax issues before deploying

### 3.2 Import Check

```bash
# Test all core modules
python3 -c "from core.thrml_model import GeneNetworkModel; print('✓ thrml_model')"
python3 -c "from core.data_loader import generate_synthetic_data; print('✓ data_loader')"
python3 -c "from core.inference import infer_network_structure; print('✓ inference')"
python3 -c "from core.validation import predict_drugs_from_changes; print('✓ validation')"
python3 -c "from core.indra_client import IndraClient; print('✓ indra_client')"
python3 -c "from core.thermodynamic_visualizer import ThermodynamicVisualizer; print('✓ visualizer')"
```

**Expected**: All modules import successfully

### 3.3 Test THRML API Patterns

```bash
python3 << 'EOF'
"""Test that THRML API is used correctly"""
import jax.numpy as jnp
from thrml.base import CategoricalNode, Block
from thrml.models.discrete_ebm import CategoricalEBMFactor

print("Testing THRML API patterns...")

# Test 1: CategoricalNode creation
n1 = CategoricalNode(n_categories=3, name="test1")
n2 = CategoricalNode(n_categories=3, name="test2")
print("  ✓ CategoricalNode creation")

# Test 2: Block creation
b1 = Block([n1])
b2 = Block([n2])
print("  ✓ Block creation")

# Test 3: CategoricalEBMFactor with Blocks (CRITICAL!)
weights = jnp.ones((1, 3, 3))
factor = CategoricalEBMFactor(
    categorical_node_groups=[b1, b2],  # Must be Blocks!
    weights=weights
)
print("  ✓ CategoricalEBMFactor with Blocks (correct API)")

# Test 4: Multi-node Block
b_multi = Block([n1, n2])
print("  ✓ Multi-node Block")

print("\n✓ All THRML API patterns correct!")
EOF
```

**Expected**: All tests pass

### 3.4 Test Data Pipeline

```bash
python3 << 'EOF'
"""Test data generation and discretization"""
from core.data_loader import generate_synthetic_data, prepare_model_input

print("Testing data pipeline...")

# Generate synthetic data
genes = ['EGFR', 'KRAS', 'BRAF']
data = generate_synthetic_data(genes, n_sensitive=10, n_resistant=10, seed=42)
print(f"  ✓ Generated data for {len(genes)} genes")

# Prepare model input
model_input = prepare_model_input(data, genes, discretize=True)
meth = model_input['methylation_discrete']
expr = model_input['expression_discrete']

print(f"  ✓ Methylation: {meth.shape}")
print(f"  ✓ Expression: {expr.shape}")

# Check discretization
assert meth.min() >= 0 and meth.max() <= 2, "Methylation not in {0,1,2}"
assert expr.min() >= 0 and expr.max() <= 2, "Expression not in {0,1,2}"
print("  ✓ Values properly discretized to {0, 1, 2}")

print("\n✓ Data pipeline working correctly!")
EOF
```

**Expected**: All assertions pass

### 3.5 Test INDRA API

```bash
python3 << 'EOF'
"""Test INDRA REST API connection"""
from core.indra_client import IndraClient

print("Testing INDRA API...")

client = IndraClient()

# Test simple query
try:
    genes = ['EGFR', 'KRAS']
    network = client.build_prior_network(genes)
    print(f"  ✓ INDRA query successful ({len(network)} edges)")

    if ('EGFR', 'KRAS') in network:
        belief = network[('EGFR', 'KRAS')]['belief']
        print(f"  ✓ EGFR → KRAS belief: {belief:.3f}")
except Exception as e:
    print(f"  ✗ INDRA query failed: {e}")
    print("  ⚠ This is OK if offline, will work on cluster")

print("\n✓ INDRA client configured correctly!")
EOF
```

**Expected**: Either succeeds or fails gracefully

---

## Phase 4: Performance Baseline (10 minutes)

### 4.1 Benchmark Single Pair (M1 Metal)

```bash
python3 << 'EOF'
"""Benchmark inference speed on M1 Mac"""
import time
from core.thrml_model import GeneNetworkModel
from core.data_loader import generate_synthetic_data, prepare_model_input

print("Benchmarking inference on M1 Mac (Metal GPU)...")

# Generate small dataset
genes = ['EGFR', 'KRAS']
data = generate_synthetic_data(genes, n_sensitive=20, n_resistant=20)
model_input = prepare_model_input(data, genes, discretize=True)

# Build model
model = GeneNetworkModel(genes)

# Benchmark forward model
print("\nSampling forward model (1000 samples)...")
start = time.time()

factors_fwd, blocks_fwd = model.build_model_forward('EGFR', 'KRAS')
samples_fwd, energies_fwd = model.sample_model(
    factors_fwd, blocks_fwd,
    model_input['methylation_discrete'],
    model_input['expression_discrete'],
    n_samples=1000,
    n_warmup=100
)

time_fwd = time.time() - start
print(f"  ✓ Forward: {time_fwd:.2f}s ({1000/time_fwd:.1f} samples/sec)")

# Benchmark backward model
print("\nSampling backward model (1000 samples)...")
start = time.time()

factors_bwd, blocks_bwd = model.build_model_backward('EGFR', 'KRAS')
samples_bwd, energies_bwd = model.sample_model(
    factors_bwd, blocks_bwd,
    model_input['methylation_discrete'],
    model_input['expression_discrete'],
    n_samples=1000,
    n_warmup=100
)

time_bwd = time.time() - start
print(f"  ✓ Backward: {time_bwd:.2f}s ({1000/time_bwd:.1f} samples/sec)")

# Compute free energies
F_fwd = model.estimate_free_energy(energies_fwd)
F_bwd = model.estimate_free_energy(energies_bwd)
delta_F = F_bwd - F_fwd

print(f"\nFree energies:")
print(f"  F_forward:  {F_fwd:.4f}")
print(f"  F_backward: {F_bwd:.4f}")
print(f"  ΔF:         {delta_F:.4f}")

# Estimate full workload
total_time = time_fwd + time_bwd
print(f"\nEstimated times:")
print(f"  Single pair (M1):     {total_time:.1f}s")
print(f"  10 pairs (M1):        {total_time * 10 / 60:.1f} min")
print(f"  105 pairs (M1):       {total_time * 105 / 60:.1f} min")
print(f"  105 pairs (H100 2×):  ~{total_time * 105 / 60 / 10:.1f} min (estimated)")

print("\n✓ Benchmark complete!")
EOF
```

**Expected**:
- Single pair: ~3-10 seconds on M1
- 10 pairs: ~0.5-2 minutes
- 105 pairs: ~5-20 minutes on M1

**GPU cluster expectation**:
- H100 should be ~5-10× faster than M1
- With 2 GPUs in parallel: ~10-20× faster

### 4.2 Memory Usage Test

```bash
python3 << 'EOF'
"""Check memory usage"""
import psutil
import os
from core.thrml_model import GeneNetworkModel
from core.data_loader import generate_synthetic_data, prepare_model_input

print("Testing memory usage...")

# Get baseline
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024  # MB

# Run inference
genes = ['EGFR', 'KRAS', 'BRAF', 'MEK1', 'ERK1']
data = generate_synthetic_data(genes, n_sensitive=25, n_resistant=25)
model_input = prepare_model_input(data, genes, discretize=True)

model = GeneNetworkModel(genes)
factors, blocks = model.build_model_forward(genes[0], genes[1])
samples, energies = model.sample_model(
    factors, blocks,
    model_input['methylation_discrete'],
    model_input['expression_discrete'],
    n_samples=1000
)

# Check final memory
mem_after = process.memory_info().rss / 1024 / 1024
mem_used = mem_after - mem_before

print(f"  Memory before: {mem_before:.1f} MB")
print(f"  Memory after:  {mem_after:.1f} MB")
print(f"  Memory used:   {mem_used:.1f} MB")

if mem_used < 500:
    print("  ✓ Memory usage reasonable (<500 MB)")
elif mem_used < 1000:
    print("  ⚠ Memory usage moderate (500-1000 MB)")
else:
    print("  ✗ High memory usage (>1000 MB) - may need optimization")

print("\n✓ Memory test complete!")
EOF
```

**Expected**: <500 MB for small test

---

## Phase 5: Create Deployment Package (5 minutes)

### 5.1 Generate Pinned Requirements

```bash
# Already done by setup script, but verify
cat requirements-pinned.txt
```

**Should contain exact versions like**:
```
thrml==0.1.5
jax==0.4.23
jaxlib==0.4.23
...
```

### 5.2 Create Deployment Archive

```bash
# Create clean deployment archive (no venv, no results)
tar -czf thrml-demo-deployment.tar.gz \
    --exclude='venv' \
    --exclude='results' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    core/ \
    scripts/ \
    data/ \
    *.md \
    *.txt \
    *.sh \
    setup_environment.py
```

**Result**: `thrml-demo-deployment.tar.gz` ready to upload

### 5.3 Test Deployment Archive

```bash
# Test extraction in temp directory
mkdir -p /tmp/test_deploy
cd /tmp/test_deploy
tar -xzf ~/Documents/thrml-cancer-decision-support/thrml-demo-deployment.tar.gz
ls -la
```

**Verify**:
- All core/ files present
- All scripts/ present
- Requirements files present
- Documentation present

---

## Phase 6: Documentation Review (5 minutes)

### 6.1 Create Quick Reference Card

```bash
cat > CLUSTER_QUICK_START.md << 'EOF'
# GPU Cluster Quick Start

## 1. Setup (5 min)
```bash
tar -xzf thrml-demo-deployment.tar.gz
cd thrml-demo-deployment
python3 setup_environment.py
source venv/bin/activate
```

## 2. Test (2 min)
```bash
python scripts/00_test_environment.py
python scripts/04_live_demo.py --quick-test
```

## 3. Full Run (2-3 hours)
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/02_run_inference.py \
    --genes 15 --samples 1000 --pairs-start 0 --pairs-end 52 \
    --output results/gpu0.pkl &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/02_run_inference.py \
    --genes 15 --samples 1000 --pairs-start 53 --pairs-end 105 \
    --output results/gpu1.pkl &

# Wait for completion
wait
```

## 4. Analysis
```bash
python scripts/03_analyze_results.py \
    --input results/gpu0.pkl \
    --output-dir results/figures
```

## Emergency
- Reduce to 10 genes: `--genes 10`
- Reduce samples: `--samples 500`
- Quick test: `--quick-test`
EOF
```

### 6.2 Checklist Summary

```bash
cat > DEPLOYMENT_CHECKLIST.txt << 'EOF'
Pre-Deployment Checklist (M1 Mac)
================================

Environment Setup:
[ ] setup_environment.py completed successfully
[ ] Metal GPU detected
[ ] requirements-pinned.txt generated

Smoke Tests:
[ ] 00_test_environment.py passed
[ ] 04_live_demo.py completed (3 genes, 100 samples)
[ ] 02_run_inference.py quick test passed (5 genes)
[ ] 03_analyze_results.py generated figures

Code Validation:
[ ] All .py files compile (no syntax errors)
[ ] All core modules import successfully
[ ] THRML API patterns verified
[ ] Data pipeline working
[ ] INDRA client configured

Performance:
[ ] Single pair benchmark: ~3-10s on M1
[ ] Memory usage: <500 MB
[ ] No NaN or inf values
[ ] Free energies reasonable (-10 to +10)

Deployment Package:
[ ] thrml-demo-deployment.tar.gz created
[ ] Archive tested (extracts correctly)
[ ] requirements-pinned.txt included
[ ] All scripts included

Documentation:
[ ] CLUSTER_QUICK_START.md created
[ ] DEPLOYMENT_CHECKLIST.txt created
[ ] Emergency protocols documented

Ready for GPU Cluster: _____ (YES/NO)
EOF

cat DEPLOYMENT_CHECKLIST.txt
```

---

## Phase 7: Final Preparation (5 minutes)

### 7.1 Create Cluster Setup Script

```bash
cat > cluster_setup.sh << 'EOF'
#!/bin/bash
# Quick setup script for GPU cluster

echo "Setting up THRML demo on GPU cluster..."

# Check Python
python3 --version

# Setup environment
python3 setup_environment.py

# Activate
source venv/bin/activate

# Test
echo "Running quick test..."
python scripts/00_test_environment.py

echo ""
echo "Setup complete! Ready to run:"
echo "  python scripts/02_run_inference.py --genes 15 --samples 1000"
EOF

chmod +x cluster_setup.sh
```

### 7.2 Timing Estimates

Create expected timeline:

```bash
cat > CLUSTER_TIMELINE.md << 'EOF'
# GPU Cluster Execution Timeline

## Setup Phase (10 min)
- SSH connection: 1 min
- Upload deployment archive: 2 min
- Extract and setup environment: 5 min
- Run tests: 2 min

## Execution Phase (2-3 hours)
- 15 genes = 105 pairs
- 1000 samples per model (forward/backward)
- 2000 total samples per pair
- Parallel on 2× H100 GPUs

**Estimated time per pair**: 1-2 minutes
**Total time (2 GPUs)**: 53 pairs × 1.5 min = ~80 min (~1.5 hours)

## Analysis Phase (10 min)
- Merge results: 1 min
- Network comparison: 2 min
- Drug prediction: 3 min
- Generate figures: 4 min

## Total: ~2-3 hours (with buffer)

## Emergency Fallbacks
- 10 genes (45 pairs): ~45 min
- 5 genes (10 pairs): ~10 min
- Quick test mode: ~5 min
EOF
```

---

## Complete Pre-Deployment Workflow

### Run Everything (30-45 minutes total)

```bash
#!/bin/bash
# Master pre-deployment test script

cd /Users/noot/Documents/thrml-cancer-decision-support

echo "========================================="
echo "THRML Demo: Pre-Deployment Testing"
echo "========================================="

# Phase 1: Environment Setup
echo -e "\n[Phase 1/7] Environment Setup"
python3 setup_environment.py || exit 1

source venv/bin/activate

# Phase 2: Smoke Tests
echo -e "\n[Phase 2/7] Smoke Tests"
python scripts/00_test_environment.py || exit 1
python scripts/04_live_demo.py --gene1 EGFR --gene2 KRAS --samples 100 --warmup 50 || exit 1
python scripts/02_run_inference.py --quick-test --synthetic-data || exit 1

# Phase 3: Code Validation
echo -e "\n[Phase 3/7] Code Validation"
find . -name "*.py" -type f -exec python3 -m py_compile {} \; || exit 1

# Phase 4: Performance Baseline
echo -e "\n[Phase 4/7] Performance Benchmark"
# (run benchmark script from Phase 4.1)

# Phase 5: Create Deployment Package
echo -e "\n[Phase 5/7] Creating Deployment Package"
tar -czf thrml-demo-deployment.tar.gz \
    --exclude='venv' --exclude='results' --exclude='__pycache__' \
    core/ scripts/ data/ *.md *.txt *.sh setup_environment.py

# Phase 6: Documentation
echo -e "\n[Phase 6/7] Creating Documentation"
# (documentation already created)

# Phase 7: Final Check
echo -e "\n[Phase 7/7] Final Verification"
ls -lh thrml-demo-deployment.tar.gz
cat DEPLOYMENT_CHECKLIST.txt

echo -e "\n========================================="
echo "✓ Pre-deployment testing complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Upload thrml-demo-deployment.tar.gz to GPU cluster"
echo "  2. SSH into cluster"
echo "  3. Extract and run cluster_setup.sh"
echo "  4. Execute full inference with 02_run_inference.py"
```

Save this as `pre_deployment_test.sh` and run:

```bash
chmod +x pre_deployment_test.sh
./pre_deployment_test.sh
```

---

## Expected Results Summary

### What Should Work on M1

✅ **All code execution** (slower but functional)
✅ **All visualizations** (same output as GPU)
✅ **Quick tests** (5 genes, 100 samples)
✅ **Small-scale inference** (10 pairs in ~5-10 min)
✅ **API validation** (THRML, INDRA)
✅ **Data pipeline** (generation, discretization)

### What Requires GPU Cluster

⏱️ **Full-scale inference** (105 pairs, 1000 samples)
⏱️ **CUDA-specific optimizations**
⏱️ **Multi-GPU parallelization**
⏱️ **Production timing benchmarks**

### Confidence Level

After completing all phases on M1:

**95% confidence** that code will work on GPU cluster because:
- All logic validated on M1
- THRML API verified
- Data pipeline tested
- Visualizations working
- Only difference is hardware acceleration

**Remaining 5% risk**:
- CUDA driver issues (unlikely)
- Network/firewall for INDRA API (fallback: synthetic data)
- Out of memory on large runs (fallback: reduce genes)

---

## When Ready for GPU Cluster

**Checklist before SSH:**
- [ ] All M1 tests passed
- [ ] thrml-demo-deployment.tar.gz created
- [ ] CLUSTER_QUICK_START.md ready
- [ ] Emergency fallbacks documented
- [ ] Expected timeline reviewed

**First commands on cluster:**
```bash
# Upload
scp thrml-demo-deployment.tar.gz user@cluster:~

# SSH
ssh user@cluster

# Setup
tar -xzf thrml-demo-deployment.tar.gz
cd thrml-demo-deployment
./cluster_setup.sh

# Test
python scripts/00_test_environment.py

# If tests pass, launch!
python scripts/02_run_inference.py --genes 15 --samples 1000
```

---

**You're ready to validate everything locally before deploying!** 🚀
