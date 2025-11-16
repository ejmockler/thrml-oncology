#!/bin/bash
# Quick Setup Script for GPU Cluster
# Run this after extracting thrml-demo-deployment.tar.gz on the cluster

set -e

echo "========================================================================"
echo "THRML Demo - GPU Cluster Setup"
echo "========================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1)
echo "  $PYTHON_VERSION"

# Check for NVIDIA GPUs
echo ""
echo "Checking for NVIDIA GPUs..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  ⚠ nvidia-smi not found (CPU mode)"
fi

echo ""
echo "Setting up environment..."
python3 setup_environment.py

if [ $? -ne 0 ]; then
    echo "✗ Setup failed"
    exit 1
fi

echo ""
echo "Activating environment..."
source venv/bin/activate

echo ""
echo "Running tests..."
python scripts/00_test_environment.py

if [ $? -ne 0 ]; then
    echo "✗ Tests failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✓ Setup Complete!"
echo "========================================================================"
echo ""
echo "Environment activated. Ready to run:"
echo ""
echo "  Quick test (5 genes, ~5-10 min):"
echo "    python scripts/02_run_inference.py --quick-test --synthetic-data"
echo ""
echo "  Full inference (15 genes, ~1-2 hours on 2× H100):"
echo "    python scripts/02_run_inference.py --genes 15 --samples 1000"
echo ""
echo "  Parallel on 2 GPUs:"
echo "    CUDA_VISIBLE_DEVICES=0 python scripts/02_run_inference.py \\"
echo "        --genes 15 --samples 1000 --pairs-start 0 --pairs-end 52 \\"
echo "        --output results/gpu0.pkl &"
echo "    CUDA_VISIBLE_DEVICES=1 python scripts/02_run_inference.py \\"
echo "        --genes 15 --samples 1000 --pairs-start 53 --pairs-end 105 \\"
echo "        --output results/gpu1.pkl &"
echo "    wait"
echo ""
echo "========================================================================"
