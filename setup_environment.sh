#!/bin/bash
# Environment Setup Script for THRML Cancer Decision Support
# This script creates a clean venv, installs latest packages, and pins versions

set -e  # Exit on error

echo "======================================================================"
echo "THRML Cancer Decision Support - Environment Setup"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}✗ Python 3.10+ required. Found: Python $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
echo ""

# Check if venv exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment 'venv' already exists${NC}"
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf venv
    else
        echo "Using existing venv..."
    fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Using existing virtual environment${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip, setuptools, wheel
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded to $(pip --version | awk '{print $2}')${NC}"
echo ""

# Install from unpinned requirements
echo "======================================================================"
echo "Installing packages (latest compatible versions)..."
echo "======================================================================"
echo ""

if [ -f "requirements-unpinned.txt" ]; then
    echo "Using requirements-unpinned.txt..."
    pip install -r requirements-unpinned.txt
else
    echo -e "${YELLOW}⚠ requirements-unpinned.txt not found, using requirements.txt${NC}"
    pip install -r requirements.txt
fi

echo ""
echo -e "${GREEN}✓ All packages installed successfully${NC}"
echo ""

# Run quick import test
echo "======================================================================"
echo "Testing imports..."
echo "======================================================================"
echo ""

python3 << 'EOF'
import sys

packages = [
    ("thrml", "THRML"),
    ("jax", "JAX"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("scipy", "SciPy"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("networkx", "NetworkX"),
    ("requests", "Requests"),
]

optional_packages = [
    ("anthropic", "Anthropic"),
    ("tqdm", "tqdm"),
]

print("Core packages:")
all_ok = True
for module, name in packages:
    try:
        m = __import__(module)
        version = getattr(m, '__version__', 'unknown')
        print(f"  ✓ {name:15s} {version}")
    except ImportError as e:
        print(f"  ✗ {name:15s} FAILED: {e}")
        all_ok = False

print("\nOptional packages:")
for module, name in optional_packages:
    try:
        m = __import__(module)
        version = getattr(m, '__version__', 'unknown')
        print(f"  ✓ {name:15s} {version}")
    except ImportError:
        print(f"  - {name:15s} not installed (optional)")

if not all_ok:
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ Import test failed${NC}"
    echo "Some packages could not be imported. Check the errors above."
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All imports successful${NC}"
echo ""

# Test THRML basic functionality
echo "======================================================================"
echo "Testing THRML basic functionality..."
echo "======================================================================"
echo ""

python3 << 'EOF'
import jax
import jax.numpy as jnp
from thrml.base import CategoricalNode, Block

# Test CategoricalNode creation
try:
    node = CategoricalNode(n_categories=3, name="test_node")
    print("  ✓ CategoricalNode creation successful")
except Exception as e:
    print(f"  ✗ CategoricalNode failed: {e}")
    exit(1)

# Test Block creation
try:
    block = Block([node])
    print("  ✓ Block creation successful")
except Exception as e:
    print(f"  ✗ Block failed: {e}")
    exit(1)

# Test JAX GPU detection
devices = jax.devices()
print(f"  ✓ JAX detected {len(devices)} device(s):")
for i, device in enumerate(devices):
    print(f"    - Device {i}: {device.device_kind}")

if any(d.device_kind == 'gpu' for d in devices):
    print("  ✓ GPU acceleration available")
else:
    print("  ⚠ No GPU detected (CPU-only mode)")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ THRML test failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ THRML basic functionality verified${NC}"
echo ""

# Generate pinned requirements
echo "======================================================================"
echo "Generating pinned requirements..."
echo "======================================================================"
echo ""

pip freeze > requirements-pinned.txt

echo -e "${GREEN}✓ Pinned requirements saved to requirements-pinned.txt${NC}"
echo ""

# Show installed versions
echo "======================================================================"
echo "Installed Package Versions"
echo "======================================================================"
echo ""

python3 << 'EOF'
import thrml
import jax
import jaxlib
import numpy as np
import pandas as pd
import scipy
import matplotlib
import seaborn as sns
import networkx as nx
import requests

packages = [
    ("THRML", thrml.__version__),
    ("JAX", jax.__version__),
    ("JAXlib", jaxlib.__version__),
    ("NumPy", np.__version__),
    ("Pandas", pd.__version__),
    ("SciPy", scipy.__version__),
    ("Matplotlib", matplotlib.__version__),
    ("Seaborn", sns.__version__),
    ("NetworkX", nx.__version__),
    ("Requests", requests.__version__),
]

for name, version in packages:
    print(f"  {name:15s} {version}")

# Try optional packages
try:
    import anthropic
    print(f"  {'Anthropic':15s} {anthropic.__version__} (optional)")
except ImportError:
    pass

try:
    import tqdm
    print(f"  {'tqdm':15s} {tqdm.__version__} (optional)")
except ImportError:
    pass
EOF

echo ""
echo "======================================================================"
echo "Environment Setup Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the environment:"
echo "     ${GREEN}source venv/bin/activate${NC}"
echo ""
echo "  2. Run environment tests:"
echo "     ${GREEN}python scripts/00_test_environment.py${NC}"
echo ""
echo "  3. Run quick demo:"
echo "     ${GREEN}python scripts/04_live_demo.py --gene1 EGFR --gene2 KRAS${NC}"
echo ""
echo "  4. For future installations, use pinned versions:"
echo "     ${GREEN}pip install -r requirements-pinned.txt${NC}"
echo ""
echo "Environment details:"
echo "  - Python: $(python --version)"
echo "  - Virtual environment: $(pwd)/venv"
echo "  - Pinned requirements: $(pwd)/requirements-pinned.txt"
echo ""
echo -e "${GREEN}✓ Ready for thermodynamic computing!${NC}"
echo ""
