#!/usr/bin/env python3
"""
Cross-platform Environment Setup for THRML Cancer Decision Support
Creates venv, installs latest packages, tests imports, and pins versions.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

    @staticmethod
    def disable():
        """Disable colors on Windows if not supported"""
        if platform.system() == 'Windows':
            Colors.RED = Colors.GREEN = Colors.YELLOW = Colors.BLUE = Colors.NC = ''


def print_header(text):
    """Print section header"""
    print("=" * 70)
    print(text)
    print("=" * 70)
    print()


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")


def check_python_version():
    """Check if Python version is 3.10+"""
    print("Checking Python version...")
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_error(f"Python 3.10+ required. Found: Python {version.major}.{version.minor}.{version.micro}")
        return False

    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
    print()
    return True


def get_venv_python():
    """Get path to Python executable in venv"""
    if platform.system() == 'Windows':
        return Path('venv') / 'Scripts' / 'python.exe'
    else:
        return Path('venv') / 'bin' / 'python'


def get_venv_pip():
    """Get path to pip executable in venv"""
    if platform.system() == 'Windows':
        return Path('venv') / 'Scripts' / 'pip.exe'
    else:
        return Path('venv') / 'bin' / 'pip'


def create_venv(force=False):
    """Create virtual environment"""
    venv_path = Path('venv')

    if venv_path.exists():
        print_warning("Virtual environment 'venv' already exists")

        if force:
            print("Removing existing venv...")
            import shutil
            shutil.rmtree(venv_path)
        else:
            response = input("Remove and recreate? (y/n): ").strip().lower()
            if response == 'y':
                print("Removing existing venv...")
                import shutil
                shutil.rmtree(venv_path)
            else:
                print_success("Using existing virtual environment")
                print()
                return True

    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print_success("Virtual environment created")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False


def upgrade_pip():
    """Upgrade pip, setuptools, and wheel"""
    print("Upgrading pip, setuptools, and wheel...")

    pip_exe = str(get_venv_pip())

    try:
        subprocess.run(
            [pip_exe, 'install', '--upgrade', 'pip', 'setuptools', 'wheel'],
            check=True,
            capture_output=True
        )

        # Get pip version
        result = subprocess.run(
            [pip_exe, '--version'],
            check=True,
            capture_output=True,
            text=True
        )
        pip_version = result.stdout.split()[1]
        print_success(f"pip upgraded to {pip_version}")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to upgrade pip: {e}")
        return False


def install_requirements():
    """Install packages from requirements"""
    print_header("Installing packages (latest compatible versions)...")

    pip_exe = str(get_venv_pip())

    # Determine which requirements file to use
    if Path('requirements-unpinned.txt').exists():
        req_file = 'requirements-unpinned.txt'
        print("Using requirements-unpinned.txt...")
    else:
        req_file = 'requirements.txt'
        print_warning("requirements-unpinned.txt not found, using requirements.txt")

    try:
        subprocess.run(
            [pip_exe, 'install', '-r', req_file],
            check=True
        )
        print()
        print_success("All packages installed successfully")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install requirements: {e}")
        return False


def test_imports():
    """Test importing all required packages"""
    print_header("Testing imports...")

    python_exe = str(get_venv_python())

    test_script = """
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

print("\\nOptional packages:")
for module, name in optional_packages:
    try:
        m = __import__(module)
        version = getattr(m, '__version__', 'unknown')
        print(f"  ✓ {name:15s} {version}")
    except ImportError:
        print(f"  - {name:15s} not installed (optional)")

if not all_ok:
    sys.exit(1)
"""

    try:
        subprocess.run([python_exe, '-c', test_script], check=True)
        print()
        print_success("All imports successful")
        print()
        return True
    except subprocess.CalledProcessError:
        print()
        print_error("Import test failed")
        print("Some packages could not be imported. Check the errors above.")
        return False


def test_thrml():
    """Test THRML basic functionality"""
    print_header("Testing THRML basic functionality...")

    python_exe = str(get_venv_python())

    test_script = """
import jax
import jax.numpy as jnp
from thrml import CategoricalNode, Block

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
"""

    try:
        subprocess.run([python_exe, '-c', test_script], check=True)
        print()
        print_success("THRML basic functionality verified")
        print()
        return True
    except subprocess.CalledProcessError:
        print()
        print_error("THRML test failed")
        return False


def generate_pinned_requirements():
    """Generate pinned requirements file"""
    print_header("Generating pinned requirements...")

    pip_exe = str(get_venv_pip())

    try:
        result = subprocess.run(
            [pip_exe, 'freeze'],
            check=True,
            capture_output=True,
            text=True
        )

        with open('requirements-pinned.txt', 'w') as f:
            f.write("# Generated by setup_environment.py\n")
            f.write("# Install with: pip install -r requirements-pinned.txt\n\n")
            f.write(result.stdout)

        print_success("Pinned requirements saved to requirements-pinned.txt")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to generate pinned requirements: {e}")
        return False


def show_package_versions():
    """Display installed package versions"""
    print_header("Installed Package Versions")

    python_exe = str(get_venv_python())

    version_script = """
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
"""

    subprocess.run([python_exe, '-c', version_script])
    print()


def main():
    """Main setup process"""
    # Disable colors on Windows if needed
    if platform.system() == 'Windows' and not sys.stdout.isatty():
        Colors.disable()

    print_header("THRML Cancer Decision Support - Environment Setup")

    # Check Python version
    if not check_python_version():
        return 1

    # Create virtual environment
    if not create_venv():
        return 1

    # Upgrade pip
    if not upgrade_pip():
        return 1

    # Install requirements
    if not install_requirements():
        return 1

    # Test imports
    if not test_imports():
        return 1

    # Test THRML
    if not test_thrml():
        return 1

    # Generate pinned requirements
    if not generate_pinned_requirements():
        return 1

    # Show installed versions
    show_package_versions()

    # Success message
    print_header("Environment Setup Complete!")

    print("Next steps:")
    print()

    if platform.system() == 'Windows':
        print("  1. Activate the environment:")
        print(f"     {Colors.GREEN}venv\\Scripts\\activate{Colors.NC}")
    else:
        print("  1. Activate the environment:")
        print(f"     {Colors.GREEN}source venv/bin/activate{Colors.NC}")

    print()
    print("  2. Run environment tests:")
    print(f"     {Colors.GREEN}python scripts/00_test_environment.py{Colors.NC}")
    print()
    print("  3. Run quick demo:")
    print(f"     {Colors.GREEN}python scripts/04_live_demo.py --gene1 EGFR --gene2 KRAS{Colors.NC}")
    print()
    print("  4. For future installations, use pinned versions:")
    print(f"     {Colors.GREEN}pip install -r requirements-pinned.txt{Colors.NC}")
    print()

    print("Environment details:")
    print(f"  - Python: {sys.version.split()[0]}")
    print(f"  - Platform: {platform.system()} {platform.machine()}")
    print(f"  - Virtual environment: {Path('venv').absolute()}")
    print(f"  - Pinned requirements: {Path('requirements-pinned.txt').absolute()}")
    print()
    print_success("Ready for thermodynamic computing!")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
