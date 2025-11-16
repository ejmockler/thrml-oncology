# Environment Setup Guide

## Quick Start

### Option 1: Automated Setup (Recommended)

**On macOS/Linux:**
```bash
./setup_environment.sh
```

**On any platform (including Windows):**
```bash
python3 setup_environment.py
```

This will:
1. ✅ Check Python version (3.10+ required)
2. ✅ Create virtual environment (`venv/`)
3. ✅ Install latest compatible package versions
4. ✅ Test all imports
5. ✅ Verify THRML functionality
6. ✅ Generate pinned requirements (`requirements-pinned.txt`)

**Total time**: ~2-5 minutes depending on internet speed

---

## What Gets Installed

### Core Dependencies

**THRML Stack:**
- `thrml` - Thermodynamic computing library for Extropic hardware
- `jax` - Accelerated array computing
- `jaxlib` - JAX backend with GPU support

**Scientific Computing:**
- `numpy` - Numerical arrays
- `pandas` - Data manipulation
- `scipy` - Scientific algorithms

**Visualization:**
- `matplotlib` - Plotting library
- `seaborn` - Statistical visualization
- `networkx` - Network/graph visualization

**Utilities:**
- `requests` - HTTP library for INDRA API calls

### Optional Dependencies

- `anthropic` - For LLM-based drug interpretation (optional)
- `tqdm` - Progress bars (optional)

---

## Manual Setup

If you prefer manual control:

### Step 1: Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 2: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Packages

**Option A: Latest versions (recommended for new setup)**
```bash
pip install -r requirements-unpinned.txt
```

**Option B: Pinned versions (for reproducibility)**
```bash
pip install -r requirements-pinned.txt
```

### Step 4: Test Installation

```bash
python scripts/00_test_environment.py
```

### Step 5: Generate Pinned Requirements

```bash
pip freeze > requirements-pinned.txt
```

---

## Requirements Files Explained

### `requirements-unpinned.txt`

Contains package names **without version pins**:
```
thrml
jax
jaxlib
numpy
...
```

**Use when:**
- Setting up fresh environment
- Want latest compatible versions
- Updating dependencies

### `requirements-pinned.txt`

Contains exact versions from `pip freeze`:
```
thrml==0.1.5
jax==0.4.23
jaxlib==0.4.23
numpy==1.26.3
...
```

**Use when:**
- Reproducing exact environment
- Deploying to production
- Sharing with collaborators

### `requirements.txt`

Contains **minimum version constraints**:
```
thrml>=0.1.3
jax>=0.4.20
jaxlib>=0.4.20
numpy>=1.24.0
...
```

**Use when:**
- Want flexibility with updates
- Specifying minimum requirements

---

## Version Strategy

### Why Unpin First?

**Problem with old pins:**
- Dependencies evolve rapidly
- Security patches released
- Performance improvements
- Bug fixes

**Our approach:**
1. Install latest compatible versions (unpinned)
2. Test thoroughly
3. Pin working versions
4. Use pinned versions for deployment

### Recommended Update Cycle

**Every 3 months:**
```bash
# Test with latest versions
pip install -r requirements-unpinned.txt

# Run full test suite
python scripts/00_test_environment.py
python scripts/04_live_demo.py --quick-test

# If successful, update pins
pip freeze > requirements-pinned.txt
```

---

## Platform-Specific Notes

### macOS

**Installation:**
```bash
# Ensure Python 3.10+ installed
python3 --version

# Run setup
./setup_environment.sh
```

**GPU Support:**
- M1/M2/M3 Macs: JAX supports Metal acceleration
- Intel Macs: CPU-only (still fast)

### Linux

**Installation:**
```bash
# Ubuntu/Debian: Install Python 3.10+
sudo apt update
sudo apt install python3.10 python3.10-venv

# Run setup
./setup_environment.sh
```

**GPU Support:**
- CUDA GPUs: Install NVIDIA drivers + CUDA toolkit
- JAX will auto-detect GPU

### Windows

**Installation:**
```powershell
# Ensure Python 3.10+ installed
python --version

# Run setup
python setup_environment.py
```

**GPU Support:**
- CUDA GPUs: Install NVIDIA drivers
- JAX GPU support on Windows requires WSL2

---

## Troubleshooting

### Issue: Python version too old

**Error:**
```
✗ Python 3.10+ required. Found: Python 3.9.x
```

**Solution:**
```bash
# Install Python 3.10+ from python.org
# Or use pyenv:
pyenv install 3.10.13
pyenv local 3.10.13
```

### Issue: JAX not finding GPU

**Symptoms:**
```
⚠ No GPU detected (CPU-only mode)
```

**Solutions:**

**For NVIDIA GPUs:**
```bash
# Install CUDA-enabled JAX
pip install --upgrade "jax[cuda12]"
```

**For Mac Metal:**
```bash
# Install Metal-enabled JAX (M1/M2/M3)
pip install --upgrade "jax[metal]"
```

**For TPU:**
```bash
# Install TPU-enabled JAX
pip install --upgrade "jax[tpu]"
```

### Issue: THRML import fails

**Error:**
```
ModuleNotFoundError: No module named 'thrml'
```

**Solutions:**

1. **Check venv is activated:**
   ```bash
   which python  # Should point to venv/bin/python
   ```

2. **Install THRML explicitly:**
   ```bash
   pip install thrml
   ```

3. **Check Python version:**
   ```bash
   python --version  # Must be 3.10+
   ```

### Issue: Matplotlib backend errors

**Error:**
```
RuntimeError: Failed to initialize platform
```

**Solution:**
```bash
# Set non-interactive backend
export MPLBACKEND=Agg

# Or add to script:
import matplotlib
matplotlib.use('Agg')
```

### Issue: Out of memory

**Symptoms:**
- Script crashes during sampling
- "Out of memory" errors

**Solutions:**

1. **Reduce samples:**
   ```bash
   python scripts/04_live_demo.py --samples 100
   ```

2. **Reduce genes:**
   ```bash
   python scripts/02_run_inference.py --genes 5
   ```

3. **Enable JAX memory preallocation:**
   ```bash
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   ```

---

## Verification

### Quick Test

```bash
python -c "import thrml; print(f'THRML {thrml.__version__} OK')"
```

**Expected output:**
```
THRML 0.1.5 OK
```

### Full Test Suite

```bash
python scripts/00_test_environment.py
```

**Expected output:**
```
✓ THRML SpinNode: OK
✓ THRML CategoricalNode: OK
✓ THRML Block Gibbs: OK
✓ JAX GPU: OK (or CPU)
✓ NumPy: OK
✓ INDRA API: OK
```

### Live Demo Test

```bash
python scripts/04_live_demo.py --gene1 EGFR --gene2 KRAS
```

**Expected:**
- Completes in ~10 seconds
- Generates 7 PNG files in `results/live_demo/`
- No errors or warnings

---

## Environment Comparison

### Development Environment

**Purpose:** Daily development, testing, experimentation

**Setup:**
```bash
pip install -r requirements-unpinned.txt
pip install -e .  # Editable install
```

**Characteristics:**
- Latest package versions
- Fast iteration
- May have minor breaking changes

### Production Environment

**Purpose:** Hackathon demo, deployment, reproducibility

**Setup:**
```bash
pip install -r requirements-pinned.txt
```

**Characteristics:**
- Exact versions pinned
- Guaranteed reproducibility
- Tested and validated

### Testing Environment

**Purpose:** CI/CD, validation, compatibility testing

**Setup:**
```bash
pip install -r requirements.txt  # Minimum versions
```

**Characteristics:**
- Tests backward compatibility
- Ensures minimum requirements work
- Catches version-specific issues

---

## Best Practices

### 1. Always Use Virtual Environments

**Why:**
- Isolation from system packages
- Reproducible environments
- Easy cleanup (just delete venv/)

**Never:**
```bash
pip install -r requirements.txt  # ✗ Global install
```

**Always:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # ✓ Isolated install
```

### 2. Pin Production Dependencies

**Development:**
```bash
pip install -r requirements-unpinned.txt
```

**Production:**
```bash
pip install -r requirements-pinned.txt
```

### 3. Test After Updates

**After updating any package:**
```bash
python scripts/00_test_environment.py
python scripts/04_live_demo.py --quick-test
```

### 4. Commit Pinned Requirements

**Git workflow:**
```bash
# After successful testing
pip freeze > requirements-pinned.txt
git add requirements-pinned.txt
git commit -m "Update pinned dependencies"
```

---

## GPU Configuration

### NVIDIA CUDA Setup

**1. Check GPU:**
```bash
nvidia-smi
```

**2. Install CUDA toolkit:**
```bash
# Ubuntu
sudo apt install nvidia-cuda-toolkit

# Check version
nvcc --version
```

**3. Install JAX with CUDA:**
```bash
pip install --upgrade "jax[cuda12]"  # For CUDA 12.x
# or
pip install --upgrade "jax[cuda11]"  # For CUDA 11.x
```

**4. Verify:**
```python
import jax
print(jax.devices())  # Should show GPU
```

### Apple Silicon (M1/M2/M3)

**1. Install Metal-enabled JAX:**
```bash
pip install --upgrade "jax[metal]"
```

**2. Verify:**
```python
import jax
print(jax.devices())  # Should show METAL
```

**Performance:**
- M1 Max: ~2× faster than CPU
- M2 Ultra: ~4× faster than CPU
- M3 Max: ~3× faster than CPU

### CPU-Only Mode

**If no GPU available:**

JAX automatically falls back to CPU. Performance is still good for:
- Quick tests (`--samples 100`)
- Small networks (`--genes 5`)
- Development and debugging

**For production:**
- Use cloud GPUs (Google Colab, AWS, etc.)
- Reduce problem size
- Increase patience 😊

---

## Docker Alternative

### Using Docker (Advanced)

**Create Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements-pinned.txt .
RUN pip install -r requirements-pinned.txt

# Copy code
COPY . .

# Run demo
CMD ["python", "scripts/04_live_demo.py", "--gene1", "EGFR", "--gene2", "KRAS"]
```

**Build and run:**
```bash
docker build -t thrml-demo .
docker run -v $(pwd)/results:/app/results thrml-demo
```

**With GPU (NVIDIA):**
```bash
docker run --gpus all -v $(pwd)/results:/app/results thrml-demo
```

---

## Summary

**Automated Setup:**
```bash
# One command
python3 setup_environment.py

# Then activate
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

**Manual Setup:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-unpinned.txt
pip freeze > requirements-pinned.txt
```

**Verification:**
```bash
python scripts/00_test_environment.py
python scripts/04_live_demo.py --quick-test
```

**For Production:**
```bash
pip install -r requirements-pinned.txt
```

---

## Support

**Issues:** See troubleshooting section above

**Questions:** Check THRML docs at https://docs.extropic.ai

**Updates:** Run setup script periodically to refresh dependencies

---

**Ready for thermodynamic computing!** 🔥
