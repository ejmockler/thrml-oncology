# Engineering Philosophy: Thermodynamic Computing for Biomedical Devices

**Project**: Cancer Drug Response Prediction via Thermodynamic Causal Inference
**Date**: November 16, 2024
**Context**: XTR-0 Hackathon Demo → Production Biomedical Device Foundation

---

## Core Distinction

### What We Are NOT Building

❌ **Academic Prototype**
- One-off analysis scripts
- "Works on my machine" code
- Hardcoded file paths
- Jupyter notebooks with manual execution
- CSV files passed around via email
- "Just run it again" error handling
- Documentation as afterthought

❌ **Bioinformatics Cruft**
- Pipeline held together with shell scripts and prayers
- Undocumented R functions from 2012
- "See the paper for methods" code comments
- Magic numbers scattered throughout
- Data preprocessing as art, not engineering
- Results that can't be reproduced 6 months later

### What We ARE Building

✅ **Foundation for Production Biomedical Device**
- **Thermodynamic computing** as physical reality, not statistical heuristic
- **Deterministic reproducibility** built into architecture
- **Clinical-grade reliability** from day one
- **Regulatory-ready** data provenance and audit trails
- **Hardware-aware** design (TSU simulation → future ASIC deployment)
- **Medical device mindset**: failures must be impossible, not just unlikely

---

## Physical Truth Foundation

### Thermodynamics ≠ Statistics

**Traditional bioinformatics**:
```python
# Correlation analysis (no causality)
correlation = pearson(gene_A, gene_B)
if correlation > 0.7:
    print("genes are related")  # Maybe? Who knows?
```

**Thermodynamic computing**:
```python
# Free energy discrimination (physical causality)
ΔF = F(B|A) - F(A|B)  # Actual thermodynamic quantity
if ΔF < 0:
    # A → B is thermodynamically favored
    # This is PHYSICS, not a p-value
    causal_direction = "A causes B"
```

**Why this matters**:
- Thermodynamic quantities are **objective physical observables**
- Not subject to multiple testing correction
- Not dependent on sample size (beyond statistical sampling error)
- **Ground truth**: Energy minimization is how nature actually works

### TSU Hardware Basis

**We are simulating real hardware**:
```
THRML library → TSU instruction set → (Future) Thermodynamic silicon
     ↓                 ↓                        ↓
Software sim    Hardware spec          Physical device
```

**This means**:
1. **Discrete states** (CategoricalNode: {0, 1, 2}) map to physical qubits/spins
2. **Energy functions** correspond to actual Hamiltonian terms in hardware
3. **Sampling** will run on thermodynamic annealing circuits (not Monte Carlo forever)
4. **Performance** improvements translate directly to ASIC efficiency gains

**Engineering implication**: Every design decision should be **hardware-conscious**
- Minimize state space (fewer physical qubits needed)
- Sparse factor graphs (fewer physical couplings)
- Efficient energy function encoding (faster annealing)

---

## Production Engineering Standards

### 1. Deterministic Reproducibility

**Every result must be exactly reproducible**:

```python
# BAD (academic prototype)
np.random.seed(42)  # "42 worked for me"
samples = model.sample(1000)

# GOOD (production device)
from hashlib import sha256

def deterministic_seed(data_hash: str, model_version: str, run_id: str) -> int:
    """
    Generate cryptographically-determined random seed.

    Ensures: Same data + same model + same run_id → identical results
    Enables: Regulatory audit trails, clinical validation
    """
    combined = f"{data_hash}:{model_version}:{run_id}"
    return int(sha256(combined.encode()).hexdigest(), 16) % (2**32)

data_hash = hash_dataset(expression_data, methylation_data)
seed = deterministic_seed(data_hash, model_version="1.0.0", run_id="patient_12345")
rng = np.random.default_rng(seed)
samples = model.sample(10000, rng=rng)
```

**Result**: Given same inputs, get **bit-exact** same outputs, forever.

### 2. Data Provenance and Audit Trails

**Medical device requirement**: Know exactly where every data point came from.

```python
# BAD
expr_data = pd.read_csv("data.csv")

# GOOD
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass(frozen=True)
class DataProvenance:
    """Immutable record of data origin."""
    source_file: Path
    file_hash: str  # SHA-256 of file contents
    download_date: datetime
    source_url: str
    depmap_release: str
    file_size_bytes: int
    row_count: int
    column_count: int

    def verify(self) -> bool:
        """Verify file hasn't changed."""
        current_hash = sha256(self.source_file.read_bytes()).hexdigest()
        return current_hash == self.file_hash

provenance = DataProvenance(
    source_file=Path("data/raw/ccle/CCLE_expression_TPM.csv"),
    file_hash="a3f2e9...",  # Actual SHA-256
    download_date=datetime(2024, 11, 16, 14, 38),
    source_url="https://depmap.org/portal/download/...",
    depmap_release="25Q3",
    file_size_bytes=543252480,
    row_count=1755,
    column_count=19221
)

# Later: Verify data hasn't been tampered with
assert provenance.verify(), "Data integrity check failed"
```

### 3. Physics-Based Validation

**Academic**: "p < 0.05, therefore significant"
**Production**: "Violates thermodynamic constraints, therefore impossible"

```python
# Example: Validate methylation → expression causality
def validate_causal_direction(M_g: np.ndarray, E_g: np.ndarray,
                               min_delta_F: float = -1.0) -> bool:
    """
    Validate that methylation → expression shows thermodynamic preference.

    Physical constraint: If M causes E, then:
        ΔF = F(E|M) - F(M|E) < 0  (forward direction favored)

    Returns:
        True if causal direction is thermodynamically valid
        False if physics is violated (indicates model error)
    """
    delta_F = compute_free_energy_difference(M_g, E_g)

    if delta_F >= 0:
        raise PhysicsViolationError(
            f"Methylation → Expression shows ΔF = {delta_F:.2f} ≥ 0. "
            f"This violates known biology (methylation represses expression). "
            f"Check model configuration or data quality."
        )

    if delta_F > min_delta_F:
        warnings.warn(
            f"Weak causal signal: ΔF = {delta_F:.2f} (threshold: {min_delta_F})"
        )

    return True
```

**No statistical tests** → **Physical consistency checks**

### 4. Fail-Fast, Fail-Loud

**Academic**: Silent failures, results look reasonable, paper gets published, irreproducible
**Production**: Crash immediately with actionable error messages

```python
# BAD
try:
    data = load_data(filepath)
except:
    data = None  # Hope for the best!

# GOOD
class DataIntegrityError(Exception):
    """Raised when data fails integrity checks."""
    pass

def load_expression_data(filepath: Path) -> pd.DataFrame:
    """
    Load CCLE expression data with comprehensive validation.

    Raises:
        FileNotFoundError: File doesn't exist
        DataIntegrityError: File corrupted or wrong format
        ValueError: Data outside expected biological range
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Expression data not found at {filepath}.\n"
            f"Expected file: CCLE_expression_TPM.csv\n"
            f"Download from: https://depmap.org/portal/download/\n"
            f"See: DEPMAP_REQUIRED_FILES.md for instructions"
        )

    # Verify file isn't HTML error page
    with open(filepath, 'r') as f:
        first_line = f.readline()
        if first_line.startswith('<!doctype') or first_line.startswith('<html'):
            raise DataIntegrityError(
                f"File {filepath} is HTML, not CSV.\n"
                f"This indicates a failed download (likely API auth issue).\n"
                f"Delete file and re-download manually via browser."
            )

    # Load with explicit error handling
    try:
        df = pd.read_csv(filepath, index_col=0)
    except pd.errors.ParserError as e:
        raise DataIntegrityError(
            f"Failed to parse {filepath}: {e}\n"
            f"File may be corrupted. Re-download recommended."
        )

    # Validate biological constraints
    if 'ModelID' not in df.columns:
        raise ValueError(
            f"Expected 'ModelID' column in expression data, found: {df.columns[:5]}"
        )

    # Check for log2(TPM+1) range (should be 0-15)
    gene_cols = [c for c in df.columns if '(' in c]  # Gene columns have Entrez IDs
    expr_values = df[gene_cols].values.flatten()
    expr_values = expr_values[~np.isnan(expr_values)]

    if expr_values.min() < -1 or expr_values.max() > 20:
        raise ValueError(
            f"Expression values outside expected log2(TPM+1) range [0, 15]:\n"
            f"  Min: {expr_values.min():.2f}\n"
            f"  Max: {expr_values.max():.2f}\n"
            f"Check data normalization or file format."
        )

    return df
```

### 5. Configuration Management

**Academic**: Magic numbers everywhere
**Production**: Explicit, versioned configuration

```python
# BAD
threshold = 0.7  # Why 0.7? Who knows!
bins = 3  # Because I said so

# GOOD
from pydantic import BaseModel, Field
from typing import Literal

class PreprocessingConfig(BaseModel):
    """
    Immutable configuration for data preprocessing.

    All parameters have physical/biological justification.
    Changes to this config = new model version.
    """

    # Discretization
    num_states: Literal[3] = Field(
        default=3,
        description=(
            "Number of discrete states for CategoricalNodes. "
            "3 = {low, medium, high} provides balance between "
            "resolution and TSU hardware qubit requirements."
        )
    )

    discretization_method: Literal["tertile", "quantile", "kmeans"] = Field(
        default="tertile",
        description=(
            "Tertile binning: 33rd/67th percentiles. "
            "Ensures equal sample sizes per state, "
            "which is optimal for thermodynamic sampling."
        )
    )

    # Cell line stratification
    sensitive_percentile: float = Field(
        default=33.0,
        ge=0.0,
        le=50.0,
        description=(
            "IC50 percentile threshold for 'sensitive' group. "
            "33rd percentile = top third most sensitive. "
            "Biological justification: Clear responders vs non-responders."
        )
    )

    resistant_percentile: float = Field(
        default=67.0,
        ge=50.0,
        le=100.0,
        description="IC50 percentile for 'resistant' group (bottom third)."
    )

    # Quality control
    min_expression_value: float = Field(
        default=0.0,
        description="Minimum log2(TPM+1) value (physics: can't have negative expression)"
    )

    max_expression_value: float = Field(
        default=15.0,
        description="Maximum expected log2(TPM+1) for QC (biology: ~30,000 TPM max)"
    )

    methylation_na_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum fraction of NaN values allowed in methylation data. "
            "0.5 = drop genes with >50% missing coverage."
        )
    )

    class Config:
        frozen = True  # Immutable

# Usage
config = PreprocessingConfig()
print(config.json(indent=2))  # Serialize for provenance
```

---

## Code Architecture Principles

### Separation of Concerns

**Bioinformatics pattern** (BAD):
```python
# monolithic_analysis.py (2000 lines)
# - Load data
# - Preprocess
# - Run analysis
# - Make plots
# - Everything coupled together
```

**Device architecture** (GOOD):
```
core/
├── data_loader.py      # Pure data I/O (no analysis)
├── preprocessing.py    # Deterministic transformations
├── thrml_model.py      # Energy function specification
├── inference.py        # Thermodynamic sampling
├── validation.py       # Physics-based validation
└── provenance.py       # Audit trail management

Each module:
- Single responsibility
- Explicit inputs/outputs
- Comprehensive error handling
- Unit testable
- Hardware-deployable
```

### Type Safety

```python
# BAD (stringly-typed chaos)
def load_data(filename):  # What type? Who knows!
    return pd.read_csv(filename)

# GOOD (explicit contracts)
from typing import Protocol
from pathlib import Path

class CellLineData(Protocol):
    """Interface for multi-omics cell line data."""
    model_id: str
    stripped_name: str
    expression: dict[str, float]  # Gene -> log2(TPM+1)
    methylation: dict[str, float]  # Gene -> β-value
    tissue: str

def load_expression(
    filepath: Path,
    model_csv: Path,
    required_genes: list[str]
) -> dict[str, CellLineData]:
    """
    Load expression data with cell line metadata.

    Args:
        filepath: Path to CCLE_expression_TPM.csv
        model_csv: Path to Model.csv (for ID mapping)
        required_genes: List of HGNC gene symbols to extract

    Returns:
        Dictionary mapping ModelID -> CellLineData

    Raises:
        FileNotFoundError: Required files missing
        DataIntegrityError: File format invalid
        ValueError: Required genes not found
    """
    ...
```

### Hardware-Aware Data Structures

```python
# Academic: Use whatever pandas gives you
expr_df = pd.read_csv("data.csv")  # Dense matrix, lots of NaNs

# Production: Optimize for TSU hardware
from dataclasses import dataclass
import numpy as np

@dataclass
class DiscretizedGeneNetwork:
    """
    Compact representation optimized for TSU deployment.

    Design rationale:
    - Fixed-size arrays (no dynamic allocation in hardware)
    - Discrete states map directly to qubit registers
    - Sparse factor graph reduces coupling hardware
    """
    num_genes: int
    num_samples: int
    num_states: int = 3  # {0, 1, 2}

    # Gene expression: [samples × genes] discrete states
    expression: np.ndarray  # dtype=uint8 (not float!)

    # Methylation: [samples × genes] discrete states
    methylation: np.ndarray  # dtype=uint8

    # Metadata for provenance
    gene_names: list[str]
    sample_ids: list[str]

    # Thresholds for reverse mapping (if needed)
    expression_thresholds: np.ndarray  # [genes × 2] (low/mid, mid/high)
    methylation_thresholds: np.ndarray

    def __post_init__(self):
        """Validate hardware constraints."""
        assert self.expression.dtype == np.uint8
        assert self.methylation.dtype == np.uint8
        assert self.expression.max() < self.num_states
        assert self.methylation.max() < self.num_states

    def to_hardware_format(self) -> bytes:
        """
        Serialize to TSU-compatible binary format.

        Future: This will be the actual format sent to hardware.
        """
        # Header: magic number, version, dimensions
        header = struct.pack(
            '<4sHHHH',  # Little-endian, 4-char magic, 4 uint16
            b'THRM',
            1,  # Version
            self.num_genes,
            self.num_samples,
            self.num_states
        )

        # Data: Packed uint8 arrays
        data = self.expression.tobytes() + self.methylation.tobytes()

        return header + data
```

---

## Testing Philosophy

### Not Academic "Validation"

**Academic**:
- "We tested on 5 cell lines"
- "Results look reasonable"
- "p < 0.05, success!"

**Production**:
- **Unit tests**: Every function has explicit test cases
- **Integration tests**: End-to-end pipeline reproducibility
- **Physics tests**: Thermodynamic constraints validated
- **Regression tests**: Bit-exact results on reference data
- **Stress tests**: Edge cases, malformed inputs, hardware limits

```python
# tests/test_thermodynamics.py
import pytest
import numpy as np
from core.inference import compute_free_energy_difference

class TestThermodynamicConstraints:
    """Validate physical consistency of energy calculations."""

    def test_methylation_expression_causality(self):
        """
        Known biology: Methylation → Expression (not reverse).

        Physical test: ΔF(E|M) - ΔF(M|E) < 0
        """
        # Synthetic data: High methylation → Low expression
        M = np.array([2, 2, 2, 0, 0, 0])  # High, High, High, Low, Low, Low
        E = np.array([0, 0, 0, 2, 2, 2])  # Low, Low, Low, High, High, High

        delta_F = compute_free_energy_difference(M, E)

        assert delta_F < 0, (
            f"Methylation → Expression should be favored (ΔF < 0), "
            f"got ΔF = {delta_F:.3f}"
        )
        assert delta_F < -0.5, "Causal signal too weak for clinical use"

    def test_symmetric_variables_zero_delta_F(self):
        """
        If X and Y are identically distributed, ΔF should be ~0.

        Physics: No thermodynamic preference for either direction.
        """
        rng = np.random.default_rng(42)
        X = rng.choice([0, 1, 2], size=1000)
        Y = rng.permutation(X)  # Same distribution, different order

        delta_F = compute_free_energy_difference(X, Y)

        assert abs(delta_F) < 0.1, (
            f"Symmetric variables should have ΔF ≈ 0, got {delta_F:.3f}"
        )

    def test_energy_minimization_converges(self):
        """
        Thermodynamic sampling should reduce energy over time.

        Physics: System should evolve toward lower energy states.
        """
        from core.thrml_model import build_model

        model = build_model(num_genes=5, num_samples=100)
        initial_energy = model.compute_energy()

        # Run sampling
        model.sample(num_steps=10000)
        final_energy = model.compute_energy()

        assert final_energy < initial_energy, (
            "Thermodynamic sampling failed to minimize energy"
        )
```

---

## Documentation as Product Specification

**Academic**: "See paper for details"
**Production**: "Code is self-documenting AND we have comprehensive specs"

Every module should have:

1. **Physical/biological justification** for algorithms
2. **Hardware mapping** for data structures
3. **Failure modes** and error handling
4. **Provenance** tracking for regulatory compliance
5. **Performance characteristics** (time/space complexity + hardware requirements)

Example:
```python
def discretize_tertiles(
    continuous_values: np.ndarray,
    labels: list[str] = ['low', 'medium', 'high']
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Discretize continuous biological measurements to 3 states.

    Physical Justification:
        Tertile binning ensures equal sample sizes per state, which:
        - Maximizes statistical power for thermodynamic sampling
        - Balances hardware resource allocation (equal qubit occupancy)
        - Matches biological intuition (low/medium/high expression)

    Hardware Mapping:
        Output states {0, 1, 2} map directly to 2-qubit registers:
            00 = low (0)
            01 = medium (1)
            10 = high (2)
            11 = unused (reserved for future 4-state models)

    Alternative Approaches Considered:
        - K-means (k=3): Non-deterministic, hardware-unfriendly
        - Equal-width bins: Unbalanced sample sizes, poor statistics
        - Quantile (k>3): More hardware qubits required

    Args:
        continuous_values: Array of measurements (e.g., log2(TPM+1) or β-values)
        labels: Human-readable state names (for provenance only)

    Returns:
        (discrete_states, thresholds)
        discrete_states: Array of {0, 1, 2} matching input length
        thresholds: (t1, t2) where t1=33rd percentile, t2=67th percentile

    Raises:
        ValueError: If input has <3 unique values (can't create 3 bins)

    Performance:
        Time: O(n log n) for percentile calculation
        Space: O(1) auxiliary (in-place compatible)
        Hardware: 2 qubits per variable

    Provenance:
        Thresholds are deterministic functions of input data.
        Same data → same thresholds → same discretization.

    Example:
        >>> expr = np.array([0.5, 2.3, 5.1, 1.2, 8.9, 3.4])
        >>> discrete, (t1, t2) = discretize_tertiles(expr)
        >>> discrete
        array([0, 1, 1, 0, 2, 1])  # dtype=uint8
        >>> t1, t2
        (1.75, 4.25)  # 33rd and 67th percentiles
    """
    ...
```

---

## Deployment Vision

### Near-term (Hackathon Demo)
```
Laptop → Python → THRML library → Software TSU simulation
```

### Mid-term (Clinical Validation)
```
Server → Optimized C++ → THRML compiler → GPU-accelerated sampling
```

### Long-term (Biomedical Device)
```
Embedded system → ASIC firmware → TSU hardware → Real-time inference
                                                    ↓
                                          Clinical decision support
```

**Engineering principle**: Every line of code should be **forward-compatible** with hardware deployment.

---

## Summary: Why This Matters

### Academic Prototype
- Goal: Publish paper
- Timeline: Until graduation
- Reproducibility: "Trust me"
- Validation: p-values
- Deployment: Never

### Production Biomedical Device
- Goal: **Save lives**
- Timeline: **Decades of clinical use**
- Reproducibility: **Bit-exact, auditable**
- Validation: **Physical consistency + clinical trials**
- Deployment: **FDA-cleared hardware**

**We are building the latter.**

### Core Philosophy

1. **Physics, not statistics**: Thermodynamic truth, not p-values
2. **Hardware-aware**: Simulate TSU reality, not abstract math
3. **Deterministic**: Same input → same output, always
4. **Fail-fast**: Invalid states impossible, not just unlikely
5. **Auditable**: Full provenance, clinical-grade records
6. **Production-ready**: Code that will run on silicon, not just slides

---

**Start date**: November 16, 2024
**Engineering standard**: Medical device, not research prototype
**Physical basis**: Thermodynamic computing on TSU hardware
**Mission**: Demonstrate that **physics beats statistics** for biomedical AI

Let's build something real.
