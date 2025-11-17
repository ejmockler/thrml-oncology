# Quickstart Guide

**Production-ready thermodynamic causal inference for cancer research**

Last Updated: 2025-11-16

---

## 30-Second Start

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Verify THRML installation
python3 -c "import thrml; print(f'THRML v{thrml.__version__} ready')"

# 3. Download data (manual - see below)
# Files needed: CCLE_expression_TPM.csv, CCLE_RRBS_methylation.txt, Model.csv, GDSC data

# 4. Run preprocessing
python3 core/data_loader.py \
  --expression data/raw/ccle/CCLE_expression_TPM.csv \
  --methylation data/raw/ccle/CCLE_RRBS_methylation.txt \
  --model data/raw/ccle/Model.csv \
  --gdsc data/raw/gdsc/GDSC1_fitted_dose_response.xlsx \
  --output-dir data/processed/
```

---

## What This Project Does

**Goal**: Predict alternative drugs for erlotinib-resistant lung cancer using thermodynamic causal inference.

**Approach**:
1. Load multi-omics data (expression + methylation) from CCLE
2. Stratify cell lines by erlotinib response (IC50 from GDSC)
3. Discretize continuous values → {low, medium, high} states
4. Build Energy-Based Models using THRML (Extropic's thermodynamic computing library)
5. Infer causal networks via free energy discrimination
6. Identify bypass mechanisms (network changes in resistant cells)
7. Predict drugs targeting bypasses
8. Validate predictions with GDSC IC50 data

**Current Status**: ✅ Data preprocessing pipeline complete and validated

---

## Data Acquisition

### Required Files

**CCLE (Cancer Cell Line Encyclopedia)**:
- `CCLE_expression_TPM.csv` (~543 MB) - RNA-seq expression data
- `CCLE_RRBS_methylation.txt` (~149 MB) - DNA methylation data
- `Model.csv` (~0.7 MB) - Cell line metadata

**GDSC (Genomics of Drug Sensitivity in Cancer)**:
- `GDSC1_fitted_dose_response.xlsx` (~3.4 MB) - Drug response (IC50)

### Download Instructions

**DepMap Portal** (CCLE data):
1. Visit https://depmap.org/portal/download/
2. Search for files (may require free account):
   - Expression: "CCLE_expression_TPM.csv" or "OmicsExpressionProteinCodingGenesTPMLogp1"
   - Methylation: "CCLE_RRBS_methylation.txt" or "CCLE_RRBS_tss1kb"
   - Model: "Model.csv"
3. Save to `data/raw/ccle/`

**GDSC Portal**:
1. Visit https://www.cancerrxgene.org/downloads/bulk_download
2. Download "GDSC1 fitted dose response"
3. Save to `data/raw/gdsc/GDSC1_fitted_dose_response.xlsx`

---

## Production Pipeline

### Step 1: Data Preprocessing

**Status**: ✅ COMPLETE - Production-grade implementation

```bash
source venv/bin/activate

python3 core/data_loader.py \
  --expression data/raw/ccle/CCLE_expression_TPM.csv \
  --methylation data/raw/ccle/CCLE_RRBS_methylation.txt \
  --model data/raw/ccle/Model.csv \
  --gdsc data/raw/gdsc/GDSC1_fitted_dose_response.xlsx \
  --output-dir data/processed/
```

**Output**:
- `data/processed/sensitive_discretized.pkl` - Sensitive cell line data (60 samples × 12 genes)
- `data/processed/resistant_discretized.pkl` - Resistant cell line data (60 samples × 12 genes)
- `data/processed/preprocessing_report.txt` - Audit trail

**What it does**:
1. Loads 1,754 expression profiles (19,215 genes)
2. Loads 843 methylation profiles (21,338 TSS regions)
3. Aligns via Model.csv → 832 overlapping cell lines
4. Filters for erlotinib IC50 data → 181 cell lines
5. Stratifies by IC50 percentiles (p33/p67) → 60 sensitive + 60 resistant
6. Extracts 12 EGFR pathway genes with both expression + methylation
7. Discretizes to 3 states (tertile binning)
8. Validates data integrity (checksums, ranges, sample alignment)

See `core/DATA_LOADER_README.md` for technical details.

### Step 2: THRML Model Building

**Status**: 🚧 TO IMPLEMENT

Build hardware-aware Energy-Based Models:
- Methylation → Expression anti-concordance factors
- Expression ↔ Expression INDRA pathway priors
- CategoricalNodes (3 states per gene, uint8 dtype for TSU deployment)

See `THRML_API_VERIFIED.md` for verified v0.1.3 API patterns.

### Step 3: Causal Inference

**Status**: 🚧 TO IMPLEMENT

Free energy discrimination for pairwise causality:
- Sample from forward model (G1 → G2)
- Sample from backward model (G2 → G1)
- Compute ΔF = F_backward - F_forward
- Decision: |ΔF| > 1.0 k_B T determines direction

See `docs/INFERENCE_API.md` for interface specification.

### Step 4: Validation

**Status**: 🚧 TO IMPLEMENT

Physics-based validation framework:
- Detailed balance verification
- Ergodicity checks
- Free energy convergence
- IC50 validation (precision metrics)

See `core/VALIDATION_README.md` for methodology.

---

## File Structure

```
thrml-cancer-decision-support/
├── README.md                           # Project overview
├── QUICKSTART.md                       # This file
├── ENGINEERING_PHILOSOPHY.md           # Medical-grade code standards
├── RIGOROUS_METHODOLOGY.md            # Complete methodology
├── ENVIRONMENT_SETUP.md               # Detailed setup guide
├── DOCUMENTATION_INDEX.md             # Navigation hub
│
├── data/
│   ├── raw/ccle/                      # Downloaded CCLE data
│   ├── raw/gdsc/                      # Downloaded GDSC data
│   ├── processed/                     # Preprocessed .pkl files
│   ├── DATA_SOURCES.md                # Data provenance
│   ├── DATA_INVENTORY.md              # What data we have
│   └── VERSION_COMPATIBILITY_ANALYSIS.md  # 2018 vs 2025 data alignment
│
├── core/
│   ├── data_loader.py                 # ✅ Production preprocessing (820 lines)
│   ├── DATA_LOADER_README.md          # Technical documentation
│   ├── thrml_model.py                 # 🚧 THRML model construction
│   ├── inference.py                   # 🚧 Causal inference
│   ├── validation.py                  # 🚧 Validation framework
│   └── VALIDATION_README.md           # Validation specification
│
├── docs/
│   ├── INFERENCE_API.md               # Inference interface
│   └── INFERENCE_QUICKSTART.md        # Inference guide
│
├── scripts/
│   ├── 01_download_data.sh            # Data download automation
│   └── 02_run_inference.py            # 🚧 Main pipeline
│
├── THRML_COMPREHENSIVE_DOCUMENTATION.md  # THRML v0.1.3 reference
├── THRML_API_VERIFIED.md                 # Verified API patterns
└── requirements.txt                      # Dependencies (THRML v0.1.3)
```

---

## Key Technical Decisions

### EGFR Pathway Genes (12 genes)
```python
['EGFR', 'KRAS', 'BRAF', 'PIK3CA', 'AKT1', 'MTOR',
 'MAP2K1', 'MAPK1', 'SOS1', 'GRB2', 'PTEN', 'NF1']
```
**Rationale**: Well-characterized erlotinib resistance mechanisms

### Discretization: Tertile Binning
- 0 = low (0-33rd percentile)
- 1 = medium (33-66th percentile)
- 2 = high (66-100th percentile)

**Rationale**: Balanced categories, hardware-compatible uint8, interpretable

### IC50 Stratification
- Sensitive: IC50 < p33 (2.331)
- Resistant: IC50 > p67 (3.037)
- Excluded: Intermediate responders

**Result**: 60 sensitive + 60 resistant cell lines (n=120 total)

### Data Provenance
- SHA-256 checksums for all input files
- Pydantic V2 immutable configuration
- Complete audit trail in preprocessing_report.txt

**Rationale**: Medical device standards - deterministic, auditable, fail-fast

---

## Next Implementation Steps

**Priority 1**: `core/thrml_model.py`
- Implement `build_categorical_nodes()`
- Implement `create_methylation_factors()`
- Implement `create_expression_factors()` with INDRA priors
- Implement `build_sampling_program()`

**Priority 2**: `core/inference.py`
- Implement `infer_pairwise_causality()`
- Implement `estimate_free_energy()`
- Implement `compare_networks()`

**Priority 3**: `core/validation.py`
- Implement `validate_detailed_balance()`
- Implement `check_ergodicity()`
- Implement `validate_ic50_predictions()`

**Priority 4**: `scripts/02_run_inference.py`
- Orchestrate full pipeline
- Generate results and figures

---

## Reference Documentation

### Quick References
- **Data preprocessing**: `core/DATA_LOADER_README.md`, `core/DATA_LOADER_QUICK_REFERENCE.md`
- **THRML API**: `THRML_API_VERIFIED.md` (verified against v0.1.3)
- **Causal inference**: `docs/INFERENCE_API.md`, `docs/INFERENCE_QUICKSTART.md`
- **Validation**: `core/VALIDATION_README.md`

### Comprehensive Guides
- **Engineering standards**: `ENGINEERING_PHILOSOPHY.md`
- **Complete methodology**: `RIGOROUS_METHODOLOGY.md`
- **THRML deep dive**: `THRML_COMPREHENSIVE_DOCUMENTATION.md`
- **Environment setup**: `ENVIRONMENT_SETUP.md`

### Navigation
- **Documentation hub**: `DOCUMENTATION_INDEX.md`

---

## Verification Checklist

**Data Pipeline**:
- [x] Expression data loaded (1,754 cell lines × 19,215 genes)
- [x] Methylation data loaded (843 cell lines × 21,338 regions)
- [x] Cell lines aligned via Model.csv (832 overlapping)
- [x] IC50 stratification (60 sensitive + 60 resistant)
- [x] EGFR pathway genes extracted (12/12 with E+M data)
- [x] Tertile discretization applied
- [x] Output files generated (.pkl + report.txt)
- [x] SHA-256 provenance tracked

**THRML Model** (TO DO):
- [ ] CategoricalNodes created for all genes
- [ ] Methylation → Expression factors built
- [ ] INDRA pathway priors integrated
- [ ] Block Gibbs sampling configured
- [ ] Model samples successfully

**Inference** (TO DO):
- [ ] Pairwise causality tested
- [ ] Free energies estimated
- [ ] Networks compared (sensitive vs resistant)
- [ ] Bypass mechanisms identified

**Validation** (TO DO):
- [ ] Detailed balance verified
- [ ] Ergodicity confirmed
- [ ] Drug predictions generated
- [ ] IC50 validation complete
- [ ] Precision > 40% achieved

---

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'thrml'"
```bash
# Ensure venv activated
source venv/bin/activate

# Reinstall
pip install thrml
```

**Issue**: Data files not found
```bash
# Check paths
ls -lh data/raw/ccle/
ls -lh data/raw/gdsc/

# Re-download if missing (see Data Acquisition section)
```

**Issue**: Preprocessing fails with "DataIntegrityError"
- Check input file formats match expected (CSV, TSV, XLSX)
- Verify file sizes (~543 MB expression, ~149 MB methylation)
- Check preprocessing_output.log for detailed error

**Issue**: Out of memory during preprocessing
```bash
# Reduce gene set (edit EGFR_PATHWAY_GENES in core/data_loader.py)
# Or use machine with more RAM (16GB+ recommended)
```

---

## Performance Expectations

**Data Preprocessing**:
- Time: ~30 seconds (MacBook Pro M2)
- Memory: ~4 GB peak
- Output: 2 × ~50 KB .pkl files

**THRML Sampling** (projected):
- Time: ~5-10 seconds per gene pair (GPU)
- Time: ~30-60 seconds per gene pair (CPU)
- Full network (66 pairs): ~5-40 minutes

**TSU Hardware** (projected):
- Time: ~11 ms per gene pair
- Speedup: ~600× vs GPU
- Energy: ~100× more efficient

---

## Support & References

**Environment issues**: See `ENVIRONMENT_SETUP.md`

**THRML API questions**: See `THRML_API_VERIFIED.md` (verified v0.1.3)

**Data issues**: See `data/README.md`, `data/VERSION_COMPATIBILITY_ANALYSIS.md`

**Methodology questions**: See `RIGOROUS_METHODOLOGY.md`

**Project overview**: See `README.md`

**All documentation**: See `DOCUMENTATION_INDEX.md`

---

## Citation

Thermodynamic Causal Inference for Cancer Drug Resistance
Uses Extropic's THRML library (v0.1.3)
Multi-omics data from DepMap/CCLE and GDSC

---

**Status**: Data pipeline complete. THRML model implementation in progress.

**Last Updated**: 2025-11-16
