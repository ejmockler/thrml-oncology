# Quick Start Guide - XTR-0 Hackathon

**Last Updated**: November 16, 2024

---

## 📚 Navigation
- **[← Back to Index](DOCUMENTATION_INDEX.md)** - All documentation
- **[README.md](README.md)** - Project overview
- **[RIGOROUS_METHODOLOGY.md](RIGOROUS_METHODOLOGY.md)** - Detailed methodology
- **[DATA_DOWNLOAD_SUMMARY.md](data/DATA_DOWNLOAD_SUMMARY.md)** - Data acquisition guide

---

## TL;DR - What You Need to Do Now

### ⚠️ CRITICAL: Manual Data Download Required

The automated download script encountered API authentication issues with DepMap. **You need to manually download 2 files**:

```bash
# 1. Visit DepMap Portal
open https://depmap.org/portal/download/

# 2. Download these files (search by name):
#    - "OmicsExpressionProteinCodingGenesTPMLogp1.csv" (~200 MB)
#    - "CCLE_RRBS_tss1kb" (methylation data)
#
# 3. Save them as:
#    data/raw/ccle/CCLE_expression_TPM.csv
#    data/raw/ccle/CCLE_RRBS_methylation.txt
```

**Alternative**: Use the data from the Figshare link in `DATA_DOWNLOAD_SUMMARY.md`

### ✓ What's Already Done

1. **GDSC Data Downloaded** (3 files, 295 MB total)
   - IC50 drug response data ✓
   - Cell line expression data ✓

2. **Complete Methodology Documented**
   - `RIGOROUS_METHODOLOGY.md` (1200+ lines) ✓
   - Every step from data → validated predictions ✓

3. **Infrastructure Ready**
   - Download scripts ✓
   - Data directory structure ✓
   - Documentation ✓

---

## 30-Second Overview

**What This Project Does**:
- Predicts which drugs will work for erlotinib-resistant cancer cells
- Uses thermodynamic causal inference (new approach)
- Validates predictions with real IC50 data

**How It Works**:
1. Load CCLE expression + methylation data
2. Discretize to {low, med, high} (THRML requirement)
3. Build energy-based model with INDRA priors
4. Infer causal networks (sensitive vs resistant cells)
5. Find bypass mechanisms (changed edges)
6. Predict drugs that target bypasses
7. Validate with GDSC IC50 data

**Expected Result**: 40-50% precision (vs 15% baseline)

---

## File Organization

```
thrml-cancer-decision-support/
├── README.md                            ← Original hackathon brief
├── RIGOROUS_METHODOLOGY.md              ← ⭐ COMPLETE METHODOLOGY (read this!)
├── DATA_AND_METHODOLOGY_SUMMARY.md      ← Executive summary
├── QUICK_START.md                       ← This file
│
├── data/
│   ├── raw/
│   │   ├── ccle/
│   │   │   ├── CCLE_expression_TPM.csv       ← ⚠️ NEEDS MANUAL DOWNLOAD
│   │   │   ├── CCLE_RRBS_methylation.txt     ← ⚠️ NEEDS MANUAL DOWNLOAD
│   │   │   └── Model.csv                     ← ✓ Downloaded (15 KB)
│   │   └── gdsc/
│   │       ├── GDSC1_fitted_dose_response.xlsx  ← ✓ Downloaded (3.3 MB)
│   │       ├── GDSC2_fitted_dose_response.xlsx  ← ✓ Downloaded (48 KB)
│   │       └── Cell_line_RMA_proc_basalExp.txt  ← ✓ Downloaded (292 MB)
│   │
│   ├── processed/                  ← Output from preprocessing
│   └── DATA_SOURCES.md             ← Data provenance documentation
│
├── scripts/
│   ├── 00_test_environment.py      ← Verify THRML works
│   ├── 01_download_data.sh         ← Automated download (partial)
│   ├── 02_run_inference.py         ← Main pipeline (TO IMPLEMENT)
│   └── 03_analyze_results.py       ← Validation & figures (TO IMPLEMENT)
│
├── core/
│   ├── data_loader.py              ← Preprocessing functions (TO IMPLEMENT)
│   ├── thrml_model.py              ← THRML model construction (partial)
│   ├── indra_client.py             ← INDRA API wrapper (partial)
│   ├── inference.py                ← Causal inference (TO IMPLEMENT)
│   └── validation.py               ← IC50 validation (TO IMPLEMENT)
│
└── results/                        ← Output directory for demo results
```

---

## Implementation Priority

### Phase 1: Data Acquisition (NOW - 30 min)
```bash
# Manual download from DepMap portal
# Files: CCLE expression + methylation
# See "CRITICAL" section above
```

### Phase 2: Core Implementation (4-6 hours)
```python
# 1. data_loader.py (2 hours)
#    - Load CCLE/GDSC data
#    - Stratify cell lines (sensitive/resistant)
#    - Discretize to categorical {0, 1, 2}

# 2. thrml_model.py (1 hour)
#    - Build CategoricalNodes
#    - Create factors with INDRA priors
#    - Set up Block Gibbs sampling

# 3. inference.py (2 hours)
#    - Free energy estimation
#    - Pairwise causal direction testing
#    - Network comparison

# 4. validation.py (1 hour)
#    - Drug prediction from network changes
#    - IC50 validation
#    - Statistical tests
```

### Phase 3: Testing (1 hour)
```bash
# Quick test (5 genes, synthetic data)
python scripts/02_run_inference.py --quick-test

# If passes, proceed to full run
```

### Phase 4: Full Run (2-4 hours)
```bash
# 15 genes, 1000 samples
python scripts/02_run_inference.py \
  --genes 15 \
  --samples 1000 \
  --output-dir results/full_run
```

### Phase 5: Analysis & Presentation (2 hours)
```bash
# Generate figures and report
python scripts/03_analyze_results.py \
  --results results/full_run \
  --make-figures \
  --make-report
```

---

## Key Technical Decisions Already Made

### 1. Gene Set: EGFR Pathway (15-20 genes)
```python
EGFR_PATHWAY_GENES = [
    'EGFR', 'KRAS', 'BRAF', 'PIK3CA', 'PTEN',
    'AKT1', 'MTOR', 'MEK1', 'ERK2', 'MET',
    'HER2', 'HER3', 'SOS1', 'GRB2', 'TP53'
]
```
**Rationale**: Well-characterized pathway, erlotinib resistance mechanisms known

### 2. Discretization: Tertile-Based Binning
```python
# Low: 0-33rd percentile
# Med: 33-66th percentile
# High: 66-100th percentile
```
**Rationale**: Balanced categories, interpretable thresholds

### 3. Cell Line Stratification: IC50 Quartiles
```python
# Sensitive: IC50 < 25th percentile
# Resistant: IC50 > 75th percentile
# Exclude: Intermediate responders
```
**Rationale**: Clear phenotype separation, ~50-100 lines per group

### 4. THRML Model: CategoricalEBMFactor
```python
# Methylation → Expression: Negative correlation
# Expression ↔ Expression: INDRA pathway priors
# Sampling: Block Gibbs, 500 warmup + 1000 samples
```
**Rationale**: Matches THRML API capabilities, principled priors

### 5. Causal Test: Free Energy Discrimination
```python
# ΔF = F_reverse - F_forward
# Threshold: 1.0 (1 k_B T)
# Confidence: |ΔF| / σ_ΔF > 2.0
```
**Rationale**: Thermodynamically principled, statistically rigorous

### 6. Validation: Direct IC50 Check
```python
# Precision = (# predicted drugs with IC50 < median) / (# predicted)
# Baseline: 15% (random selection)
# Target: >40% (2.7× enrichment)
```
**Rationale**: Ground truth available, clear success metric

---

## Critical Dependencies

### Required Python Packages
```bash
pip install thrml jax jaxlib          # TSU framework
pip install numpy pandas scipy        # Data processing
pip install matplotlib seaborn         # Visualization
pip install networkx                   # Network graphs
pip install requests                   # INDRA API
pip install openpyxl                   # Excel reading
```

### External APIs
- **INDRA**: `http://api.indra.bio:8000` (no key required)
- **DepMap Portal**: https://depmap.org/portal/download/ (may require login)

---

## Success Metrics

### Minimum Viable Demo
- [x] Environment works (THRML imports successfully)
- [ ] Data loaded and preprocessed
- [ ] Inference completes (even on 5 genes)
- [ ] Network comparison shows ≥1 changed edge
- [ ] Drug prediction generates ≥1 testable drug
- [ ] IC50 validation shows precision >30%

**Probability**: 70% (if CCLE data acquired)

### Competitive Demo
- All of Minimum Viable +
- [ ] Full 15 genes × 1000 samples
- [ ] Precision >40%, enrichment >2.5×
- [ ] P-value < 0.05 (statistically significant)
- [ ] ≥3 novel bypass mechanisms identified
- [ ] Clean visualizations ready

**Probability**: 40%

### Winning Demo
- All of Competitive +
- [ ] Precision >50%
- [ ] Novel biological insight (unknown mechanism)
- [ ] TSU hardware mapping detailed
- [ ] Publication-quality methodology

**Probability**: 15%

---

## Fallback Plans

### If CCLE Data Can't Be Acquired
```bash
# Use synthetic data with known ground truth
python scripts/02_run_inference.py --synthetic-data

# Pros: Can demonstrate algorithm works
# Cons: No real IC50 validation, limited impact
```

### If THRML Has Bugs
```python
# Use NumPy fallback sampler (slower but correct)
from core.fallback_sampler import NumpyGibbsSampler

# Pros: Same math, still shows principle
# Cons: 10× slower, less impressive for TSU demo
```

### If Inference Too Slow
```bash
# Reduce scope: 10 genes, 500 samples
python scripts/02_run_inference.py --genes 10 --samples 500

# Or: Focus on methodology demonstration
# Show quick test works, extrapolate to full scale
```

### If Validation Precision Low
**Still Win By**:
- Methodology is sound (novel approach)
- Validatable in principle (ground truth exists)
- TSU advantage independent of precision (speedup is algorithmic)
- Power analysis: "Need 25+ genes for 50%"

---

## Timeline Estimate

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| 1 | Manual CCLE download | 30 min | 0:30 |
| 2 | Implement data_loader.py | 2 hours | 2:30 |
| 3 | Implement thrml_model.py | 1 hour | 3:30 |
| 4 | Implement inference.py | 2 hours | 5:30 |
| 5 | Implement validation.py | 1 hour | 6:30 |
| 6 | Quick test & debug | 1 hour | 7:30 |
| 7 | Full inference run | 2-4 hours | 9:30-11:30 |
| 8 | Analysis & visualization | 1 hour | 10:30-12:30 |
| 9 | Report & presentation | 1 hour | 11:30-13:30 |

**Total**: 11-13 hours (with 3-5 hours of parallel compute time)

**For 8-Hour Hackathon**: Must start with implementations ready OR use quick test only

---

## What to Read First

1. **This file (QUICK_START.md)**: Overview and priorities
2. **RIGOROUS_METHODOLOGY.md Sections 1-2**: Preprocessing + THRML model
3. **RIGOROUS_METHODOLOGY.md Section 5.1**: Full pipeline script template
4. **DATA_DOWNLOAD_SUMMARY.md**: Data acquisition troubleshooting

**Time Investment**: 1 hour reading → saves 3 hours implementation

---

## Next Immediate Actions

```bash
# 1. Download CCLE data manually (30 min)
#    See "CRITICAL" section at top

# 2. Verify data loaded correctly
python -c "import pandas as pd; \
  expr = pd.read_csv('data/raw/ccle/CCLE_expression_TPM.csv'); \
  print(f'Expression: {expr.shape}'); \
  meth = pd.read_csv('data/raw/ccle/CCLE_RRBS_methylation.txt', sep='\t'); \
  print(f'Methylation: {meth.shape}')"

# 3. Test THRML environment
python scripts/00_test_environment.py

# 4. Start implementation (see Phase 2 above)
```

---

## Questions? Check These Docs

- **Data issues**: `DATA_DOWNLOAD_SUMMARY.md`
- **Methodology details**: `RIGOROUS_METHODOLOGY.md`
- **Overall project**: `README.md`
- **Quick reference**: This file

---

## Contact & Support

- **DepMap Help**: depmap@broadinstitute.org
- **THRML Docs**: https://docs.extropic.ai/ (if available)
- **This Project**: See commit history for iterative development

**Last Updated**: November 16, 2024 22:20 UTC
